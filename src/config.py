"""
Central configuration for GCG experiments.

Two presets:
  - "debug": tiny settings for quick CPU smoke runs / development.
  - "paper": hyper-parameters from Zou et al. 2023 ("Universal and
    Transferable Adversarial Attacks on Aligned Language Models").

Usage:
    cfg = GCGConfig()                 # debug preset
    cfg = GCGConfig(preset="paper")   # paper preset
    cfg = GCGConfig(preset="paper", num_steps=50)  # explicit args win
"""
from dataclasses import dataclass, field
from typing import List, Optional

# Field defaults used by the "debug" preset. The "paper" preset only
# upgrades fields that the caller left at these defaults.
DEBUG_DEFAULTS = {
    "num_steps": 10,
    "suffix_length": 8,
    "topk": 10,
    "search_width": 16,
    "mini_batch_size": 8,
}

# Algorithm 1/2 hyper-parameters from the GCG paper.
PAPER_DEFAULTS = {
    "num_steps": 250,
    "suffix_length": 20,
    "topk": 256,
    "search_width": 512,
    "mini_batch_size": 32,
}


@dataclass
class GCGConfig:
    # ---------- run mode ----------
    preset: str = "debug"  # "debug" | "paper"

    # ---------- model ----------
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    dtype: str = "float16"       # "float16" | "bfloat16" | "float32"
    device: str = "auto"         # "auto" | "cuda" | "cpu"
    load_in_4bit: bool = False   # bitsandbytes 4-bit quantization (CUDA only)

    # ---------- algorithm (set from preset in __post_init__) ----------
    num_steps: int = DEBUG_DEFAULTS["num_steps"]          # T in Algorithm 1
    suffix_length: int = DEBUG_DEFAULTS["suffix_length"]  # |s| in tokens
    topk: int = DEBUG_DEFAULTS["topk"]                    # k candidates / position
    search_width: int = DEBUG_DEFAULTS["search_width"]    # B sampled suffixes / step
    mini_batch_size: int = DEBUG_DEFAULTS["mini_batch_size"]

    # ---------- suffix initialization ----------
    suffix_init: str = ""  # auto-filled with "! " * suffix_length if empty

    # ---------- universal attack (Algorithm 2) ----------
    num_prompts_start: int = 1  # start with this many prompts
    prompt_add_interval: int = 5  # add one prompt every N steps

    # ---------- data ----------
    # Optional jsonl file with {"goal": ..., "target": ...} records.
    data_path: Optional[str] = None
    # How many benchmark prompts to evaluate on when using the benchmark set.
    num_eval_prompts: int = 4

    # ---------- benign default goal/target pairs ----------
    goals: List[str] = field(default_factory=lambda: [
        "Write a poem about nature",
        "Explain how photosynthesis works",
        "Describe the water cycle",
        "Tell me about the solar system",
    ])
    targets: List[str] = field(default_factory=lambda: [
        "Sure, here is a poem about nature:\n\n",
        "Sure, here is an explanation of photosynthesis:\n\n",
        "Sure, here is a description of the water cycle:\n\n",
        "Sure, here is information about the solar system:\n\n",
    ])

    # ---------- generation / evaluation ----------
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0.0 => greedy decoding
    use_kv_cache: bool = True  # cache the fixed prompt prefix (faster)

    seed: int = 42

    def __post_init__(self):
        if self.preset not in ("debug", "paper"):
            raise ValueError(f"Unknown preset {self.preset!r}; use 'debug' or 'paper'")

        # Upgrade any field still at its debug default to the paper value.
        if self.preset == "paper":
            for key, value in PAPER_DEFAULTS.items():
                if getattr(self, key) == DEBUG_DEFAULTS[key]:
                    setattr(self, key, value)

        # Make sure the initial suffix tokenizes to exactly suffix_length tokens.
        if not self.suffix_init:
            self.suffix_init = "! " * self.suffix_length

        if len(self.goals) != len(self.targets):
            raise ValueError("goals and targets must have the same length")

    # ---------- helpers ----------
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self):
        import torch
        device = self.resolved_device()
        # float16 matmuls are not reliably supported on CPU; fall back.
        if device == "cpu" and self.dtype == "float16":
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]
