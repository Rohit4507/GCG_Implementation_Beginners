"""
Run provenance: what model was ACTUALLY used, and what the run cost.

Two jobs, both about honesty of the run records in `results/`:

1. `describe_model` — records the model that was actually loaded, not the one
   named in the config. Previously every run record wrote `cfg.model_id`, which
   printed `HuggingFaceTB/SmolLM2-360M-Instruct` even for `--tiny` runs on the
   offline random model. That is the kind of thing that ends up in a published
   results table and gets caught in review. The tell was quantitative: the tiny
   model's vocab is 512, so random guessing floors the loss at ln(512) ~= 6.238,
   and every recorded loss sat at ~6.25.

2. `RunAccounting` — computes the hardware-independent cost of a run (steps,
   forward passes) alongside wall-clock and money. "Cost per successful attack"
   is undefined at 0 successes; this module records that honestly instead of
   printing 0, which would read as "free attacks".

Used by every runner in `experiments/`.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

# Conventional name for the offline random-weight model built by src/tiny_model.py
TINY_MODEL_LABEL = "tiny(offline random weights)"


def describe_model(
    *,
    tiny: bool,
    model_id: str,
    tokenizer=None,
    seed: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> Dict:
    """
    Describe the model a run actually used.

    Args:
        tiny: True if the run used the offline random-weight tiny model.
        model_id: The HF id from the config (may be unused when tiny=True).
        tokenizer: The loaded tokenizer, used to read the real vocabulary size.
        seed: Tiny-model seed, when applicable.

    Returns a dict that is written verbatim into the run record.
    """
    if vocab_size is None and tokenizer is not None:
        try:
            vocab_size = len(tokenizer)
        except TypeError:  # pragma: no cover - unusual tokenizer implementations
            vocab_size = getattr(tokenizer, "vocab_size", None)

    if tiny:
        return {
            "model_label": f"{TINY_MODEL_LABEL}, seed={seed}" if seed is not None
            else TINY_MODEL_LABEL,
            "model_kind": "tiny_random",
            "hf_model_id": None,
            "random_weights": True,
            "seed": seed,
            "vocab_size": vocab_size,
            "note": (
                "Randomly-initialised model. It has no refusal behaviour to "
                "bypass, so any ASR or attack-success number from this run is "
                "meaningless as an attack result. Loss is floored at "
                f"ln(vocab) = ln({vocab_size}) = "
                f"{__import__('math').log(vocab_size):.3f}."
                if vocab_size
                else "Randomly-initialised model; attack results are meaningless."
            ),
        }

    return {
        "model_label": model_id,
        "model_kind": "hf_pretrained",
        "hf_model_id": model_id,
        "random_weights": False,
        "seed": seed,
        "vocab_size": vocab_size,
        "note": "Pretrained checkpoint loaded from the Hugging Face hub.",
    }


def loss_floor(vocab_size: Optional[int]) -> Optional[float]:
    """
    Cross-entropy of uniform random guessing over `vocab_size` tokens.

    A run whose loss hovers at this value is not learning anything; on the tiny
    model that is expected, on a real model it means the attack is failing.
    """
    if not vocab_size:
        return None
    import math
    return math.log(vocab_size)


@dataclass
class RunAccounting:
    """
    Compute cost of one experiment run.

    Hardware-independent units (`*_forwards`, `*_per_success`) are the ones worth
    comparing across runs; dollars depend on where you ran it.
    """

    device: str = ""
    # --- what the optimiser did ---
    steps_run: int = 0
    gradient_forwards: int = 0      # one backward pass per optimisation step
    candidate_forwards: int = 0     # forward passes spent scoring candidates
    # --- wall clock and money ---
    wall_clock_s: float = 0.0
    gpu_hourly_usd: float = 0.0     # 0.0 for free tiers (Kaggle / Colab)
    model_load_s: float = 0.0
    # --- outcome, needed for cost-per-success ---
    n_eval_pairs: int = 0
    n_successes: int = 0
    notes: str = ""

    @property
    def total_forwards(self) -> int:
        return self.gradient_forwards + self.candidate_forwards

    @property
    def seconds_per_step(self) -> Optional[float]:
        if not self.steps_run:
            return None
        return self.wall_clock_s / self.steps_run

    @property
    def estimated_cost_usd(self) -> float:
        return (self.wall_clock_s / 3600.0) * self.gpu_hourly_usd

    @property
    def cost_per_success_usd(self) -> Optional[float]:
        """
        None when there were no successes.

        Returning 0.0 here would read as "attack success is free", which is the
        opposite of what an ASR of 0% means. `undefined_rate` says why.
        """
        if self.n_successes <= 0:
            return None
        return self.estimated_cost_usd / self.n_successes

    @property
    def forwards_per_success(self) -> Optional[float]:
        if self.n_successes <= 0:
            return None
        return self.total_forwards / self.n_successes

    def as_dict(self) -> Dict:
        d = {
            "device": self.device,
            "steps_run": self.steps_run,
            "gradient_forwards": self.gradient_forwards,
            "candidate_forwards": self.candidate_forwards,
            "total_forwards": self.total_forwards,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "model_load_s": round(self.model_load_s, 3),
            "seconds_per_step": (
                round(self.seconds_per_step, 4) if self.seconds_per_step else None
            ),
            "gpu_hourly_usd": self.gpu_hourly_usd,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "n_eval_pairs": self.n_eval_pairs,
            "n_successes": self.n_successes,
            "cost_per_success_usd": (
                None if self.cost_per_success_usd is None
                else round(self.cost_per_success_usd, 6)
            ),
            "forwards_per_success": (
                None if self.forwards_per_success is None
                else round(self.forwards_per_success, 1)
            ),
            "cost_per_success_defined": self.n_successes > 0,
        }
        if self.n_successes <= 0:
            d["undefined_reason"] = (
                "no successful attacks in this run, so cost-per-success is "
                "undefined rather than zero"
            )
        if self.gpu_hourly_usd == 0.0 and self.wall_clock_s > 0:
            d["cost_note"] = (
                "gpu_hourly_usd is 0 (free-tier compute); report forward-pass "
                "counts for a hardware-independent comparison"
            )
        if self.notes:
            d["notes"] = self.notes
        return d


def count_successes(evaluation: Dict) -> int:
    """
    Number of successful attacks in an `evaluate_pairs` result.

    Reads the per-sample `success` flags rather than re-deriving the rate from
    the summary, so rounding in the reported ASR cannot hide a non-zero count.
    """
    samples = evaluation.get("samples") or []
    return sum(1 for s in samples if s.get("success"))
