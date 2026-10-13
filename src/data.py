"""
Dataset loading for GCG experiments.

Primary source is a local jsonl file of {"goal": ..., "target": ...} records
(bundled samples live in data/). If the `datasets` library and network access
are available, the full AdvBench benchmark (walledai/AdvBench, 520 behaviours)
is pulled from the Hugging Face hub.

The bundled `advbench_sample.jsonl` is a 30-pair fallback for offline
development and CI. It is large enough to report a rate over, but it is a
sample: publish the full AdvBench numbers and say which was used.
"""
import json
import os
import random
from typing import List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BENIGN_PATH = os.path.join(DATA_DIR, "benign_pairs.jsonl")
ADVBENCH_SAMPLE_PATH = os.path.join(DATA_DIR, "advbench_sample.jsonl")


def load_pairs(path: str) -> Tuple[List[str], List[str]]:
    """Load (goals, targets) from a jsonl file with 'goal'/'target' keys."""
    goals, targets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            goals.append(record["goal"])
            targets.append(record["target"])
    return goals, targets


def load_benign_pairs() -> Tuple[List[str], List[str]]:
    return load_pairs(BENIGN_PATH)


def load_benchmark_pairs(
    n: Optional[int] = None,
    seed: int = 42,
    prefer_hub: bool = True,
    return_source: bool = False,
):
    """
    Load n harmful-request / affirmative-target evaluation pairs.

    Tries the Hugging Face hub (walledai/AdvBench, 520 behaviours) first, then
    falls back to the bundled AdvBench-style sample so the pipeline works
    offline. Which one was used is a published-results detail, not a debug
    message, so `return_source=True` hands it back for the run record.

    Returns (goals, targets), or (goals, targets, source) when
    `return_source=True`.
    """
    pairs = None
    source = f"bundled sample ({os.path.relpath(ADVBENCH_SAMPLE_PATH)})"

    if prefer_hub:
        try:
            from datasets import load_dataset
            ds = load_dataset("walledai/AdvBench", split="train")
            rows = [{"goal": r["prompt"], "target": r["target"]} for r in ds]
            rng = random.Random(seed)
            rng.shuffle(rows)
            pairs = ([r["goal"] for r in rows], [r["target"] for r in rows])
            source = "walledai/AdvBench (Hugging Face hub, 520 behaviours)"
        except Exception as exc:  # offline, missing datasets, hub down, ...
            print(f"[data] Hub AdvBench unavailable ({type(exc).__name__}); "
                  f"falling back to bundled sample.")

    if pairs is None:
        pairs = load_pairs(ADVBENCH_SAMPLE_PATH)

    goals, targets = pairs
    if n is not None:
        goals, targets = goals[:n], targets[:n]
    print(f"[data] Loaded {len(goals)} benchmark pairs from {source}")
    if return_source:
        return goals, targets, source
    return goals, targets
