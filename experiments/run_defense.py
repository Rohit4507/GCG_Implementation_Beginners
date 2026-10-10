"""
Defense baseline experiment: can perplexity-based filtering detect
GCG-optimized suffixes?

Optimizes a suffix quickly, then compares the perplexity of:
  - the optimized (adversarial) suffix,
  - a random-token suffix,
  - a natural-language suffix,
and reports whether a simple threshold detector separates them.

    python -m experiments.run_defense
    python -m experiments.run_defense --preset paper --benchmark --model <hf-id>
"""
import argparse
import json
import os
from datetime import datetime, timezone

import torch

from src.algorithm1 import run_gcg
from src.config import GCGConfig
from src.data import load_benchmark_pairs
from src.defense import defense_experiment
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device


def parse_args():
    p = argparse.ArgumentParser(description="Perplexity defense baseline")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny", action="store_true",
                   help="Use the tiny offline model (no HF download needed)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = GCGConfig(preset=args.preset)
    if args.model:
        cfg.model_id = args.model
    if args.steps:
        cfg.num_steps = args.steps
    torch.manual_seed(cfg.seed)

    if args.benchmark:
        goals, targets = load_benchmark_pairs(n=1)
    else:
        goals, targets = cfg.goals[:1], cfg.targets[:1]
    goal, target = goals[0], targets[0]

    if args.tiny:
        from src.tiny_model import build_tiny_model_and_tokenizer
        print("Building tiny offline model (random weights) ...")
        model, tokenizer = build_tiny_model_and_tokenizer(
            goals=goals, targets=targets, seed=cfg.seed,
        )
        device = "cpu"
    else:
        print(f"Loading {cfg.model_id} ...")
        model, tokenizer = load_model_and_tokenizer(
            cfg.model_id, cfg.dtype, args.device,
        )
        device = resolve_device(args.device)
    suffix_ids = init_suffix_ids(
        tokenizer, cfg.suffix_init, cfg.suffix_length, device,
    )

    print(f"Optimizing suffix ({cfg.num_steps} steps) on: {goal!r}")
    best_suffix, loss_history = run_gcg(
        model, tokenizer, goal, target, suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache,
    )

    result = defense_experiment(
        model, tokenizer, goal, target, best_suffix, device,
    )

    os.makedirs("results/reports/generated", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"results/reports/generated/defense_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"goal": goal, "loss_first": loss_history[0],
                   "loss_last": loss_history[-1], **result}, f, indent=2)

    print("\n" + "=" * 60)
    print("Perplexity of suffixes under the attacked model:")
    for name, value in result["perplexities"].items():
        verdict = result["flagged_as_adversarial"][name]
        print(f"  {name:>10}: PPL={value:8.2f}  flagged_as_adversarial={verdict}")
    print(f"\nDetector threshold: {result['threshold']:.2f}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
