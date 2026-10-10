"""
GCG vs. random search: does the gradient signal actually help?

Runs Algorithm 1 and a gradient-blind random-search baseline with identical
budgets (same steps, search width, seed, initialization) and compares:
  - target-loss curves,
  - final suffix ASR / refusal rate via the evaluation harness.

    python -m experiments.run_baselines --tiny
    python -m experiments.run_baselines --preset paper --benchmark --model <hf-id>
"""
import argparse
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.algorithm1 import run_gcg
from src.baselines import run_random_search, target_loss
from src.config import GCGConfig
from src.data import load_benchmark_pairs
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.token_manager import TokenManager


def parse_args():
    p = argparse.ArgumentParser(description="GCG vs random-search baseline")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", default=None)
    p.add_argument("--pair-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny", action="store_true",
                   help="Use the tiny offline model (no HF download needed)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = GCGConfig(preset=args.preset)
    if args.model:
        cfg.model_id = args.model
    if args.steps:
        cfg.num_steps = args.steps
    torch.manual_seed(cfg.seed)

    # ---- Data ----
    if args.benchmark:
        goals, targets = load_benchmark_pairs(n=cfg.num_eval_prompts)
    elif args.data:
        from src.data import load_pairs
        goals, targets = load_pairs(args.data)
    else:
        goals, targets = cfg.goals, cfg.targets
    goal, target = goals[args.pair_index], targets[args.pair_index]

    # ---- Model ----
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
    print(f"Goal:   {goal}\nTarget: {target!r}")
    print(f"Budget: {cfg.num_steps} steps x {cfg.search_width} candidates\n")

    # ---- Run GCG (seeded) ----
    torch.manual_seed(cfg.seed)
    gcg_suffix, gcg_losses = run_gcg(
        model, tokenizer, goal, target, suffix_ids.clone(),
        num_steps=cfg.num_steps, topk=cfg.topk,
        search_width=cfg.search_width, mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache, verbose=True,
    )

    # ---- Run random search (identical budget, reseeded) ----
    torch.manual_seed(cfg.seed)
    rnd_suffix, rnd_losses = run_random_search(
        model, tokenizer, goal, target, suffix_ids.clone(),
        num_steps=cfg.num_steps, search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size, use_kv_cache=cfg.use_kv_cache,
        verbose=True,
    )

    # ---- Evaluate both suffixes ----
    evaluation = {}
    for name, suffix in (("gcg", gcg_suffix), ("random_search", rnd_suffix),
                         ("initial", suffix_ids)):
        ev = evaluate_pairs(
            model, tokenizer, goals, targets, suffix,
            max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
            device=device,
        )
        tm = TokenManager(tokenizer, goal, target, suffix, device)
        ev["mean_target_loss"] = target_loss(model, tm, suffix)
        evaluation[name] = ev

    # ---- Plot loss curves ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(gcg_losses) + 1), gcg_losses, label="GCG (gradient-guided)")
    plt.plot(range(1, len(rnd_losses) + 1), rnd_losses, label="Random search (blind)")
    plt.xlabel("Step")
    plt.ylabel("Best suffix target cross-entropy")
    plt.title(f"GCG vs random search — {cfg.model_id} ({args.preset})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/baselines_loss.png", dpi=150)
    print("Saved plot to results/plots/baselines_loss.png")

    # ---- Persist + summarize ----
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "model_id": cfg.model_id,
        "preset": args.preset,
        "seed": cfg.seed,
        "num_steps": cfg.num_steps,
        "search_width": cfg.search_width,
        "goal": goal,
        "target": target,
        "gcg_final_loss": gcg_losses[-1],
        "random_search_final_loss": rnd_losses[-1],
        "gcg_suffix": tokenizer.decode(gcg_suffix, skip_special_tokens=True),
        "random_search_suffix": tokenizer.decode(rnd_suffix, skip_special_tokens=True),
        "evaluation": evaluation,
    }
    path = f"results/reports/generated/baselines_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {path}")

    print("\n" + "=" * 64)
    print(f"{'':>16} | {'final loss':>10} | {'ASR':>5} | {'refusal':>7}")
    for name in ("gcg", "random_search", "initial"):
        e = evaluation[name]
        print(f"{name:>16} | {e['mean_target_loss']:>10.4f} | "
              f"{e['summary']['asr_prefix_match']:>4.0%} | "
              f"{e['summary']['refusal_rate']:>6.0%}")
    print("=" * 64)
    if gcg_losses[-1] < rnd_losses[-1]:
        print("✅ Gradient guidance wins on final loss "
              f"({gcg_losses[-1]:.4f} < {rnd_losses[-1]:.4f})")
    else:
        print("⚠️ Random search matched/beat GCG on final loss — check budget "
              "(`--steps`) or increase `search_width`/`topk`.")


if __name__ == "__main__":
    main()
