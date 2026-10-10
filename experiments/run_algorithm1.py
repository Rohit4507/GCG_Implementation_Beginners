"""
Run per-prompt GCG (Algorithm 1) on a single goal/target pair,
then evaluate the optimized suffix (ASR + refusal rate + perplexity).

Examples:
    # Quick debug run on a benign pair (CPU-friendly defaults)
    python -m experiments.run_algorithm1

    # Paper-scale settings on the bundled benchmark sample
    python -m experiments.run_algorithm1 --preset paper --benchmark

    # Larger model with 4-bit quantization (needs CUDA + HF access)
    python -m experiments.run_algorithm1 --preset paper --benchmark \
        --model Qwen/Qwen3-1.7B --load-in-4bit
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
from src.config import GCGConfig
from src.data import load_benchmark_pairs
from src.defense import defense_experiment
from src.evaluate import evaluate_pairs, suffix_perplexity
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.token_manager import TokenManager


def parse_args():
    p = argparse.ArgumentParser(description="GCG Algorithm 1 (individual attack)")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None, help="HF model id (overrides config)")
    p.add_argument("--benchmark", action="store_true",
                   help="Use benchmark goal/target pairs instead of benign ones")
    p.add_argument("--data", default=None, help="jsonl file with goal/target pairs")
    p.add_argument("--pair-index", type=int, default=0, help="Which pair to attack")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
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
    cfg.load_in_4bit = args.load_in_4bit
    cfg.use_kv_cache = not args.no_kv_cache
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
            cfg.model_id, cfg.dtype, args.device, cfg.load_in_4bit,
        )
        device = resolve_device(args.device)

    suffix_ids = init_suffix_ids(
        tokenizer, cfg.suffix_init, cfg.suffix_length, device,
    )
    print(f"Initial suffix ({suffix_ids.shape[0]} tokens): {tokenizer.decode(suffix_ids)!r}")
    print(f"Goal:   {goal}\nTarget: {target!r}\n")

    # ---- Optimize (Algorithm 1) ----
    best_suffix, loss_history = run_gcg(
        model, tokenizer, goal, target, suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache,
    )
    suffix_text = tokenizer.decode(best_suffix, skip_special_tokens=True)
    print(f"\nOptimized suffix: {suffix_text!r}")

    # ---- Evaluate on ALL available pairs (in-domain + generalization) ----
    eval_goals, eval_targets = goals, targets
    eval_result = evaluate_pairs(
        model, tokenizer, eval_goals, eval_targets, best_suffix,
        max_new_tokens=args.max_new_tokens,
        temperature=cfg.temperature,
        device=device,
    )

    # ---- Perplexity diagnostic + defense baseline ----
    tm = TokenManager(tokenizer, goal, target, best_suffix, device)
    ppl_optimized = suffix_perplexity(model, tm, best_suffix)
    defense = defense_experiment(
        model, tokenizer, goal, target, best_suffix, device,
    )

    # ---- Persist artifacts ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Step")
    plt.ylabel("Target cross-entropy")
    plt.title(f"GCG Algorithm 1 — {cfg.model_id} ({args.preset})")
    plt.tight_layout()
    plt.savefig("results/plots/loss_individual.png", dpi=150)
    print("Saved plot to results/plots/loss_individual.png")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "model_id": cfg.model_id,
        "preset": args.preset,
        "seed": cfg.seed,
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "topk": cfg.topk,
        "search_width": cfg.search_width,
        "use_kv_cache": cfg.use_kv_cache,
        "goal": goal,
        "target": target,
        "initial_suffix": cfg.suffix_init,
        "optimized_suffix": suffix_text,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "suffix_perplexity": ppl_optimized,
        "defense": defense,
        "evaluation": eval_result,
    }
    json_path = f"results/reports/generated/algorithm1_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {json_path}")

    # Console summary
    s = eval_result["summary"]
    print("\n" + "=" * 60)
    print(f"ASR (prefix match): {s['asr_prefix_match']:.0%} "
          f"over {s['n_pairs']} pairs | refusal rate: {s['refusal_rate']:.0%}")
    print(f"Suffix perplexity: {ppl_optimized:.1f} "
          f"(defense threshold {defense['threshold']:.1f})")
    for i, sample in enumerate(eval_result["samples"]):
        print(f"\n--- Pair {i}: {sample['goal']} ---")
        print(f"  success={sample['success']} refusal={sample['refusal']}")
        print(f"  output: {sample['response'][:160]!r}")


if __name__ == "__main__":
    main()
