"""
Run per-prompt GCG (Algorithm 1) on a single goal/target pair,
then evaluate the optimized suffix (ASR + refusal rate + perplexity).

Records what the run actually did: the model that was really loaded, how many
forward passes it took, how long it took, and what that cost. See
src/provenance.py for why the model label is not simply `cfg.model_id`.

Examples:
    # Quick debug run on a benign pair (CPU-friendly defaults)
    python -m experiments.run_algorithm1

    # Paper-scale settings on the bundled benchmark sample
    python -m experiments.run_algorithm1 --preset paper --benchmark --num-prompts 25

    # Real model, 4-bit, on a free-tier GPU (Kaggle T4 / Colab)
    python -m experiments.run_algorithm1 --preset paper --benchmark \
        --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit --steps 500 \
        --num-prompts 25
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.algorithm1 import run_gcg
from src.config import GCGConfig
from src.data import load_benchmark_pairs, load_pairs
from src.defense import defense_experiment
from src.evaluate import evaluate_pairs, suffix_perplexity
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.provenance import RunAccounting, count_successes, describe_model, loss_floor
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
    p.add_argument("--num-prompts", type=int, default=None,
                   help="How many benchmark pairs to evaluate on (default: config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-rate-usd-per-hour", type=float, default=0.0,
                   help="Cost of the GPU you are using. 0.0 = free tier (Kaggle/Colab)")
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
    if args.num_prompts:
        cfg.num_eval_prompts = args.num_prompts
    cfg.load_in_4bit = args.load_in_4bit
    cfg.use_kv_cache = not args.no_kv_cache
    torch.manual_seed(cfg.seed)

    # ---- Data ----
    dataset_source = "built-in benign demo pairs"
    if args.benchmark:
        goals, targets, dataset_source = load_benchmark_pairs(
            n=cfg.num_eval_prompts, return_source=True,
        )
    elif args.data:
        goals, targets = load_pairs(args.data)
        dataset_source = args.data
    else:
        goals, targets = cfg.goals, cfg.targets

    goal, target = goals[args.pair_index], targets[args.pair_index]

    # ---- Model ----
    t_load = time.perf_counter()
    if args.tiny:
        from src.tiny_model import build_tiny_model_and_tokenizer
        print("Building tiny offline model (random weights) ...")
        model, tokenizer = build_tiny_model_and_tokenizer(
            goals=goals, targets=targets, seed=cfg.seed,
        )
        device = "cpu"
        model_info = describe_model(
            tiny=True, model_id=cfg.model_id, tokenizer=tokenizer, seed=cfg.seed,
        )
    else:
        print(f"Loading {cfg.model_id} ...")
        model, tokenizer = load_model_and_tokenizer(
            cfg.model_id, cfg.dtype, args.device, cfg.load_in_4bit,
        )
        device = resolve_device(args.device)
        model_info = describe_model(
            tiny=False, model_id=cfg.model_id, tokenizer=tokenizer, seed=cfg.seed,
        )
    model_load_s = time.perf_counter() - t_load

    floor = loss_floor(model_info["vocab_size"])
    print(f"Model actually used : {model_info['model_label']}")
    print(f"Random-weights floor: ln({model_info['vocab_size']}) = {floor:.3f}"
          if floor else "Random-weights floor: n/a")

    suffix_ids = init_suffix_ids(
        tokenizer, cfg.suffix_init, cfg.suffix_length, device,
    )
    print(f"Initial suffix ({suffix_ids.shape[0]} tokens): {tokenizer.decode(suffix_ids)!r}")
    print(f"Goal:   {goal}\nTarget: {target!r}\n")

    # ---- Optimize (Algorithm 1) ----
    stats = {}
    best_suffix, loss_history = run_gcg(
        model, tokenizer, goal, target, suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache,
        stats_out=stats,
    )
    suffix_text = tokenizer.decode(best_suffix, skip_special_tokens=True)
    print(f"\nOptimized suffix: {suffix_text!r}")

    # ---- Evaluate on ALL available pairs (in-domain + generalization) ----
    eval_result = evaluate_pairs(
        model, tokenizer, goals, targets, best_suffix,
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

    # ---- Cost accounting ----
    accounting = RunAccounting(
        device=device,
        steps_run=stats.get("steps_run", len(loss_history)),
        gradient_forwards=stats.get("gradient_forwards", 0),
        candidate_forwards=stats.get("candidate_forwards", 0),
        wall_clock_s=stats.get("wall_clock_s", 0.0),
        model_load_s=model_load_s,
        gpu_hourly_usd=args.gpu_rate_usd_per_hour,
        n_eval_pairs=len(goals),
        n_successes=count_successes(eval_result),
    )

    # ---- Persist artifacts ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    if floor:
        plt.axhline(floor, linestyle="--", linewidth=1,
                    label=f"random-guess floor ln({model_info['vocab_size']})")
        plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Target cross-entropy")
    plt.title(f"GCG Algorithm 1 — {model_info['model_label']} ({args.preset})")
    plt.tight_layout()
    plt.savefig("results/plots/loss_individual.png", dpi=150)
    print("Saved plot to results/plots/loss_individual.png")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        # --- who/what this run actually was ---
        "run_kind": "algorithm1_per_prompt",
        "model": model_info,
        "dataset_source": dataset_source,
        "n_prompts_evaluated": len(goals),
        "preset": args.preset,
        "seed": cfg.seed,
        # --- optimiser settings ---
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "topk": cfg.topk,
        "search_width": cfg.search_width,
        "use_kv_cache": cfg.use_kv_cache,
        # --- attack ---
        "goal": goal,
        "target": target,
        "initial_suffix": cfg.suffix_init,
        "optimized_suffix": suffix_text,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "random_guess_floor": floor,
        "suffix_perplexity": ppl_optimized,
        "defense": defense,
        "evaluation": eval_result,
        # --- cost ---
        "accounting": accounting.as_dict(),
    }
    json_path = f"results/reports/generated/algorithm1_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {json_path}")

    # ---- Console summary ----
    s = eval_result["summary"]
    print("\n" + "=" * 66)
    print(f"Model (actually used): {model_info['model_label']}")
    print(f"ASR (prefix match): {s['asr_prefix_match']:.0%} over {s['n_pairs']} pairs"
          f" | refusal rate: {s['refusal_rate']:.0%}")
    if s["asr_prefix_match"] == 0:
        print("  -> 0% ASR. Check loss vs the random-guess floor above: a loss "
              "sitting at the floor means the attack is not biting.")
    print(f"Compute: {accounting.steps_run} steps | "
          f"{accounting.total_forwards:,} forwards | "
          f"{accounting.wall_clock_s:.1f}s "
          f"({accounting.seconds_per_step:.2f}s/step)")
    if accounting.cost_per_success_usd is None:
        print("Cost per successful attack: UNDEFINED (0 successes in this run)")
    else:
        print(f"Cost per successful attack: ${accounting.cost_per_success_usd:.4f} "
              f"({accounting.forwards_per_success:,.0f} forwards/success)")
    print(f"Suffix perplexity: {ppl_optimized:.1f} "
          f"(defense threshold {defense['threshold']:.1f})")
    for i, sample in enumerate(eval_result["samples"]):
        print(f"\n--- Pair {i}: {sample['goal'][:80]} ---")
        print(f"  success={sample['success']} refusal={sample['refusal']}")
        print(f"  output: {sample['response'][:160]!r}")


if __name__ == "__main__":
    main()
