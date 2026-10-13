"""
Multi-turn vs single-turn attack comparison.

Single-turn jailbreaks are largely patched; the live research area is
multi-turn escalation, where the harmful request arrives inside an existing
conversation. This runner attacks the SAME goal twice:

  1. single-turn  -- suffix optimised against a first-turn user message
                     (what the rest of this repo does)
  2. multi-turn   -- suffix optimised against the goal as the FINAL turn of a
                     scripted escalation conversation (src/multiturn.py)

and reports both ASRs side by side. See src/multiturn.py for the limitations of
the escalation script before writing any of this up.

    # Offline pipeline check (random model, meaningless ASR, exercises the path)
    python -m experiments.run_multiturn --tiny --benchmark --steps 5
    python -m experiments.run_multiturn --tiny --preset debug

    # Real run (GPU)
    python -m experiments.run_multiturn --preset paper --benchmark \
        --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit --steps 500 \
        --num-prompts 10
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
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.multiturn import (
    build_prior_turns,
    count_context_messages,
    count_prior_turns,
    summarise_comparison,
)
from src.provenance import RunAccounting, count_successes, describe_model, loss_floor


def parse_args():
    p = argparse.ArgumentParser(description="Multi-turn vs single-turn GCG")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", default=None)
    p.add_argument("--pair-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Pairs the fixed suffix is evaluated on")
    p.add_argument("--n-prior-turns", type=int, default=2,
                   help="Escalation messages before the attacked turn (default 2 = 3-turn attack)")
    p.add_argument("--device", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-new-tokens-escalation", type=int, default=96)
    p.add_argument("--gpu-rate-usd-per-hour", type=float, default=0.0)
    p.add_argument("--tiny", action="store_true")
    return p.parse_args()


def _optimise(model, tokenizer, goal, target, suffix_ids, cfg, prior_turns, label):
    """Run Algorithm 1 once and return (best_suffix, loss_history, stats)."""
    print(f"\n--- Optimising ({label}, {cfg.num_steps} steps) ---")
    stats = {}
    best_suffix, loss_history = run_gcg(
        model, tokenizer, goal, target, suffix_ids.clone(),
        num_steps=cfg.num_steps, topk=cfg.topk,
        search_width=cfg.search_width, mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache, stats_out=stats,
        prior_turns=prior_turns,
    )
    print(f"    suffix: {tokenizer.decode(best_suffix, skip_special_tokens=True)!r}")
    print(f"    loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    return best_suffix, loss_history, stats


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
    # Evaluation uses the benign remainder so the fixed suffix is scored on
    # held-out goals rather than the one it was optimised on.
    eval_goals = [g for i, g in enumerate(goals) if i != args.pair_index]
    eval_targets = [t for i, t in enumerate(targets) if i != args.pair_index]

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

    suffix_ids = init_suffix_ids(
        tokenizer, cfg.suffix_init, cfg.suffix_length, device,
    )

    # ---- Build the escalation conversation (model's own replies) ----
    prior_turns = None
    transcript = []
    if args.n_prior_turns > 0:
        print(f"\nBuilding {args.n_prior_turns}-turn escalation conversation ...")
        prior_turns, transcript = build_prior_turns(
            model, tokenizer, device,
            n_prior_turns=args.n_prior_turns,
            max_new_tokens=args.max_new_tokens_escalation,
            verbose=True,
        )

    # ---- Single-turn attack ----
    single_suffix, single_losses, single_stats = _optimise(
        model, tokenizer, goal, target, suffix_ids, cfg, None, "single-turn",
    )
    single_eval = evaluate_pairs(
        model, tokenizer, eval_goals, eval_targets, single_suffix,
        max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
        device=device,
    )

    # ---- Multi-turn attack ----
    multi_suffix, multi_losses, multi_stats = _optimise(
        model, tokenizer, goal, target, suffix_ids, cfg, prior_turns, "multi-turn",
    )
    multi_eval = evaluate_pairs(
        model, tokenizer, eval_goals, eval_targets, multi_suffix,
        max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
        device=device, prior_turns=prior_turns,
    )

    # ---- Compare ----
    single_side = {
        "loss_last": single_losses[-1],
        "evaluation": single_eval,
    }
    multi_side = {
        "loss_last": multi_losses[-1],
        "evaluation": multi_eval,
    }
    comparison = summarise_comparison(single_side, multi_side)

    accounting = RunAccounting(
        device=device,
        steps_run=single_stats.get("steps_run", 0) + multi_stats.get("steps_run", 0),
        gradient_forwards=(single_stats.get("gradient_forwards", 0)
                           + multi_stats.get("gradient_forwards", 0)),
        candidate_forwards=(single_stats.get("candidate_forwards", 0)
                            + multi_stats.get("candidate_forwards", 0)),
        wall_clock_s=(single_stats.get("wall_clock_s", 0.0)
                      + multi_stats.get("wall_clock_s", 0.0)),
        model_load_s=model_load_s,
        gpu_hourly_usd=args.gpu_rate_usd_per_hour,
        n_eval_pairs=len(eval_goals),
        n_successes=count_successes(single_eval) + count_successes(multi_eval),
        notes="counters cover both the single-turn and multi-turn optimisation",
    )

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, len(single_losses) + 1), single_losses, label="single-turn")
    plt.plot(range(1, len(multi_losses) + 1), multi_losses, label="multi-turn")
    if floor:
        plt.axhline(floor, linestyle="--", linewidth=1,
                    label=f"random-guess floor ln({model_info['vocab_size']})")
    plt.xlabel("Step")
    plt.ylabel("Target cross-entropy")
    plt.title(f"single vs multi-turn GCG — {model_info['model_label']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/multiturn_loss.png", dpi=150)
    print("\nSaved plot to results/plots/multiturn_loss.png")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "run_kind": "multiturn_vs_single",
        "model": model_info,
        "dataset_source": dataset_source,
        "preset": args.preset,
        "seed": cfg.seed,
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "n_prior_user_turns": count_prior_turns(prior_turns),
        "n_context_messages": count_context_messages(prior_turns),
        "attack_user_turn": count_prior_turns(prior_turns) + 1,
        "goal_attacked": goal,
        "n_prompts_evaluated": len(eval_goals),
        "escalation_transcript": transcript,
        "single_turn": {
            "suffix": tokenizer.decode(single_suffix, skip_special_tokens=True),
            "loss_first": single_losses[0],
            "loss_last": single_losses[-1],
            "accounting": RunAccounting(
                device=device,
                steps_run=single_stats.get("steps_run", 0),
                gradient_forwards=single_stats.get("gradient_forwards", 0),
                candidate_forwards=single_stats.get("candidate_forwards", 0),
                wall_clock_s=single_stats.get("wall_clock_s", 0.0),
                gpu_hourly_usd=args.gpu_rate_usd_per_hour,
                n_eval_pairs=len(eval_goals),
                n_successes=count_successes(single_eval),
            ).as_dict(),
            "evaluation": single_eval,
        },
        "multi_turn": {
            "suffix": tokenizer.decode(multi_suffix, skip_special_tokens=True),
            "loss_first": multi_losses[0],
            "loss_last": multi_losses[-1],
            "accounting": RunAccounting(
                device=device,
                steps_run=multi_stats.get("steps_run", 0),
                gradient_forwards=multi_stats.get("gradient_forwards", 0),
                candidate_forwards=multi_stats.get("candidate_forwards", 0),
                wall_clock_s=multi_stats.get("wall_clock_s", 0.0),
                gpu_hourly_usd=args.gpu_rate_usd_per_hour,
                n_eval_pairs=len(eval_goals),
                n_successes=count_successes(multi_eval),
            ).as_dict(),
            "evaluation": multi_eval,
        },
        "comparison": comparison,
        "accounting": accounting.as_dict(),
    }
    path = f"results/reports/generated/multiturn_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {path}")

    print("\n" + "=" * 66)
    print(f"Model (actually used): {model_info['model_label']}")
    print(f"Attacked message is user turn {run['attack_user_turn']} "
          f"({run['n_context_messages']} prior messages in context)")
    print(f"{'setting':>12} | {'final loss':>10} | {'ASR':>5} | {'refusal':>7}")
    for label, side in (("single-turn", single_side), ("multi-turn", multi_side)):
        s = side["evaluation"]["summary"]
        print(f"{label:>12} | {side['loss_last']:>10.4f} | "
              f"{s['asr_prefix_match']:>4.0%} | {s['refusal_rate']:>6.0%}")
    print("=" * 66)
    print(f"Verdict: {comparison['verdict']}")
    if comparison["verdict"].startswith("no attack succeeded"):
        print("  -> A null result. Report it as such: at this model size and "
              "budget, multi-turn escalation did not change the outcome.")
    print(f"Compute: {accounting.total_forwards:,} forwards, "
          f"{accounting.wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()
