"""
Run universal/multi-prompt GCG (Algorithm 2): ONE suffix optimized
across several goal/target pairs with a progressive prompt schedule.

Also evaluates:
  - in-domain ASR on the optimization prompts,
  - held-out generalization on prompts the suffix never saw.

Examples:
    python -m experiments.run_algorithm2
    python -m experiments.run_algorithm2 --preset paper --benchmark
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

from src.algorithm2 import run_universal_gcg
from src.config import GCGConfig
from src.data import load_benchmark_pairs, load_pairs
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.provenance import RunAccounting, count_successes, describe_model, loss_floor


def parse_args():
    p = argparse.ArgumentParser(description="GCG Algorithm 2 (universal attack)")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Total benchmark pairs to use, before the held-out split")
    p.add_argument("--num-prompts-start", type=int, default=None,
                   help="Initial active prompts in the progressive schedule")
    p.add_argument("--prompt-add-interval", type=int, default=None,
                   help="Add one active prompt every N steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow executing Python shipped in the model repo. Off by "
                        "default because --model is arbitrary text.")
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-rate-usd-per-hour", type=float, default=0.0,
                   help="Cost of the GPU you are using. 0.0 = free tier")
    p.add_argument("--held-out", type=int, default=1,
                   help="Number of pairs held out of optimization")
    p.add_argument("--tiny", action="store_true",
                   help="Use the tiny offline model (no HF download needed)")
    p.add_argument("--quiet", action="store_true", help="Hide the progress bar")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = GCGConfig(preset=args.preset)
    if args.model:
        cfg.model_id = args.model
    if args.steps is not None:
        cfg.num_steps = args.steps
    if args.num_prompts is not None:
        cfg.num_eval_prompts = args.num_prompts
    if args.num_prompts_start is not None:
        cfg.num_prompts_start = args.num_prompts_start
    if args.prompt_add_interval is not None:
        cfg.prompt_add_interval = args.prompt_add_interval
    cfg.load_in_4bit = args.load_in_4bit
    cfg.use_kv_cache = not args.no_kv_cache
    torch.manual_seed(cfg.seed)

    # ---- Data: split into optimization prompts + held-out prompts ----
    dataset_source = "built-in benign demo pairs"
    if args.benchmark:
        goals, targets, dataset_source = load_benchmark_pairs(
            n=cfg.num_eval_prompts, return_source=True,
        )
    elif args.data:
        goals, targets = load_pairs(args.data)
        if args.num_prompts is not None:
            goals, targets = goals[:args.num_prompts], targets[:args.num_prompts]
        dataset_source = args.data
    else:
        goals, targets = cfg.goals, cfg.targets

    if not goals:
        raise SystemExit("no goal/target pairs loaded")

    n_hold = max(args.held_out, 0)
    if n_hold >= len(goals):
        n_hold = max(len(goals) - 1, 0)
    train_goals, train_targets = goals[:-n_hold or None], targets[:-n_hold or None]
    held_goals, held_targets = goals[-n_hold:] if n_hold else [], targets[-n_hold:] if n_hold else []

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
            trust_remote_code=args.trust_remote_code,
        )
        device = resolve_device(args.device)
        model_info = describe_model(
            tiny=False, model_id=cfg.model_id, tokenizer=tokenizer, seed=cfg.seed,
            model=model, requested_4bit=cfg.load_in_4bit,
        )
    model_load_s = time.perf_counter() - t_load
    floor = loss_floor(model_info["vocab_size"])
    print(f"Model actually used : {model_info['model_label']}")

    suffix_ids = init_suffix_ids(
        tokenizer, cfg.suffix_init, cfg.suffix_length, device,
    )
    print(f"Optimization prompts: {len(train_goals)} | held out: {len(held_goals)}")
    print(f"Progressive schedule: start={cfg.num_prompts_start}, "
          f"+1 prompt every {cfg.prompt_add_interval} steps\n")

    # ---- Optimize (Algorithm 2) ----
    stats = {}
    best_suffix, loss_history = run_universal_gcg(
        model=model,
        tokenizer=tokenizer,
        goals=train_goals,
        targets=train_targets,
        suffix_ids=suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size,
        num_prompts_start=cfg.num_prompts_start,
        prompt_add_interval=cfg.prompt_add_interval,
        use_kv_cache=cfg.use_kv_cache,
        verbose=not args.quiet,
        stats_out=stats,
    )
    suffix_text = tokenizer.decode(best_suffix, skip_special_tokens=True)
    print(f"\nUniversal suffix: {suffix_text!r}")

    # ---- Evaluate: in-domain and held-out ----
    in_domain = evaluate_pairs(
        model, tokenizer, train_goals, train_targets, best_suffix,
        max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
        device=device,
    )
    held_out = None
    if held_goals:
        held_out = evaluate_pairs(
            model, tokenizer, held_goals, held_targets, best_suffix,
            max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
            device=device, index_offset=len(train_goals),
        )

    # ---- Persist artifacts ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    steps_u, losses_u = zip(*loss_history) if loss_history else ([], [])
    plt.figure(figsize=(8, 3.5))
    if steps_u:
        plt.plot(steps_u, losses_u, label="mean loss")
    if floor:
        plt.axhline(floor, linestyle="--", linewidth=1,
                    label=f"random-guess floor ln({model_info['vocab_size']})")
    if floor:
        plt.legend(fontsize=7)
    plt.xlabel("Step")
    plt.ylabel("Mean cross-entropy (active prompts)")
    plt.title(f"GCG Algorithm 2 (universal) — {model_info['model_label']} ({args.preset})")
    plt.tight_layout()
    plt.savefig("results/plots/loss_universal.png", dpi=150)
    print("Saved plot to results/plots/loss_universal.png")

    accounting = RunAccounting(
        device=device,
        steps_run=stats.get("steps_run", len(loss_history)),
        gradient_forwards=stats.get("gradient_forwards", 0),
        candidate_forwards=stats.get("candidate_forwards", 0),
        wall_clock_s=stats.get("wall_clock_s", 0.0),
        model_load_s=model_load_s,
        gpu_hourly_usd=args.gpu_rate_usd_per_hour,
        n_eval_pairs=len(train_goals) + len(held_goals),
        n_successes=count_successes(in_domain) + (count_successes(held_out) if held_out else 0),
        notes=(
            "universal attack: one suffix scored against every active prompt, so "
            "forward counts grow with the number of prompts in the schedule"
        ),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "run_kind": "algorithm2_universal",
        "model": model_info,
        "dataset_source": dataset_source,
        "preset": args.preset,
        "seed": cfg.seed,
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "topk": cfg.topk,
        "search_width": cfg.search_width,
        "use_kv_cache": cfg.use_kv_cache,
        "num_prompts_start": cfg.num_prompts_start,
        "prompt_add_interval": cfg.prompt_add_interval,
        "n_train_prompts": len(train_goals),
        "n_held_out_prompts": len(held_goals),
        "optimized_suffix": suffix_text,
        "loss_first": losses_u[0] if losses_u else None,
        "loss_last": losses_u[-1] if losses_u else None,
        "random_guess_floor": floor,
        "in_domain": in_domain,
        "held_out": held_out,
        "accounting": accounting.as_dict(),
    }
    json_path = f"results/reports/generated/algorithm2_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {json_path}")

    # Console summary
    print("\n" + "=" * 60)
    print(f"Model (actually used): {model_info['model_label']}")
    s = in_domain["summary"]
    print(f"In-domain ASR: {s['asr_prefix_match']:.0%} strict | "
          f"{s.get('asr_flexible_prefix', 0.0):.0%} flex | "
          f"{s.get('asr_engaged', 0.0):.0%} engaged over {s['n_pairs']} prompts")
    if held_out:
        hs = held_out["summary"]
        print(f"Held-out ASR:  {hs['asr_prefix_match']:.0%} strict | "
              f"{hs.get('asr_flexible_prefix', 0.0):.0%} flex | "
              f"{hs.get('asr_engaged', 0.0):.0%} engaged over {hs['n_pairs']} prompts")
    if losses_u:
        print(f"Final mean loss: {losses_u[-1]:.4f} (started {losses_u[0]:.4f})")
    if floor:
        print(f"Random-guess floor: {floor:.4f} — a loss sitting here means the "
              f"suffix is not biting")
    print(f"Compute: {accounting.steps_run} steps | "
          f"{accounting.total_forwards:,} forwards | {accounting.wall_clock_s:.1f}s")
    if accounting.cost_per_success_usd is None:
        print("Cost per successful attack: UNDEFINED (0 successes in this run)")
    else:
        print(f"Cost per successful attack: ${accounting.cost_per_success_usd:.4f}")


if __name__ == "__main__":
    main()
