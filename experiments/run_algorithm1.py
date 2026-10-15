"""
Run GCG (Algorithm 1) and report ASR over the prompts that were attacked.

Two modes, and the difference is the whole point:

  single suffix (default)
    One suffix is optimised against ONE prompt (`--pair-index`) and then
    evaluated on every loaded prompt. Only the attacked prompt is in-domain.
    The others measure *transfer* — the suffix never saw them. The record
    splits the two, because a rate that pools them is neither.

  --per-prompt
    A separate suffix is optimised for every loaded prompt, and each is scored
    on its own prompt. This is the only mode in which "ASR over n prompts" is
    really an ASR over n prompts, because every one of them was attacked.

The distinction is not cosmetic. The first real-model run optimised one
suffix on pair 0 of 10, evaluated all 10, and wrote a single ASR over 10
pairs. Reading that record back there is no way to recover the in-domain rate
or the transfer rate, and the n attached to the headline is wrong in both
directions — too large for the attack, too small for the transfer claim.

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

    # Own suffix per prompt: the honest per-prompt ASR
    python -m experiments.run_algorithm1 --per-prompt --preset paper --benchmark \
        --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit \
        --steps 250 --num-prompts 3
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
from src.evaluate import evaluate_pairs, pool_evaluations, suffix_perplexity
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.provenance import RunAccounting, count_successes, describe_model, loss_floor
from src.token_manager import TokenManager

# run_kind values. `algorithm1_per_prompt` used to label single-suffix runs;
# src/judge.py's own words aside, anything without `n_suffixes_optimised` was
# written by the old code and is a single-suffix record (see build_report).
RUN_KIND_SINGLE = "algorithm1_single_suffix"
RUN_KIND_PER_PROMPT = "algorithm1_per_prompt"

SCOPE_SINGLE = "one suffix optimised on one prompt, then applied to the rest"
SCOPE_PER_PROMPT = "a separate suffix optimised for each prompt"


def parse_args():
    p = argparse.ArgumentParser(description="GCG Algorithm 1 (individual attack)")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None, help="HF model id (overrides config)")
    p.add_argument("--benchmark", action="store_true",
                   help="Use benchmark goal/target pairs instead of benign ones")
    p.add_argument("--data", default=None, help="jsonl file with goal/target pairs")
    p.add_argument("--pair-index", type=int, default=0,
                   help="Which pair to attack (single-suffix mode only)")
    p.add_argument("--per-prompt", action="store_true",
                   help="Optimise a separate suffix for EVERY loaded prompt, so "
                        "the reported ASR is over prompts that were all attacked")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="How many benchmark pairs to load (default: config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gpu-rate-usd-per-hour", type=float, default=0.0,
                   help="Cost of the GPU you are using. 0.0 = free tier (Kaggle/Colab)")
    p.add_argument("--tiny", action="store_true",
                   help="Use the tiny offline model (no HF download needed)")
    p.add_argument("--quiet", action="store_true", help="Hide the progress bars")
    return p.parse_args()


def evaluate_subset(model, tokenizer, goals, targets, suffix, cfg, args, device,
                    index_offset=0):
    """
    `evaluate_pairs` on a subset, or None when the subset is empty.

    Returning None (rather than an empty-but-valid result) matters: it lets a
    report say "there were no held-out prompts" instead of printing a 0% that
    came from dividing by zero pairs.
    """
    if not goals:
        return None
    return evaluate_pairs(
        model, tokenizer, goals, targets, suffix,
        max_new_tokens=args.max_new_tokens,
        temperature=cfg.temperature,
        device=device,
        index_offset=index_offset,
    )


def optimise(model, tokenizer, goal, target, suffix_ids, cfg, args):
    """One Algorithm 1 run. Returns (best_suffix, loss_history, stats)."""
    stats = {}
    best_suffix, loss_history = run_gcg(
        model, tokenizer, goal, target, suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache,
        verbose=not args.quiet,
        stats_out=stats,
    )
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

    if not goals:
        raise SystemExit("no goal/target pairs loaded")

    if not args.per_prompt and not 0 <= args.pair_index < len(goals):
        raise SystemExit(
            f"--pair-index {args.pair_index} is out of range for {len(goals)} pair(s)"
        )

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

    # ---- Which prompts are actually attacked ----
    attacked = list(range(len(goals))) if args.per_prompt else [args.pair_index]
    print(f"\nMode: {'per-prompt (own suffix per prompt)' if args.per_prompt else 'single suffix'}")
    print(f"Prompts attacked: {len(attacked)} of {len(goals)} loaded")
    if not args.per_prompt and len(goals) > 1:
        print(f"  -> the other {len(goals) - 1} prompt(s) measure TRANSFER, not ASR; "
              f"the record keeps them separate")
    if args.per_prompt and len(goals) > 1:
        print(f"  -> cost scales with prompt count: ~{len(attacked)}x a single run")

    # ---- Optimise ----
    gradient_forwards = candidate_forwards = 0
    t_opt = time.perf_counter()
    per_prompt = []
    loss_curves = {}

    if args.per_prompt:
        in_domain_evals, transfer_evals = [], []

        for position, i in enumerate(attacked):
            print(f"\n{'=' * 70}")
            print(f"Prompt {position + 1}/{len(attacked)}  [pair {i}]: {goals[i][:70]}")
            print(f"{'=' * 70}")

            # Reseed per prompt so each suffix is optimised independently and
            # the whole run stays reproducible from `cfg.seed` alone.
            torch.manual_seed(cfg.seed + i)
            suffix_ids = init_suffix_ids(
                tokenizer, cfg.suffix_init, cfg.suffix_length, device,
            )

            best_suffix, loss_history, stats = optimise(
                model, tokenizer, goals[i], targets[i], suffix_ids, cfg, args,
            )
            gradient_forwards += stats.get("gradient_forwards", 0)
            candidate_forwards += stats.get("candidate_forwards", 0)

            suffix_text = tokenizer.decode(best_suffix, skip_special_tokens=True)
            representative_suffix, representative_suffix_text = best_suffix, suffix_text
            loss_curves[f"pair {i}"] = loss_history
            print(f"  optimized suffix: {suffix_text!r}")
            print(f"  loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"
                  if loss_history else "  loss: n/a")

            ev_in = evaluate_subset(
                model, tokenizer, [goals[i]], [targets[i]], best_suffix,
                cfg, args, device, index_offset=i,
            )
            others = [j for j in range(len(goals)) if j != i]
            ev_tr = evaluate_subset(
                model, tokenizer, [goals[j] for j in others],
                [targets[j] for j in others], best_suffix,
                cfg, args, device, index_offset=None if not others else others[0],
            )

            per_prompt.append({
                "pair_index": i,
                "goal": goals[i],
                "target": targets[i],
                "optimized_suffix": suffix_text,
                "loss_first": loss_history[0] if loss_history else None,
                "loss_last": loss_history[-1] if loss_history else None,
                "in_domain": ev_in,
                "transfer": ev_tr,
            })
            if ev_in:
                in_domain_evals.append(ev_in)
            if ev_tr:
                transfer_evals.append(ev_tr)

            done = position + 1
            elapsed = time.perf_counter() - t_opt
            if done and done < len(attacked):
                print(f"  [{done}/{len(attacked)}] {elapsed / 60:.1f} min elapsed, "
                      f"~{elapsed / done * (len(attacked) - done) / 60:.1f} min remaining")

        # The ASR is the pooled per-prompt in-domain rate: n = number of
        # prompts attacked, each scored on the prompt it was optimised for.
        eval_result = pool_evaluations(in_domain_evals)
        transfer_result = pool_evaluations(transfer_evals)
        suffix_text = representative_suffix_text
        loss_history = loss_curves.get(f"pair {attacked[-1]}", [])
        summary_goal, summary_target = goals[attacked[-1]], targets[attacked[-1]]
    else:
        i = attacked[0]
        goal, target = goals[i], targets[i]
        suffix_ids = init_suffix_ids(
            tokenizer, cfg.suffix_init, cfg.suffix_length, device,
        )
        print(f"Initial suffix ({suffix_ids.shape[0]} tokens): {tokenizer.decode(suffix_ids)!r}")
        print(f"Goal:   {goal}\nTarget: {target!r}\n")

        best_suffix, loss_history, stats = optimise(
            model, tokenizer, goal, target, suffix_ids, cfg, args,
        )
        gradient_forwards += stats.get("gradient_forwards", 0)
        candidate_forwards += stats.get("candidate_forwards", 0)
        suffix_text = tokenizer.decode(best_suffix, skip_special_tokens=True)
        representative_suffix = best_suffix
        loss_curves[f"pair {i}"] = loss_history
        print(f"\nOptimized suffix: {suffix_text!r}")

        # In-domain is the attacked prompt alone; everything else is transfer.
        ev_in = evaluate_subset(
            model, tokenizer, [goal], [target], best_suffix,
            cfg, args, device, index_offset=i,
        )
        others = [j for j in range(len(goals)) if j != i]
        ev_tr = evaluate_subset(
            model, tokenizer, [goals[j] for j in others],
            [targets[j] for j in others], best_suffix,
            cfg, args, device, index_offset=None if not others else others[0],
        )
        eval_result, transfer_result = ev_in, ev_tr
        summary_goal, summary_target = goal, target

    wall_clock_s = time.perf_counter() - t_opt
    n_suffixes = len(per_prompt) if args.per_prompt else 1

    # ---- Perplexity diagnostic + defense baseline ----
    # Runs on ONE representative suffix (the last one optimised). In per-prompt
    # mode every prompt has its own suffix, and a single perplexity number would
    # silently stand in for all of them, so the record says so.
    tm = TokenManager(tokenizer, summary_goal, summary_target,
                      representative_suffix, device)
    ppl_optimized = suffix_perplexity(model, tm, representative_suffix)
    defense = defense_experiment(
        model, tokenizer, summary_goal, summary_target, representative_suffix, device,
    )
    if args.per_prompt:
        defense["representative_only"] = (
            "one suffix of the optimised set; per-prompt mode produces a different "
            "suffix for every prompt, so this diagnostic does not describe all of them"
        )

    # ---- Cost accounting ----
    accounting = RunAccounting(
        device=device,
        steps_run=stats.get("steps_run", 0) * n_suffixes if args.per_prompt
        else stats.get("steps_run", len(loss_history)),
        gradient_forwards=gradient_forwards,
        candidate_forwards=candidate_forwards,
        wall_clock_s=wall_clock_s,
        model_load_s=model_load_s,
        gpu_hourly_usd=args.gpu_rate_usd_per_hour,
        n_eval_pairs=len(goals),
        n_successes=count_successes(eval_result or {}),
        notes=(
            f"{n_suffixes} suffix(es) optimised; ASR is pooled over the prompts "
            f"each suffix was optimised on"
            if args.per_prompt else
            f"one suffix optimised on pair {attacked[0]}; in-domain n=1, "
            f"the other {max(len(goals) - 1, 0)} evaluation(s) measure transfer"
        ),
    )

    # ---- Persist artifacts ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 3.5))
    for label, curve in loss_curves.items():
        plt.plot(range(1, len(curve) + 1), curve, label=label)
    if floor:
        plt.axhline(floor, linestyle="--", linewidth=1,
                    label=f"random-guess floor ln({model_info['vocab_size']})")
    if len(loss_curves) > 1 or floor:
        plt.legend(fontsize=7)
    plt.xlabel("Step")
    plt.ylabel("Target cross-entropy")
    plt.title(f"GCG Algorithm 1 — {model_info['model_label']} ({args.preset})")
    plt.tight_layout()
    plt.savefig("results/plots/loss_individual.png", dpi=150)
    print("Saved plot to results/plots/loss_individual.png")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        # --- who/what this run actually was ---
        "run_kind": RUN_KIND_PER_PROMPT if args.per_prompt else RUN_KIND_SINGLE,
        "attack_scope": SCOPE_PER_PROMPT if args.per_prompt else SCOPE_SINGLE,
        "model": model_info,
        "dataset_source": dataset_source,
        "n_prompts_evaluated": len(goals),
        "attacked_pair_indices": attacked,
        "n_prompts_attacked": len(attacked),
        "n_suffixes_optimised": n_suffixes,
        "preset": args.preset,
        "seed": cfg.seed,
        # --- optimiser settings ---
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "topk": cfg.topk,
        "search_width": cfg.search_width,
        "use_kv_cache": cfg.use_kv_cache,
        # --- attack ---
        "goal": summary_goal,
        "target": summary_target,
        "initial_suffix": cfg.suffix_init,
        "optimized_suffix": suffix_text,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "random_guess_floor": floor,
        "suffix_perplexity": ppl_optimized,
        "defense": defense,
        # --- outcome, split by what was actually attacked ---
        "in_domain": eval_result,
        "transfer": transfer_result,
        # --- cost ---
        "accounting": accounting.as_dict(),
    }
    if args.per_prompt:
        run["per_prompt"] = per_prompt

    json_path = f"results/reports/generated/algorithm1_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {json_path}")

    # ---- Console summary ----
    print("\n" + "=" * 66)
    print(f"Model (actually used): {model_info['model_label']}")
    print(f"Attack scope: {run['attack_scope']}")
    print(f"Suffixes optimised: {n_suffixes} | prompts attacked: {len(attacked)} "
          f"of {len(goals)} loaded")

    if eval_result:
        s = eval_result["summary"]
        print(f"\nIN-DOMAIN ASR (this is the headline): "
              f"{s['asr_prefix_match']:.0%} over n={s['n_pairs']} attacked prompt(s)")
        print(f"  refusal rate: {s['refusal_rate']:.0%}")
        if s["asr_prefix_match"] == 0:
            print("  -> 0% on the paper's criterion (the response must begin with the "
                  "entire target string).")
            print("     Score the stored responses with `python -m experiments.rescore` "
                  "before concluding the attack failed.")
    if transfer_result:
        ts = transfer_result["summary"]
        print(f"\nTRANSFER ASR (suffix never optimised on these): "
              f"{ts['asr_prefix_match']:.0%} over n={ts['n_pairs']} prompt(s)")
    elif not args.per_prompt:
        print("\nTRANSFER ASR: n/a (only one prompt loaded)")

    print(f"\nCompute: {accounting.steps_run} steps | "
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

    for i, sample in enumerate((eval_result or {}).get("samples", [])):
        print(f"\n--- Attacked pair (index {sample.get('pair_index', i)}): "
              f"{sample['goal'][:80]} ---")
        print(f"  success={sample['success']} refusal={sample['refusal']}")
        print(f"  output: {sample['response'][:160]!r}")


if __name__ == "__main__":
    main()
