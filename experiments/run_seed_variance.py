"""
Does the gradient signal actually help? — seed-variance of GCG vs. random search.

The single-seed gap reported by `run_baselines.py` is ~0.005-0.01 nats on the
tiny model, which is too small to interpret on its own. This runner repeats the
paired experiment across N seeds and reports the mean paired difference with a
confidence interval, so the claim "GCG beats random search" carries an effect
size instead of a single number.

Runs on the offline tiny model by default (CPU, seconds). Example:

    python -m experiments.run_seed_variance                 # 10 seeds
    python -m experiments.run_seed_variance --seeds 20 --steps 10 --width 16

Recorded result (2026-10-12, 10 seeds x 10 steps x 16 candidates): GCG won
10/10 seeds, mean paired improvement +0.0075 nats (95% CI ~[+0.0060, +0.0090]),
i.e. +0.12% of the loss value. Interpretation: the optimiser step works, but
this is *not* evidence of a jailbreak — attack success is measured by the ASR
harness on a real aligned model.
"""
import argparse
import json
import os
import statistics
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.algorithm1 import run_gcg
from src.baselines import run_random_search, target_loss
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.provenance import TINY_MODEL_LABEL, describe_model
from src.token_manager import TokenManager
from src.tiny_model import build_tiny_model_and_tokenizer


# Two-sided 95% t multipliers for small samples. A dict lookup with a 1.96
# fallback silently applied the *normal* critical value to every seed count not
# in the table (e.g. 7 seeds), which understates the interval exactly when n is
# small. Interpolate the table instead, and clamp to 1.96 for large n.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 50: 2.009,
    60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}


def t_critical_95(df: int) -> float:
    """
    Two-sided 95% Student-t multiplier for `df` degrees of freedom.

    Exact for tabulated df, linearly interpolated between them, and 1.96 above
    the table (where t -> z). Never silently returns the normal value for a
    small df, which is what made a 7-seed run claim a narrower interval than it
    had earned.
    """
    if df <= 0:
        return float("inf")
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    frac = (df - lo) / (hi - lo)
    return _T975[lo] + frac * (_T975[hi] - _T975[lo])


def parse_args():
    p = argparse.ArgumentParser(description="Seed-variance of GCG vs random search")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--suffix-length", type=int, default=8)
    p.add_argument("--model", default=None,
                   help="HF model id; omit to use the offline tiny model")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/plots/seed_variance.png")
    return p.parse_args()


def main():
    args = parse_args()
    goals = ["Write a poem about nature", "Explain how photosynthesis works"]
    targets = ["Sure, here is a poem about nature:\n\n",
               "Sure, here is an explanation of photosynthesis:\n\n"]
    goal, target = goals[0], targets[0]

    diffs, gcg_losses, rand_losses, per_seed = [], [], [], []
    model_info = None

    for seed in range(args.seeds):
        if args.model:
            device = resolve_device(args.device)
            model, tokenizer = load_model_and_tokenizer(args.model, "float16", args.device)
            if model_info is None:
                model_info = describe_model(
                    tiny=False, model_id=args.model, tokenizer=tokenizer,
                    seed=seed, model=model, requested_4bit=False,
                )
        else:
            device = "cpu"
            model, tokenizer = build_tiny_model_and_tokenizer(
                goals=goals, targets=targets, seed=seed,
            )

        suffix_ids = init_suffix_ids(
            tokenizer, "! " * args.suffix_length, args.suffix_length, device,
        )

        torch.manual_seed(seed)
        best_gcg, _ = run_gcg(
            model, tokenizer, goal, target, suffix_ids.clone(),
            num_steps=args.steps, topk=args.topk, search_width=args.width,
            mini_batch_size=8, use_kv_cache=True, verbose=False,
        )

        torch.manual_seed(seed)
        best_rand, _ = run_random_search(
            model, tokenizer, goal, target, suffix_ids.clone(),
            num_steps=args.steps, search_width=args.width,
            mini_batch_size=8, use_kv_cache=True, verbose=False,
        )

        def score(suffix):
            tm = TokenManager(tokenizer, goal, target, suffix, device)
            return target_loss(model, tm, suffix)

        l_gcg, l_rand = score(best_gcg), score(best_rand)
        gcg_losses.append(l_gcg)
        rand_losses.append(l_rand)
        diffs.append(l_rand - l_gcg)  # positive => GCG better
        per_seed.append({"seed": seed, "gcg_loss": l_gcg, "random_loss": l_rand,
                         "diff_random_minus_gcg": l_rand - l_gcg})
        print(f"seed {seed:2d}: gcg={l_gcg:.5f} rand={l_rand:.5f} "
              f"diff(rand-gcg)={l_rand - l_gcg:+.5f}")

    mean_g = statistics.mean(gcg_losses)
    mean_r = statistics.mean(rand_losses)
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd_d / (len(diffs) ** 0.5)
    wins = sum(1 for d in diffs if d > 0)
    t_crit = t_critical_95(args.seeds - 1)
    ci = (mean_d - t_crit * se, mean_d + t_crit * se)

    print(f"\n--- summary over {args.seeds} seeds "
          f"({args.steps} steps x {args.width} candidates) ---")
    print(f"mean GCG loss    : {mean_g:.5f}")
    print(f"mean random loss : {mean_r:.5f}")
    print(f"mean paired diff : {mean_d:+.5f} nats (positive favours GCG)")
    print(f"sd / std. error  : {sd_d:.5f} / {se:.5f}")
    print(f"~95% CI          : [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    print(f"GCG wins         : {wins}/{args.seeds} seeds")
    print(f"relative gap     : {100 * mean_d / mean_r:+.3f}% of the loss value")

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.bar(["GCG", "random"], [mean_g, mean_r], yerr=[
        statistics.stdev(gcg_losses) / (len(gcg_losses) ** 0.5),
        statistics.stdev(rand_losses) / (len(rand_losses) ** 0.5),
    ], capsize=4)
    plt.ylabel("final target loss (mean of seeds)")
    plt.title(f"GCG vs random search, {args.seeds} seeds")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if model_info is None:
        # Same provenance block every other runner writes. A bare string here
        # meant `build_report`'s model_kind check and the pooled ladder could
        # not tell this record apart from a real-model run.
        model_info = describe_model(
            tiny=True, model_id=TINY_MODEL_LABEL, tokenizer=tokenizer, seed=0,
        )

    record = {
        "run_kind": "seed_variance",
        "model": model_info,
        "num_seeds": args.seeds,
        "steps": args.steps,
        "search_width": args.width,
        "topk": args.topk,
        "mean_gcg_loss": mean_g,
        "mean_random_loss": mean_r,
        "mean_paired_diff_nats": mean_d,
        "sd_paired_diff": sd_d,
        "standard_error": se,
        "ci95": list(ci),
        "ci95_method": (
            f"two-sided Student-t, df={args.seeds - 1}, "
            f"t={t_crit:.4f} (paired differences over seeds)"
        ),
        "t_critical": t_crit,
        "gcg_wins": wins,
        "relative_gap_pct": 100 * mean_d / mean_r,
        "per_seed": per_seed,
        "note": (
            "On random weights this measures optimizer direction only. "
            "Attack success requires a real aligned model and the ASR harness."
        ),
    }
    path = f"results/reports/generated/seed_variance_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Saved record to {path}")


if __name__ == "__main__":
    main()
