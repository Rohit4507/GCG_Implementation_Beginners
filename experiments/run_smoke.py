"""
Offline smoke test for the entire GCG pipeline.

Builds a tiny random model (src/tiny_model.py) and exercises:
  1. TokenManager layout / slices / re-tokenization filter
  2. Algorithm 1 with and without KV-cache (losses must match)
  3. Algorithm 2 (universal, progressive prompt schedule)
  4. Evaluation harness (prefix-match ASR + refusal heuristic)
  5. Perplexity defense baseline

Runs in well under a minute on CPU. Any assertion failure = regression.

    python -m experiments.run_smoke
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.algorithm1 import run_gcg
from src.algorithm2 import run_universal_gcg
from src.data import load_benign_pairs, load_benchmark_pairs
from src.defense import defense_experiment
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids
from src.tiny_model import build_tiny_model_and_tokenizer

RESULTS_DIR = "results"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

SEED = 0
STEPS = 10
SUFFIX_LEN = 8
TOPK = 8
SEARCH_WIDTH = 16


def main():
    torch.manual_seed(SEED)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    benign_goals, benign_targets = load_benign_pairs()
    bench_goals, bench_targets = load_benchmark_pairs(n=2, prefer_hub=False)

    print("Building tiny model + tokenizer (offline)...")
    model, tokenizer = build_tiny_model_and_tokenizer(
        goals=benign_goals + bench_goals,
        targets=benign_targets + bench_targets,
        seed=SEED,
    )
    device = "cpu"

    results = {"seed": SEED, "steps": STEPS, "suffix_length": SUFFIX_LEN, "checks": {}}

    # ------------------------------------------------------------------
    # 1. Algorithm 1: KV-cache path must match the plain path
    # ------------------------------------------------------------------
    goal, target = benign_goals[0], benign_targets[0]
    suffix_init = init_suffix_ids(tokenizer, "! " * SUFFIX_LEN, SUFFIX_LEN, device)

    # 1a. Deterministic equality on IDENTICAL inputs: gradients and losses.
    from src.algorithm1 import compute_token_gradient, evaluate_candidates, sample_candidates
    from src.kv_cache import PrefixCache
    from src.token_manager import TokenManager

    tm = TokenManager(tokenizer, goal, target, suffix_init, device)
    pcache = PrefixCache(model, tm.before_suffix_ids)

    grad_p, loss_p = compute_token_gradient(model, tm, suffix_init, None)
    grad_k, loss_k = compute_token_gradient(model, tm, suffix_init, pcache)
    grad_diff = (grad_p - grad_k).abs().max().item()
    loss_diff = abs(loss_p - loss_k)

    torch.manual_seed(SEED)
    cands = sample_candidates(suffix_init, grad_p, TOPK, SEARCH_WIDTH)
    losses_p = evaluate_candidates(model, tm, cands, 8, None)
    losses_k = evaluate_candidates(model, tm, cands, 8, pcache)
    cand_diff = (losses_p - losses_k).abs().max().item()

    kv_ok = grad_diff < 1e-3 and loss_diff < 1e-3 and cand_diff < 1e-3
    assert kv_ok, (
        f"KV-cache path diverged: grad {grad_diff:.2e}, "
        f"loss {loss_diff:.2e}, candidates {cand_diff:.2e}"
    )

    # 1b. Full runs with the SAME seed (identical sampled candidates),
    # for a timing comparison and trajectory log.
    torch.manual_seed(SEED)
    t0 = time.time()
    best_a, losses_a = run_gcg(
        model, tokenizer, goal, target, suffix_init.clone(),
        num_steps=STEPS, topk=TOPK, search_width=SEARCH_WIDTH,
        mini_batch_size=8, use_kv_cache=False, verbose=False,
    )
    t_plain = time.time() - t0

    torch.manual_seed(SEED)
    t0 = time.time()
    best_b, losses_b = run_gcg(
        model, tokenizer, goal, target, suffix_init.clone(),
        num_steps=STEPS, topk=TOPK, search_width=SEARCH_WIDTH,
        mini_batch_size=8, use_kv_cache=True, verbose=False,
    )
    t_kv = time.time() - t0

    traj_diff = max(abs(a - b) for a, b in zip(losses_a, losses_b))
    results["checks"]["kv_cache_matches_plain"] = {
        "pass": kv_ok,
        "grad_max_abs_diff": grad_diff,
        "grad_loss_abs_diff": loss_diff,
        "candidate_loss_max_abs_diff": cand_diff,
        "trajectory_max_abs_diff": traj_diff,
        "time_plain_s": round(t_plain, 2),
        "time_kv_cache_s": round(t_kv, 2),
        "loss_first": losses_a[0],
        "loss_last": losses_a[-1],
    }
    print(f"[1] Algorithm 1 KV-cache vs plain: grad|Δ|={grad_diff:.2e} "
          f"loss|Δ|={loss_diff:.2e} cand|Δ|={cand_diff:.2e} ({'PASS' if kv_ok else 'FAIL'}) "
          f"[plain {t_plain:.1f}s / kv {t_kv:.1f}s]")

    # Loss curve plot (Algorithm 1)
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, len(losses_a) + 1), losses_a, label="step loss")
    plt.xlabel("Step")
    plt.ylabel("Target cross-entropy")
    plt.title("GCG Algorithm 1 — smoke run (tiny random model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "smoke_loss_individual.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 2. Algorithm 2: universal suffix over 3 benign prompts
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    suffix_u = init_suffix_ids(tokenizer, "! " * SUFFIX_LEN, SUFFIX_LEN, device)
    best_u, losses_u = run_universal_gcg(
        model, tokenizer,
        goals=benign_goals[:3], targets=benign_targets[:3],
        suffix_ids=suffix_u,
        num_steps=STEPS, topk=TOPK, search_width=SEARCH_WIDTH,
        mini_batch_size=8, num_prompts_start=1, prompt_add_interval=3,
        use_kv_cache=True, verbose=False,
    )
    uni_ok = len(losses_u) == STEPS and all(
        isinstance(x, tuple) and isinstance(x[1], float) for x in losses_u
    )
    results["checks"]["algorithm2_universal"] = {
        "pass": uni_ok,
        "loss_first": losses_u[0][1],
        "loss_last": losses_u[-1][1],
    }
    print(f"[2] Algorithm 2 universal loop: {STEPS} steps "
          f"loss {losses_u[0][1]:.3f} -> {losses_u[-1][1]:.3f} "
          f"({'PASS' if uni_ok else 'FAIL'})")

    steps_u, loss_vals_u = zip(*losses_u)
    plt.figure(figsize=(8, 3.5))
    plt.plot(steps_u, loss_vals_u, label="mean loss (active prompts)")
    plt.xlabel("Step")
    plt.ylabel("Mean cross-entropy")
    plt.title("GCG Algorithm 2 (universal) — smoke run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "smoke_loss_universal.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 3. Evaluation harness (ASR + refusal heuristic)
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    bench_suffix = init_suffix_ids(tokenizer, "! " * SUFFIX_LEN, SUFFIX_LEN, device)
    best_bench, losses_bench = run_gcg(
        model, tokenizer, bench_goals[0], bench_targets[0], bench_suffix,
        num_steps=STEPS, topk=TOPK, search_width=SEARCH_WIDTH,
        mini_batch_size=8, use_kv_cache=True, verbose=False,
    )
    ev = evaluate_pairs(
        model, tokenizer, bench_goals, bench_targets, best_bench,
        max_new_tokens=24, temperature=0.0, device=device,
    )
    ev_ok = {"n_pairs", "asr_prefix_match", "refusal_rate"} <= set(ev["summary"])
    results["checks"]["evaluation_harness"] = {
        "pass": ev_ok,
        "summary": ev["summary"],
        "note": ("Tiny random model: ASR is expected to be ~0; this check "
                 "only validates that the harness computes metrics."),
    }
    print(f"[3] Evaluation harness: ASR={ev['summary']['asr_prefix_match']:.0%} "
          f"refusal_rate={ev['summary']['refusal_rate']:.0%} "
          f"({'PASS' if ev_ok else 'FAIL'})")

    # ------------------------------------------------------------------
    # 4. Perplexity defense baseline
    # ------------------------------------------------------------------
    def_exp = defense_experiment(
        model, tokenizer, bench_goals[0], bench_targets[0], best_bench, device,
    )
    ppl = def_exp["perplexities"]
    def_ok = all(k in ppl for k in ("optimized", "random", "natural"))
    results["checks"]["perplexity_defense"] = {
        "pass": def_ok,
        **def_exp,
    }
    print(f"[4] Defense baseline: PPL optimized={ppl['optimized']:.1f} "
          f"random={ppl['random']:.1f} natural={ppl['natural']:.1f} "
          f"threshold={def_exp['threshold']:.1f} ({'PASS' if def_ok else 'FAIL'})")

    # ------------------------------------------------------------------
    # Write machine-readable results
    # ------------------------------------------------------------------
    out_json = os.path.join(REPORTS_DIR, "smoke_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_pass = sum(1 for c in results["checks"].values() if c["pass"])
    n_total = len(results["checks"])
    print("\n" + "=" * 60)
    print(f"SMOKE RESULT: {n_pass}/{n_total} checks passed")
    print(f"Details: {out_json}")
    print(f"Plots:   {PLOTS_DIR}/smoke_loss_*.png")
    if n_pass != n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
