"""
Cross-model transfer experiment (the GCG paper's headline claim).

1. Optimize a suffix on SOURCE model A (Algorithm 1).
2. Carry it over as TEXT and re-tokenize with TARGET model B's tokenizer.
3. Measure ASR / target loss on B vs. an unoptimized-suffix baseline,
   and (optionally) vs. a suffix optimized directly on B (the upper bound).

    # Offline demo: two tiny random models with different tokenizers
    python -m experiments.run_transfer --tiny

    # Real pair (needs GPU + HF access)
    python -m experiments.run_transfer --preset paper --benchmark \
        --model Qwen/Qwen3-0.6B --target-model Qwen/Qwen3-1.7B
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch

from src.algorithm1 import run_gcg
from src.baselines import target_loss
from src.config import GCGConfig
from src.data import load_benchmark_pairs, load_pairs
from src.evaluate import evaluate_pairs
from src.model import init_suffix_ids, load_model_and_tokenizer, resolve_device
from src.provenance import RunAccounting, count_successes, describe_model, loss_floor
from src.token_manager import TokenManager


def parse_args():
    p = argparse.ArgumentParser(description="Cross-model suffix transfer")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None, help="SOURCE model (attack here)")
    p.add_argument("--target-model", default=None,
                   help="TARGET model (evaluate transfer here)")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", default=None)
    p.add_argument("--pair-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Benchmark pairs to evaluate the transferred suffix on")
    p.add_argument("--device", default="auto")
    # Was missing: without it a 7-8B source model will not fit on a free-tier
    # 16GB GPU, which silently blocked the whole transfer experiment. Both
    # models are quantized when this is set.
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantize BOTH source and target (needed for 7B on a T4)")
    p.add_argument("--gpu-rate-usd-per-hour", type=float, default=0.0)
    p.add_argument("--tiny", action="store_true",
                   help="Two tiny offline models (no HF download needed)")
    p.add_argument("--optimize-target", action="store_true",
                   help="Also optimize a suffix directly on the target model")
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def build_models(args, cfg, goals, targets):
    """Return (model_a, tok_a, model_b, tok_b, device, model_infos)."""
    if args.tiny:
        from src.tiny_model import build_tiny_model_and_tokenizer
        print("Building two tiny offline models (different seeds & vocabs) ...")
        model_a, tok_a = build_tiny_model_and_tokenizer(
            goals=goals, targets=targets, seed=cfg.seed, vocab_size=512,
        )
        model_b, tok_b = build_tiny_model_and_tokenizer(
            goals=goals, targets=targets, seed=cfg.seed + 1, vocab_size=451,
        )
        infos = {
            "source": describe_model(
                tiny=True, model_id=cfg.model_id, tokenizer=tok_a,
                seed=cfg.seed,
            ),
            "target": describe_model(
                tiny=True, model_id=args.target_model or "tiny",
                tokenizer=tok_b, seed=cfg.seed + 1,
            ),
        }
        return model_a, tok_a, model_b, tok_b, "cpu", infos

    device = resolve_device(args.device)
    target_id = args.target_model or cfg.model_id
    print(f"Loading SOURCE {cfg.model_id} ...")
    model_a, tok_a = load_model_and_tokenizer(
        cfg.model_id, cfg.dtype, args.device, args.load_in_4bit,
    )
    infos = {
        "source": describe_model(
            tiny=False, model_id=cfg.model_id, tokenizer=tok_a, seed=cfg.seed,
        ),
    }
    if target_id == cfg.model_id:
        # Same architecture: still a valid "self-transfer" sanity check,
        # but use a fresh copy so eval code paths are identical.
        print("Target model equals source model (self-transfer sanity check).")
        infos["target"] = dict(infos["source"])
        return model_a, tok_a, model_a, tok_a, device, infos
    print(f"Loading TARGET {target_id} ...")
    model_b, tok_b = load_model_and_tokenizer(
        target_id, cfg.dtype, args.device, args.load_in_4bit,
    )
    infos["target"] = describe_model(
        tiny=False, model_id=target_id, tokenizer=tok_b, seed=cfg.seed,
    )
    return model_a, tok_a, model_b, tok_b, device, infos


def main():
    args = parse_args()
    cfg = GCGConfig(preset=args.preset)
    if args.model:
        cfg.model_id = args.model
    if args.steps:
        cfg.num_steps = args.steps
    if args.num_prompts:
        cfg.num_eval_prompts = args.num_prompts
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

    t_load = time.perf_counter()
    model_a, tok_a, model_b, tok_b, device, model_infos = build_models(
        args, cfg, goals, targets,
    )
    model_load_s = time.perf_counter() - t_load

    # ---- 1. Optimize on SOURCE ----
    suffix_a = init_suffix_ids(tok_a, cfg.suffix_init, cfg.suffix_length, device)
    print(f"\nOptimizing on source ({cfg.num_steps} steps) for: {goal!r}")
    stats = {}
    best_a, losses_a = run_gcg(
        model_a, tok_a, goal, target, suffix_a.clone(),
        num_steps=cfg.num_steps, topk=cfg.topk,
        search_width=cfg.search_width, mini_batch_size=cfg.mini_batch_size,
        use_kv_cache=cfg.use_kv_cache, verbose=True, stats_out=stats,
    )
    suffix_text = tok_a.decode(best_a, skip_special_tokens=True)
    print(f"Source suffix: {suffix_text!r}")

    # ---- 2. Carry over as text, re-tokenize for TARGET ----
    suffix_b = init_suffix_ids(tok_b, suffix_text, cfg.suffix_length, device)
    suffix_b_text = tok_b.decode(suffix_b, skip_special_tokens=True)
    retok_ok = suffix_b.shape[0] == cfg.suffix_length
    print(f"Re-tokenized on target ({suffix_b.shape[0]} tokens): {suffix_b_text!r}")
    if suffix_b_text != suffix_text:
        print("⚠️ Round-trip differs (expected when tokenizers differ; the "
              "suffix is truncated/padded to the target length).")

    # ---- 3. Evaluate on TARGET ----
    initial_b = init_suffix_ids(tok_b, cfg.suffix_init, cfg.suffix_length, device)
    evaluation = {}
    for name, suffix in (("transferred", suffix_b), ("initial", initial_b)):
        ev = evaluate_pairs(
            model_b, tok_b, goals, targets, suffix,
            max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
            device=device,
        )
        tm = TokenManager(tok_b, goal, target, suffix, device)
        ev["mean_target_loss"] = target_loss(model_b, tm, suffix)
        evaluation[name] = ev

    if args.optimize_target:
        print("\nOptimizing directly on target (upper bound) ...")
        torch.manual_seed(cfg.seed)
        best_direct, losses_direct = run_gcg(
            model_b, tok_b, goal, target, initial_b.clone(),
            num_steps=cfg.num_steps, topk=cfg.topk,
            search_width=cfg.search_width, mini_batch_size=cfg.mini_batch_size,
            use_kv_cache=cfg.use_kv_cache, verbose=True,
        )
        ev = evaluate_pairs(
            model_b, tok_b, goals, targets, best_direct,
            max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
            device=device,
        )
        tm = TokenManager(tok_b, goal, target, best_direct, device)
        ev["mean_target_loss"] = target_loss(model_b, tm, best_direct)
        ev["suffix"] = tok_b.decode(best_direct, skip_special_tokens=True)
        ev["loss_first"] = losses_direct[0]
        ev["loss_last"] = losses_direct[-1]
        evaluation["optimized_on_target"] = ev

    # ---- Persist + summarize ----
    os.makedirs("results/reports/generated", exist_ok=True)
    n_successes = sum(count_successes(e) for e in evaluation.values())
    accounting = RunAccounting(
        device=device,
        steps_run=stats.get("steps_run", len(losses_a)),
        gradient_forwards=stats.get("gradient_forwards", 0),
        candidate_forwards=stats.get("candidate_forwards", 0),
        wall_clock_s=stats.get("wall_clock_s", 0.0),
        model_load_s=model_load_s,
        gpu_hourly_usd=args.gpu_rate_usd_per_hour,
        n_eval_pairs=len(goals) * len(evaluation),
        n_successes=n_successes,
        notes=(
            "wall-clock and forwards cover the SOURCE optimization only; the "
            "transfer evaluations are generation passes and their cost is not "
            "counted here"
        ),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "run_kind": "cross_model_transfer",
        "source_model": model_infos["source"],
        "target_model": model_infos["target"],
        "dataset_source": dataset_source,
        "preset": args.preset,
        "seed": cfg.seed,
        "num_steps": cfg.num_steps,
        "suffix_length": cfg.suffix_length,
        "load_in_4bit": args.load_in_4bit,
        "goal": goal,
        "target": target,
        "source_suffix": suffix_text,
        "retokenized_suffix": suffix_b_text,
        "retokenized_length_ok": retok_ok,
        "source_final_loss": losses_a[-1],
        "source_random_guess_floor": loss_floor(model_infos["source"]["vocab_size"]),
        "evaluation_on_target": evaluation,
        "accounting": accounting.as_dict(),
    }
    path = f"results/reports/generated/transfer_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"\nSaved run record to {path}")

    print("\n" + "=" * 66)
    print(f"Source (attacked) : {model_infos['source']['model_label']}")
    print(f"Target (transfer) : {model_infos['target']['model_label']}")
    print("Transfer evaluation on TARGET model:")
    print(f"{'variant':>20} | {'target loss':>11} | {'ASR':>5} | {'refusal':>7}")
    for name, e in evaluation.items():
        print(f"{name:>20} | {e['mean_target_loss']:>11.4f} | "
              f"{e['summary']['asr_prefix_match']:>4.0%} | "
              f"{e['summary']['refusal_rate']:>6.0%}")
    print("=" * 66)
    tl = evaluation["transferred"]["mean_target_loss"]
    il = evaluation["initial"]["mean_target_loss"]
    if tl < il:
        print(f"✅ Transferred suffix beats the unoptimized one on target "
              f"({tl:.4f} < {il:.4f})")
    else:
        print("ℹ️ No transfer gain on target (common on tiny/random models "
              "and small budgets; see the reports for real-model expectations).")
    print(f"Compute (source optimization): {accounting.total_forwards:,} forwards | "
          f"{accounting.wall_clock_s:.1f}s")
    if accounting.cost_per_success_usd is None:
        print("Cost per successful attack: UNDEFINED (0 successes in this run)")


if __name__ == "__main__":
    main()
