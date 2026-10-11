"""
PAIR: black-box jailbreaking via LLM-driven iterative refinement.

Unlike GCG (white-box, gradient-guided token optimization), PAIR never
touches the target model's weights or gradients: an attacker LLM rewrites a
natural-language prompt turn by turn, using the target's own responses and a
judge score as feedback (Chao et al., 2023, arXiv:2310.08419).

    python -m experiments.run_pair --tiny
    python -m experiments.run_pair --preset paper --benchmark --model <hf-id>
    python -m experiments.run_pair --model <target-id> --attacker-model <attacker-id> --judge model
"""
import argparse
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import PAIRConfig
from src.data import load_benchmark_pairs, load_pairs
from src.evaluate import contains_refusal, is_target_prefix_match, text_perplexity
from src.model import load_model_and_tokenizer, resolve_device
from src.pair import chat_generate, run_pair


def parse_args():
    p = argparse.ArgumentParser(description="PAIR black-box jailbreak attack")
    p.add_argument("--preset", default="debug", choices=["debug", "paper"])
    p.add_argument("--model", default=None, help="target model id")
    p.add_argument("--attacker-model", default=None, help="attacker model id (defaults to --model)")
    p.add_argument("--judge", default=None, choices=["keyword", "model"])
    p.add_argument("--judge-model", default=None, help="judge model id (only used with --judge model)")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", default=None)
    p.add_argument("--pair-index", type=int, default=0)
    p.add_argument("--n-streams", type=int, default=None)
    p.add_argument("--n-iterations", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tiny", action="store_true",
                   help="Use the tiny offline model (no HF download needed)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PAIRConfig(preset=args.preset)
    if args.model:
        cfg.model_id = args.model
    if args.attacker_model:
        cfg.attacker_model_id = args.attacker_model
    if args.judge:
        cfg.judge = args.judge
    if args.judge_model:
        cfg.judge_model_id = args.judge_model
    if args.n_streams:
        cfg.n_streams = args.n_streams
    if args.n_iterations:
        cfg.n_iterations = args.n_iterations
    torch.manual_seed(cfg.seed)

    # ---- Data ----
    if args.benchmark:
        goals, targets = load_benchmark_pairs(n=cfg.num_eval_prompts)
    elif args.data:
        goals, targets = load_pairs(args.data)
    else:
        goals, targets = cfg.goals, cfg.targets
    goal, target = goals[args.pair_index], targets[args.pair_index]

    # ---- Models ----
    same_attacker = cfg.attacker_model_id is None or cfg.attacker_model_id == cfg.model_id
    if args.tiny:
        from src.tiny_model import build_tiny_model_and_tokenizer
        print("Building tiny offline model (random weights) ...")
        target_model, target_tokenizer = build_tiny_model_and_tokenizer(
            goals=goals, targets=targets, seed=cfg.seed,
        )
        attacker_model, attacker_tokenizer = target_model, target_tokenizer
        device = "cpu"
    else:
        print(f"Loading target model {cfg.model_id} ...")
        target_model, target_tokenizer = load_model_and_tokenizer(
            cfg.model_id, cfg.dtype, args.device, cfg.load_in_4bit,
        )
        device = resolve_device(args.device)
        if same_attacker:
            attacker_model, attacker_tokenizer = target_model, target_tokenizer
        else:
            print(f"Loading attacker model {cfg.attacker_model_id} ...")
            attacker_model, attacker_tokenizer = load_model_and_tokenizer(
                cfg.attacker_model_id, cfg.dtype, args.device, cfg.load_in_4bit,
            )

    judge_model, judge_tokenizer = None, None
    if cfg.judge == "model" and cfg.judge_model_id and cfg.judge_model_id not in (
        cfg.attacker_model_id, cfg.model_id,
    ):
        print(f"Loading judge model {cfg.judge_model_id} ...")
        judge_model, judge_tokenizer = load_model_and_tokenizer(
            cfg.judge_model_id, cfg.dtype, args.device, cfg.load_in_4bit,
        )

    print(f"Goal:   {goal}\nTarget: {target!r}")
    print(f"Budget: {cfg.n_streams} streams x {cfg.n_iterations} iterations "
          f"(<= {cfg.n_streams * cfg.n_iterations} target queries), judge={cfg.judge}\n")

    best_prompt, streams = run_pair(
        target_model, target_tokenizer, goal, target,
        attacker_model=attacker_model, attacker_tokenizer=attacker_tokenizer,
        n_streams=cfg.n_streams, n_iterations=cfg.n_iterations,
        judge=cfg.judge, judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        max_new_tokens_attacker=cfg.max_new_tokens_attacker,
        max_new_tokens_target=cfg.max_new_tokens_target,
        temperature_attacker=cfg.temperature_attacker,
        temperature_target=cfg.temperature_target,
        early_stop_score=cfg.early_stop_score,
        device=device, seed=cfg.seed, verbose=True,
    )

    # ---- Final evaluation of the winning prompt ----
    final_response = chat_generate(
        target_model, target_tokenizer,
        [{"role": "user", "content": best_prompt}],
        device, cfg.max_new_tokens_target, temperature=0.0,
    )
    success = is_target_prefix_match(final_response, target)
    refusal = contains_refusal(final_response)
    # "Does the winning prompt look like gibberish?" -- the natural-language
    # counterpart to GCG's suffix-perplexity defense diagnostic. PAIR prompts
    # should look like ordinary (if devious) English, unlike GCG's suffixes.
    prompt_ppl = text_perplexity(target_model, target_tokenizer, best_prompt, device)
    goal_ppl = text_perplexity(target_model, target_tokenizer, goal, device)

    n_queries = sum(len(s) for s in streams)

    # ---- Plot: best-score-so-far per stream ----
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports/generated", exist_ok=True)

    plt.figure(figsize=(8, 4))
    for i, stream in enumerate(streams):
        scores = [r["score"] for r in stream]
        plt.plot(range(1, len(scores) + 1), scores, marker="o", alpha=0.6, label=f"stream {i}")
    plt.xlabel("Iteration")
    plt.ylabel("Judge score (1-10)")
    plt.ylim(0, 11)
    plt.title(f"PAIR — {cfg.model_id} ({args.preset}, judge={cfg.judge})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/pair_scores.png", dpi=150)
    print("Saved plot to results/plots/pair_scores.png")

    # ---- Persist + summarize ----
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "model_id": cfg.model_id,
        "attacker_model_id": cfg.attacker_model_id or cfg.model_id,
        "preset": args.preset,
        "judge": cfg.judge,
        "seed": cfg.seed,
        "n_streams": cfg.n_streams,
        "n_iterations": cfg.n_iterations,
        "n_queries": n_queries,
        "goal": goal,
        "target": target,
        "best_prompt": best_prompt,
        "final_response": final_response,
        "success_prefix_match": success,
        "refusal": refusal,
        "best_prompt_perplexity": prompt_ppl,
        "goal_perplexity": goal_ppl,
        "streams": streams,
    }
    path = f"results/reports/generated/pair_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"Saved run record to {path}")

    print("\n" + "=" * 64)
    print(f"Queries used:     {n_queries} (<= {cfg.n_streams * cfg.n_iterations} budget)")
    print(f"Best prompt:      {best_prompt!r}")
    print(f"Final response:   {final_response[:200]!r}")
    print(f"Success (ASR):    {success}")
    print(f"Refusal detected: {refusal}")
    print(f"Prompt PPL:       {prompt_ppl:.2f}  (plain goal PPL: {goal_ppl:.2f})")
    print("=" * 64)
    if success:
        print("✅ PAIR found a prompt whose response matches the target prefix.")
    else:
        print("⚠️ No full prefix match this run — try more --n-streams/--n-iterations, "
              "a stronger --attacker-model, or --judge model.")


if __name__ == "__main__":
    main()
