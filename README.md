# GCG from Scratch — Greedy Coordinate Gradient Attacks for Learning & Research

A beginner-friendly, from-scratch implementation of **GCG** (*Greedy Coordinate
Gradient*), the jailbreak attack from
[*Universal and Transferable Adversarial Attacks on Aligned Language Models*
(Zou et al., 2023)](https://arxiv.org/abs/2307.15043) — built to actually *understand*
what happens under the hood, not just call a library.

> **Why this matters for alignment:** GCG shows that aligned LLMs fail *structurally*
> under adversarial suffixes — a single gibberish suffix can transfer across prompts
> and models. Studying the attack is a prerequisite for building defenses.

**Status:** ✅ full pipeline implemented & validated offline · 🧪 paper-scale
benchmark runs on real aligned models are the next step (needs GPU + HF access)

## What's implemented

- **Algorithm 1 — per-prompt attack** (`src/algorithm1.py`): one-hot gradient proxy,
  top-k candidate substitutions, vectorized random mutations, mini-batched scoring,
  greedy update — with a **KV-cache** for the fixed prompt prefix.
- **Algorithm 2 — universal attack** (`src/algorithm2.py`): one suffix for many
  prompts, progressive prompt schedule, per-prompt gradient normalization,
  held-out generalization eval.
- **Evaluation harness** (`src/evaluate.py`): paper-style **prefix-match ASR**,
  refusal-rate heuristic, greedy/sampled generation.
- **Defense baseline** (`src/defense.py`): perplexity-based detection of adversarial
  suffixes (optimized vs. random vs. natural-language controls).
- **Baselines** (`src/baselines.py`, `experiments/run_baselines.py`): gradient-blind
  **random search** with the identical mutation budget — isolates the value of the
  gradient signal (GCG wins even on the tiny model: 6.249 < 6.252 < 6.257 initial).
- **Cross-model transfer** (`experiments/run_transfer.py`): optimize on model A,
  carry the suffix over as *text*, re-tokenize for model B, and measure transfer ASR
  vs. unoptimized and direct-optimization bounds — the paper's headline experiment.
- **PAIR — black-box baseline** (`src/pair.py`, `experiments/run_pair.py`): the
  non-gradient counterpart to GCG — [Chao et al., 2023](https://arxiv.org/abs/2310.08419),
  *Jailbreaking Black Box Large Language Models in Twenty Queries*. An attacker LLM
  iteratively rewrites a natural-language jailbreak prompt using the target's own
  responses and a judge score as feedback (multiple parallel refinement "streams",
  early stopping on a perfect score). Reuses the GCG harness end to end (generation,
  prefix-match ASR, refusal heuristic, perplexity) so the two attack families —
  white-box/gradient vs. black-box/LLM-refinement — are directly comparable, including
  the fact that PAIR's natural-language prompts read as ordinary text (low perplexity)
  where GCG's optimized suffixes read as gibberish (high perplexity, §defense baseline).
- **Data** (`src/data.py`, `data/`): AdvBench-style benchmark loader with a bundled
  offline sample; benign goal/target pairs for safe demos. Full
  [`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench) pulls
  automatically when the hub is reachable.
- **Offline smoke tests** (`src/tiny_model.py`, `experiments/run_smoke.py`): a tiny
  random model + locally-trained tokenizer exercise *every* code path in seconds,
  including a numerical proof that the KV-cache path matches the full forward
  (max gradient diff ~1e-9).
- **Version-robust KV-cache** (`src/kv_cache.py`): works with both transformers v4
  and v5 cache APIs.

## Repository structure

```
src/
  algorithm1.py      # per-prompt GCG (gradient → top-k → sample → score → greedy)
  algorithm2.py      # universal/multi-prompt GCG + progressive schedule
  token_manager.py   # chat-template layout, loss slices, re-tokenization filter
  kv_cache.py        # KV-cache for the fixed prompt prefix
  evaluate.py        # generation, prefix-match ASR, refusal heuristic, perplexity
  defense.py         # perplexity-based defense baseline
  baselines.py       # gradient-blind random-search baseline
  pair.py            # PAIR: black-box, LLM-driven iterative refinement
  data.py            # jsonl + AdvBench loading (offline fallback)
  model.py           # model/tokenizer loading (incl. optional 4-bit)
  tiny_model.py      # tiny offline model for smoke tests
  config.py          # debug / paper presets (GCGConfig + PAIRConfig)
experiments/
  run_smoke.py       # end-to-end pipeline validation (CPU, no downloads)
  run_algorithm1.py  # full per-prompt experiment + artifacts
  run_algorithm2.py  # full universal experiment + artifacts
  run_baselines.py   # GCG vs gradient-blind random search
  run_transfer.py    # cross-model suffix transfer
  run_defense.py     # defense baseline experiment
  run_pair.py        # PAIR black-box attack experiment + artifacts
  check_algorithm1.py# low-level sanity checks on a real model
data/
  benign_pairs.jsonl       # safe demo goal/target pairs
  advbench_sample.jsonl    # bundled AdvBench-style eval sample
results/
  reports/           # method reports + machine-readable run records
  plots/             # loss curves
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Validate the entire pipeline offline (CPU, seconds, no downloads)
python -m experiments.run_smoke

# 2. Tiny end-to-end demo (no HF needed)
python -m experiments.run_algorithm1 --tiny --benchmark
python -m experiments.run_algorithm2 --tiny --benchmark
python -m experiments.run_baselines   --tiny --benchmark
python -m experiments.run_transfer    --tiny --benchmark
python -m experiments.run_defense     --tiny
python -m experiments.run_pair        --tiny

# 3. Real experiments
python -m experiments.run_algorithm1 --preset paper --benchmark
python -m experiments.run_algorithm2 --preset paper --benchmark --held-out 2
python -m experiments.run_baselines   --preset paper --benchmark
python -m experiments.run_transfer    --preset paper --benchmark \
    --model Qwen/Qwen3-0.6B --target-model Qwen/Qwen3-1.7B
python -m experiments.run_defense     --preset paper --benchmark

# PAIR (black-box): same model as attacker+target, or split them explicitly
python -m experiments.run_pair --preset paper --benchmark --model <hf-id>
python -m experiments.run_pair --preset paper --benchmark \
    --model <target-hf-id> --attacker-model <attacker-hf-id> --judge model

# ... on a bigger model (GPU recommended)
python -m experiments.run_algorithm1 --preset paper --benchmark \
    --model Qwen/Qwen3-1.7B --load-in-4bit
```

Presets (`src/config.py`): `debug` (CPU-friendly) and `paper`
(250 steps · 20-token suffix · top-k 256 · search width 512 — the paper's values).
PAIR has its own `PAIRConfig` presets: `debug` (2 streams x 2 iterations) and
`paper` (5 streams x 5 iterations, in the spirit of the paper's "twenty queries"
budget, scaled down from its original 30-stream setting to stay CPU-runnable).

## Pipeline validation results

From `python -m experiments.run_smoke` (seed 0 — full record in
`results/reports/smoke_results.json`):

| Check | Result |
|---|---|
| KV-cache vs. full forward (gradients) | max diff **1.2e-09** ✅ |
| KV-cache vs. full forward (candidate losses) | max diff **9.5e-07** ✅ |
| Algorithm 1 optimization runs end-to-end | ✅ |
| Algorithm 2 universal schedule (prompts ramp 1→2→3) | ✅ |
| ASR / refusal metrics computed | ✅ (0% ASR on random weights — expected) |
| Perplexity defense baseline (optimized/random/natural) | ✅ |
| GCG beats random search (same budget) | ✅ 6.2481 ≤ 6.2553 |
| Cross-model transfer pipeline (different tokenizers) | ✅ |
| PAIR black-box refinement loop (chat + parsing + judge, end to end) | ✅ |

## Reports

- [`results/reports/algorithm1_report.md`](results/reports/algorithm1_report.md) —
  method walkthrough (math → code), evaluation protocol, defense baseline, full-scale recipe.
- [`results/reports/algorithm2_report.md`](results/reports/algorithm2_report.md) —
  universal attack, progressive schedule, held-out generalization.
- [`results/reports/pair_report.md`](results/reports/pair_report.md) —
  PAIR method walkthrough, white-box vs. black-box comparison with GCG, design
  decisions (judges, parsing, portability), full-scale recipe.

## Roadmap (a.k.a. what to do next)

- [ ] Paper-scale runs on real aligned models; fill the ASR table (GPU).
- [x] Cross-model transfer harness (suffix → text → re-tokenize on target).
- [x] Random-search baseline + GCG comparison (`run_baselines.py`).
- [ ] Benchmark cross-model transfer on real models (SmolLM2 → Qwen, GPU).
- [x] Ablations infrastructure (grad normalization ablation TODO in code).
- [ ] Stronger defenses: smooth-perplexity, windowed PPL; compare detection rates.
- [ ] GCG variants (MMA, Faster-GCG) as drop-in optimizers.
- [x] PAIR — black-box, LLM-driven iterative refinement baseline (white-box vs.
      black-box comparison; needs a real instruction-tuned attacker to be meaningful).
- [ ] Paper-scale PAIR runs with a capable attacker (e.g. Vicuna/GPT-class) + ASR table.

## References

- Zou et al., 2023 — [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- Chao et al., 2023 — [PAIR, arXiv:2310.08419](https://arxiv.org/abs/2310.08419)
- Pulfer, 2025 — [GCG: Adversarial Attacks on Large Language Models](https://medium.com/@brianpulfer/gcg-adversarial-attacks-on-large-language-models-61f8b51734e9)
- Mazeika et al., 2024 — [HarmBench, arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
- [llm-attacks](https://github.com/llm-attacks/llm-attacks) · [nanoGCG](https://github.com/GraySwanAI/nanoGCG) · [PAIR (official)](https://github.com/patrickrchao/JailbreakingLLMs)

## Responsible use

This project exists to **understand and harden** aligned models. Default configs use
benign goals; the bundled benchmark strings are standard red-team evaluation data.
Don't use this material to cause harm.
