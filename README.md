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
  data.py            # jsonl + AdvBench loading (offline fallback)
  model.py           # model/tokenizer loading (incl. optional 4-bit)
  tiny_model.py      # tiny offline model for smoke tests
  config.py          # debug / paper presets
experiments/
  run_smoke.py       # end-to-end pipeline validation (CPU, no downloads)
  run_algorithm1.py  # full per-prompt experiment + artifacts
  run_algorithm2.py  # full universal experiment + artifacts
  run_defense.py     # defense baseline experiment
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
python -m experiments.run_defense --tiny

# 3. Real experiments
python -m experiments.run_algorithm1 --preset paper --benchmark
python -m experiments.run_algorithm2 --preset paper --benchmark --held-out 2
python -m experiments.run_defense   --preset paper --benchmark

# ... on a bigger model (GPU recommended)
python -m experiments.run_algorithm1 --preset paper --benchmark \
    --model Qwen/Qwen3-1.7B --load-in-4bit
```

Presets (`src/config.py`): `debug` (CPU-friendly) and `paper`
(250 steps · 20-token suffix · top-k 256 · search width 512 — the paper's values).

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

## Reports

- [`results/reports/algorithm1_report.md`](results/reports/algorithm1_report.md) —
  method walkthrough (math → code), evaluation protocol, defense baseline, full-scale recipe.
- [`results/reports/algorithm2_report.md`](results/reports/algorithm2_report.md) —
  universal attack, progressive schedule, held-out generalization.

## Roadmap (a.k.a. what to do next)

- [ ] Paper-scale runs on real aligned models; fill the ASR table (GPU).
- [ ] Cross-model transfer of universal suffixes (SmolLM2 → Qwen).
- [ ] Baselines: random search, no-attack; ablations (grad normalization, schedule).
- [ ] Stronger defenses: smooth-perplexity, windowed PPL; compare detection rates.
- [ ] GCG variants (MMA, Faster-GCG) as drop-in optimizers.

## References

- Zou et al., 2023 — [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- Pulfer, 2025 — [GCG: Adversarial Attacks on Large Language Models](https://medium.com/@brianpulfer/gcg-adversarial-attacks-on-large-language-models-61f8b51734e9)
- Mazeika et al., 2024 — [HarmBench, arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
- [llm-attacks](https://github.com/llm-attacks/llm-attacks) · [nanoGCG](https://github.com/GraySwanAI/nanoGCG)

## Responsible use

This project exists to **understand and harden** aligned models. Default configs use
benign goals; the bundled benchmark strings are standard red-team evaluation data.
Don't use this material to cause harm.
