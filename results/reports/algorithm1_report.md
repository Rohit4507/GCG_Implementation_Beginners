# GCG Algorithm 1 — Per-Prompt Attack (Individual Suffix)

**TL;DR** — This report documents our from-scratch implementation of Algorithm 1 of
*Universal and Transferable Adversarial Attacks on Aligned Language Models*
(Zou et al., 2023): the Greedy Coordinate Gradient (GCG) attack that optimizes an
adversarial suffix for **one** goal/target pair. We validate the pipeline end-to-end
offline (gradients, KV-cache, search loop, evaluation, defense baseline) and document
the exact recipe for full-scale benchmark runs.

---

## 1. Objective & threat model

We are given:

- a **goal** `x` — a user request an aligned model should refuse (or a benign request),
- a **target** `y*` — the response we want to elicit (typically an affirmative prefix
  such as *"Sure, here is ..."*),
- a **suffix** `s = (s_1, ..., s_L)` of `L` discrete tokens appended to the goal.

We optimize `s` to **minimize the target-token cross-entropy**:

```
L(s) = - Σ_t log p_θ(y*_t | template(x ⊕ s), y*_<t)
```

where `template(·)` is the model's chat template and the loss is computed **only on
the target tokens** (everything else is masked). Because tokens are discrete, we
cannot descend the loss directly — GCG uses gradients as a *proposal distribution*
over token substitutions.

## 2. Method (mapped to the code)

| Paper step | Code | Notes |
|---|---|---|
| 1. Gradient w.r.t. suffix tokens | `src/algorithm1.py::compute_token_gradient` | One-hot proxy `z ∈ {0,1}^{L×V}` with `z.requires_grad_()`, embedded via `z @ E`. `∇_z L` tells us which token substitutions would reduce the loss. |
| 2. Top-k candidates per position | `sample_candidates` | For each suffix position `i`, keep the `k` tokens with the most negative gradient (`(-grad).topk(k)`). |
| 3. Random single-token mutations | `sample_candidates` | Sample `B` suffixes, each changing exactly one random position with one of that position's top-k tokens (one candidate keeps the current suffix). |
| 4. Re-tokenization filter | `src/token_manager.py::filter_candidates` | Decode → re-encode must round-trip; keeps substitutions "template-safe". |
| 5. Score candidates | `evaluate_candidates` | Mini-batched forward passes (no gradients needed), loss per candidate. |
| 6. Greedy update | `run_gcg` | Keep `argmin` candidate; track the best suffix ever seen. |

**KV-cache acceleration** (`src/kv_cache.py::PrefixCache`): everything before the
suffix never changes, so it is forwarded **once**; every gradient and scoring step
reuses its key/value cache. Verified numerically equivalent to the full forward
(§4). This is the trick from Brian Pulfer's
[GCG write-up](https://medium.com/@brianpulfer/gcg-adversarial-attacks-on-large-language-models-61f8b51734e9).

**Prompt layout** (`src/token_manager.py`): `[chat prefix + goal] [suffix] [chat suffix / assistant prompt] [target]`.
The suffix/target boundaries come from a text sentinel inside the *rendered chat
template* (not a token-level sentinel), so offsets survive any tokenizer.

## 3. Evaluation protocol

Generation with `goal + optimized suffix` (greedy, `temperature=0`):

- **ASR (prefix match)** — the paper/HarmBench-style criterion: the response starts
  with the normalized target string (`src/evaluate.py::is_target_prefix_match`).
- **Refusal rate** — regex heuristic over refusal phrasings ("I cannot ...",
  "I'm sorry ...", "as an AI ...") reported alongside ASR so both views are visible.
- **Suffix perplexity** — `PPL(s | prefix)`; diagnostic *and* the defense baseline (§5).

Every run writes a machine-readable record to `results/reports/generated/`.

## 4. Pipeline validation (offline, tiny random model)

We validate every code path on a 2-layer randomly-initialized GPT-2-style model with
a locally-trained BPE tokenizer (`src/tiny_model.py`) — no downloads, seconds on CPU.
Random weights cannot be jailbroken; the point is **numerical correctness of the
pipeline**. Results (`results/reports/smoke_results.json`, seed 0, T=10, L=8, k=8, B=16):

| Check | Result | Verdict |
|---|---|---|
| KV-cache vs. full forward — gradients | max&#124;Δ&#124; = 1.2e-09 | ✅ exact |
| KV-cache vs. full forward — step loss | max&#124;Δ&#124; = 4.8e-07 | ✅ |
| KV-cache vs. full forward — candidate losses | max&#124;Δ&#124; = 9.5e-07 | ✅ |
| Full run trajectory (same seed) | max&#124;Δ&#124; = 4.8e-07 | ✅ identical |
| Algorithm 1 optimization runs | loss 6.261 → 6.248 over 10 steps | ✅ |
| Evaluation harness computes ASR / refusal | ASR=0%, refusal=0% (expected for random weights) | ✅ |
| Defense baseline runs (optimized / random / natural PPL) | computed; uncalibrated on random weights | ✅ |

Run it yourself: `python -m experiments.run_smoke`.

## 5. Defense baseline: perplexity filtering

GCG suffixes are typically low-probability token gibberish, so a simple
`PPL(s | prefix) > τ` detector catches many attacks. `src/defense.py` measures
suffix perplexity for the optimized suffix vs. a random-token control vs. a
natural-language control and scores a threshold detector (`experiments/run_defense.py`).
On real (trained) models the optimized suffix PPL is far above natural text; on the
tiny random model the three values are not separated — expected, since random weights
have no language statistics.

## 6. Full-scale run recipe (real aligned models)

```bash
# Paper settings, bundled AdvBench-style eval sample, CPU-friendly model
python -m experiments.run_algorithm1 --preset paper --benchmark

# Paper settings, real target model (needs GPU + HF access)
python -m experiments.run_algorithm1 --preset paper --benchmark \
    --model Qwen/Qwen3-1.7B --load-in-4bit

# Perplexity defense on the same setup
python -m experiments.run_defense --preset paper --benchmark --model Qwen/Qwen3-1.7B
```

Full AdvBench (`walledai/AdvBench`) is pulled automatically when `datasets` and
network access are available; otherwise the bundled `data/advbench_sample.jsonl` is used.

**Expected scale** (paper settings): `num_steps=250, suffix_length=20, topk=256,
search_width=512`. We currently report only pipeline-validation numbers — ASR on
aligned models must be measured on a GPU box and recorded in
`results/reports/generated/` before quoting anywhere.

## 7. Limitations & next steps

- **No benchmark ASR numbers yet** — the sandbox used for development has no GPU and
  no Hugging Face access; the harness is ready but the full-scale table is TODO.
- Perplexity detection is only a baseline; stronger defenses (perplexity + window,
  smooth perplexity, re-ranking) are future work.
- Cross-model **transfer** of optimized suffixes is implemented for Algorithm 2
  (string-level suffix re-tokenization) but not yet benchmarked.
- Obvious extensions: GCG variants (MMA, Faster-GCG), multi-objective losses
  (GCG + fluency regularizer), and an AutoDAN-style baseline for comparison.

## 8. References

- Zou et al., 2023 — *Universal and Transferable Adversarial Attacks on Aligned Language Models* ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043))
- Pulfer, 2025 — *GCG: Adversarial Attacks on Large Language Models* ([Medium](https://medium.com/@brianpulfer/gcg-adversarial-attacks-on-large-language-models-61f8b51734e9))
- Mazeika et al., 2024 — *HarmBench* ([arXiv:2402.04249](https://arxiv.org/abs/2402.04249))
- llm-attacks ([GitHub](https://github.com/llm-attacks/llm-attacks)) · nanoGCG ([GitHub](https://github.com/GraySwanAI/nanoGCG))

---

**Responsible use.** This repository is a research/education implementation for
understanding and hardening aligned models. Default configurations use benign goals;
benchmark strings in `data/` are standard red-team evaluation data. Do not use this
material to cause harm.
