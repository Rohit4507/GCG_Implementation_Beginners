# GCG Algorithm 2 — Universal (Multi-Prompt) Attack

**TL;DR** — Algorithm 2 of Zou et al. (2023): find **one** suffix that works across
**many** goals at once. Our implementation adds a progressive prompt schedule and
per-prompt gradient normalization, evaluates held-out generalization, and reuses the
same validated building blocks as Algorithm 1.

---

## 1. Objective

Given pairs `[(x_1, y*_1), ..., (x_n, y*_n)]`, find a single suffix `s` minimizing the
**mean** target loss across prompts:

```
L_univ(s) = (1/n) Σ_i L_i(s)      with L_i from Algorithm 1
```

A universal suffix that transfers across prompts (and models) is the paper's most
celebrated — and most alignment-relevant — result: it shows jailbreaks are not
one-off quirks but a structural robustness failure.

## 2. Method (mapped to the code)

`src/algorithm2.py::run_universal_gcg`:

1. **Progressive prompt schedule** — start with `num_prompts_start` prompts and add
   one every `prompt_add_interval` steps (`num_active = min(start + step // interval, n)`).
   Optimizing against all prompts from step 0 is unstable; ramping up is much more reliable.
2. **Aggregated gradients** — per active prompt, compute `∇_z L_i`, **normalize** it
   to unit L2 norm, then average. Normalization stops one prompt's scale from
   dominating the shared update direction.
3. **Candidate sampling** — identical vectorized single-token mutation sampler as
   Algorithm 1 (`sample_candidates`), driven by the aggregated gradient.
4. **Universal re-tokenization filter** — a candidate must round-trip through
   *every* active prompt's template (`filter_candidates` applied per prompt).
5. **Multi-prompt scoring** — each candidate's loss is the mean of per-prompt losses
   (`evaluate_candidates` per prompt, summed and averaged).
6. **Greedy update + best-suffix checkpointing** — same as Algorithm 1.

**KV-cache**: each prompt's fixed prefix cache is built once and reused across steps
(`prefix_caches` in `run_universal_gcg`).

## 3. Evaluation protocol

- **In-domain ASR** — prefix-match attack success rate on the optimization prompts.
- **Held-out generalization** — the suffix is also evaluated on prompts it never saw
  during optimization (`--held-out N`); this is the cheapest proxy for the paper's
  transferability claim without a second GPU model.
- **Cross-model transfer (optional)** — an optimized suffix *string* can be
  re-tokenized with another model's tokenizer and evaluated; see `src/evaluate.py`.

## 4. Pipeline validation (offline, tiny random model)

From `results/reports/smoke_results.json` (seed 0, T=10, L=8, k=8, B=16, 3 benign
prompts, progressive schedule `1 + step//3`):

| Check | Result | Verdict |
|---|---|---|
| Universal loop completes 10 steps | mean loss 6.261 → 6.248 | ✅ |
| Prompts activate on schedule | 1 → 2 → 3 active prompts observed | ✅ |
| Loss history recorded per step | tuples `(step, mean_loss)` | ✅ |
| In-domain + held-out ASR computed | ASR=0% (expected for random weights) | ✅ |

Demo run (`python -m experiments.run_algorithm2 --tiny --benchmark`) shows the full
artifacts: `results/plots/loss_universal.png` and
`results/reports/generated/algorithm2_*.json`.

## 5. Full-scale run recipe

```bash
# Paper settings on benign prompts (safe demo)
python -m experiments.run_algorithm2 --preset paper

# Paper settings on the benchmark sample with 2 held-out prompts
python -m experiments.run_algorithm2 --preset paper --benchmark --held-out 2

# Real model (GPU + HF access)
python -m experiments.run_algorithm2 --preset paper --benchmark \
    --model Qwen/Qwen3-1.7B --load-in-4bit
```

Paper-scale knobs (`--preset paper`): `num_steps=250, suffix_length=20, topk=256,
search_width=512`, progressive schedule `num_prompts_start=1, prompt_add_interval=50`.

## 6. Limitations & next steps

- Held-out ASR on real aligned models is the key number to produce next (GPU).
- Gradient normalization vs. simple mean: we follow the normalize-then-average
  design (a la llm-attacks' `GCG` multi-prompt mode); an ablation vs. raw averaging
  would make a nice result table.
- Cross-model transfer evaluation (different tokenizers) is wired but not benchmarked.
- Compare against random-search and no-attack baselines for the report table.

## 7. References

- Zou et al., 2023 — *Universal and Transferable Adversarial Attacks on Aligned Language Models* ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043))
- Pulfer, 2025 — *GCG: Adversarial Attacks on Large Language Models* ([Medium](https://medium.com/@brianpulfer/gcg-adversarial-attacks-on-large-language-models-61f8b51734e9))
- llm-attacks ([GitHub](https://github.com/llm-attacks/llm-attacks)) · nanoGCG ([GitHub](https://github.com/GraySwanAI/nanoGCG))

---

**Responsible use.** Research/education only — see the note in
[algorithm1_report.md](algorithm1_report.md).
