# PAIR — Prompt Automatic Iterative Refinement (Black-Box Baseline)

**TL;DR** — This report documents our from-scratch implementation of **PAIR**
([Chao et al., 2023](https://arxiv.org/abs/2310.08419), *Jailbreaking Black Box Large
Language Models in Twenty Queries*): a black-box jailbreak attack that needs no
gradients, no logits, and no suffix — just a chat conversation. It's added as a
**comparison point for GCG**: same goal/target pairs, same evaluation harness, same
offline smoke tests, opposite threat model. We validate the pipeline end-to-end
offline and document the recipe for a real attacker/target run.

---

## 1. Objective & threat model

GCG (Algorithm 1 in this repo) assumes **white-box** access: it needs the target
model's gradients to optimize a discrete suffix token by token. PAIR assumes only
**black-box** query access to the target — it never looks inside the target model at
all. Instead, a separate **attacker** LLM plays a short game with itself:

1. Propose an adversarial jailbreak prompt `P` meant to make the target produce the
   objectionable behavior described by `goal`, ideally starting with `target`.
2. Send `P` to the target, observe its response `R`.
3. Score `R` with a **judge** on a 1-10 scale (10 = fully complies, 1 = refuses /
   irrelevant).
4. Feed `(R, score)` back to the attacker and ask it to refine `P`.
5. Repeat for a small number of iterations; stop early on a perfect score.

Several independent refinement **streams** are run (different random seeds /
sampling), and the best-scoring prompt across all streams wins. The paper's headline
claim is that this finds working jailbreaks in roughly "twenty queries" — far fewer
than GCG's hundreds of gradient steps — at the cost of needing a capable attacker LLM
and being inherently less reliable/reproducible (stochastic, model-dependent).

## 2. Method (mapped to the code)

| Paper concept | Code | Notes |
|---|---|---|
| Attacker system prompt + objective | `src/pair.py::ATTACKER_INSTRUCTIONS_TEMPLATE` | Folded into the **first user turn** rather than a `system` role — not every chat template supports `system`, and the bundled offline tiny tokenizer is deliberately minimal. Keeps `--attacker-model` portable across arbitrary HF chat models. |
| Attacker → target query | `src/pair.py::chat_generate` | Thin wrapper around `src/evaluate.py::generate_response` — the *same* decoding call GCG's evaluation harness uses. |
| JSON output `{"improvement", "prompt"}` | `parse_attacker_output` | Strict JSON first, then a regex over just the `"prompt"` field, then **fall back to the previous prompt unchanged**. A weak/untrained attacker (e.g. the offline tiny model) will not reliably emit valid JSON, and the loop must never crash because of that. |
| Judge (1-10 rating) | `heuristic_judge` / `model_judge` | Two interchangeable judges (`--judge keyword\|model`), see §3. |
| Iterative refinement loop | `run_pair_stream` | One conversation: attacker proposes → target responds → judge scores → feedback appended → repeat, with early stopping at `early_stop_score` (default 10). |
| Parallel streams + best-of | `run_pair` | Re-seeds and repeats `run_pair_stream` `n_streams` times, returns the best prompt found — the black-box analogue of `run_gcg`'s `(best_suffix_ids, loss_history)`. |

## 3. The judge: two options, same interface

- **`"keyword"` (default)** — reuses the exact heuristics already validated for GCG's
  evaluation harness: `is_target_prefix_match` (full paper-style success → score 10)
  and `contains_refusal` (clear refusal → score 1), otherwise 5 ("engaged but didn't
  fully comply"). No extra model calls, works offline, works on the tiny random model.
- **`"model"`** — asks an LLM (the attacker by default, or a separate `--judge-model`)
  to output a 1-10 rating directly (closer to the paper's GPT-4 judge), with a regex
  parse and a fallback to the keyword judge if the rating can't be parsed. This is the
  more faithful option but needs a reasonably capable judge model to be meaningful.

Both share the same downstream contract (`int` in `[1, 10]`), so the rest of the loop
doesn't care which one is active.

## 4. Why it reuses the GCG harness

PAIR optimizes *text*, not tokens, so none of GCG's suffix/gradient machinery
(`token_manager.py`, `kv_cache.py`, `algorithm1.py`/`algorithm2.py`) applies. What
*does* carry over directly:

- **`src/data.py`** — identical goal/target pairs (benign demo set + bundled AdvBench
  sample / full hub pull), so GCG and PAIR runs are evaluated on the same data.
- **`src/evaluate.py`** — `generate_response` (decoding), `is_target_prefix_match`
  (ASR), `contains_refusal` (refusal heuristic), `text_perplexity` (is this prompt
  gibberish or natural language?).
- **`src/model.py`** — model/tokenizer loading, including optional 4-bit.
- **`src/tiny_model.py`** — the offline random-weight model + locally-trained
  tokenizer used for smoke tests (with its training corpus extended to include PAIR's
  own templates, so they compress to a reasonable token count instead of near
  character-level under the tiny vocabulary).

## 5. White-box vs. black-box: the comparison this baseline enables

`experiments/run_baselines.py` already asks "does the gradient signal help, compared
to gradient-blind random search, under an identical token-mutation budget?" PAIR asks
a different, complementary question: **"how does white-box (GCG) compare to black-box
(PAIR) under a much smaller query budget?"**

One concrete, already-measurable difference: **suffix/prompt perplexity.**
`experiments/run_defense.py` shows GCG suffixes are high-perplexity gibberish — that's
precisely what the perplexity defense baseline (`src/defense.py`) detects. PAIR
prompts are ordinary (if devious) natural language, so `experiments/run_pair.py`
reports `text_perplexity` on the winning prompt right next to the plain goal's
perplexity for comparison: a well-functioning PAIR prompt should look nothing like a
GCG suffix under that metric, i.e. **PAIR is expected to evade a perplexity-based
defense that catches GCG.** This is exactly the kind of asymmetry the "Responsible
use" section of the main README is about understanding in order to build *better*
defenses (not relying on perplexity alone).

## 6. Pipeline validation (offline, tiny random model)

Same philosophy as the rest of the repo: a 2-layer randomly-initialized GPT-2-style
model with a locally-trained BPE tokenizer (`src/tiny_model.py`) exercises every code
path — chat-turn construction, attacker-output parsing (including its fallback path,
since an untrained model essentially never emits valid JSON), the keyword judge, and
early stopping — in seconds on CPU, no downloads. Random weights obviously cannot be
jailbroken; the point is **pipeline correctness**, not attack quality (see
`results/reports/smoke_results.json`, check `pair_blackbox_refinement`):

| Check | Result |
|---|---|
| Runs `n_streams x n_iterations` without crashing | ✅ |
| Every stream stops early or completes `n_iterations` | ✅ |
| Attacker JSON parsing degrades gracefully (fallback prompt reused) | ✅ |
| Judge always returns an int in `[1, 10]` | ✅ |
| `run_pair` returns a non-empty best prompt | ✅ |

## 7. Evaluation protocol (`experiments/run_pair.py`)

For the winning prompt found across all streams:

- **Success (ASR)** — `is_target_prefix_match` on a fresh greedy generation, same
  criterion as GCG, so ASR numbers are comparable across the two attacks.
- **Refusal heuristic** — same regex-based detector as GCG's harness.
- **Prompt perplexity** vs. the plain goal's perplexity — the natural-language
  analogue of GCG's suffix-perplexity defense diagnostic (§5).
- **Query budget used** — `sum(len(stream) for stream in streams)`, always
  `<= n_streams * n_iterations`, shown next to the configured budget so the paper's
  "twenty queries" framing is visible in the output.

Every run writes a machine-readable record to `results/reports/generated/` and a
per-stream score-vs-iteration plot to `results/plots/pair_scores.png`.

## 8. Full-scale recipe

PAIR is only as good as its attacker. On the tiny offline model (random weights, no
instruction-following at all) the attacker cannot reliably produce valid refinements —
that's expected and the smoke test says so explicitly. For a real run:

```bash
# Self-attack: one instruction-tuned model plays attacker, target, and (heuristic) judge
python -m experiments.run_pair --preset paper --benchmark --model Qwen/Qwen3-1.7B

# Closer to the paper: a separate (often stronger) attacker model, model-based judge
python -m experiments.run_pair --preset paper --benchmark \
    --model Qwen/Qwen3-0.6B --attacker-model Qwen/Qwen3-1.7B --judge model
```

`PAIRConfig` (`src/config.py`) exposes `n_streams`, `n_iterations`,
`max_new_tokens_attacker/target`, `temperature_attacker/target`, and `judge` — the
`paper` preset widens the budget in the spirit of the paper's setting while staying
runnable without a GPU cluster (see `src/config.py::PAIR_PAPER_DEFAULTS`).

## References

- Chao et al., 2023 — [PAIR, arXiv:2310.08419](https://arxiv.org/abs/2310.08419)
- Zou et al., 2023 — [GCG, arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- [Official PAIR implementation](https://github.com/patrickrchao/JailbreakingLLMs)
