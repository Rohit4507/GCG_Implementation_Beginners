# Quarantined run records — do not cite these

These six JSON records were produced **before** `src/provenance.py` existed, by a
version of the runners that wrote the *requested* model id into the record
instead of the model that was actually loaded.

Every one of them says:

```json
"model_id": "HuggingFaceTB/SmolLM2-360M-Instruct"
```

**None of them ran SmolLM2.** Every one was a `--tiny` run against the offline
2-layer randomly-initialised model built by `src/tiny_model.py`. They are kept
because deleting inconvenient data is worse than labelling it, but they must not
be cited as SmolLM2 results.

## How you can tell they are not real-model runs

The tell is quantitative, not a matter of trust.

The tiny model's vocabulary is 512 tokens. If a suffix carried no information,
uniform guessing over that vocabulary floors the loss at

```
ln(512) = 6.238
```

and every recorded loss in these files sits at ~6.25 — right on that floor. A
real SmolLM2-360M-Instruct has a vocabulary of ~49,000 tokens, so its floor is
`ln(49152) ≈ 10.80`. Losses in the 6.2s are impossible for a real SmolLM2: they
would require the model to be roughly `e^(10.80-6.24) ≈ 90×` more confident about
the target than uniform guessing, using a suffix that was never actually
optimised against it.

That gap is why the records went out mislabelled in the first place: a run on a
random-weight model and a run on a real model look identical if all you record is
a string you typed into an argument.

## Files

| File | What it was |
|---|---|
| `algorithm1_20261010T075116Z.json` | per-prompt GCG on the tiny random model |
| `algorithm2_20261010T075117Z.json` | universal GCG on the tiny random model |
| `baselines_20261010T075735Z.json` | GCG vs random search on the tiny random model |
| `defense_20261010T075132Z.json` | perplexity defence baseline on the tiny random model |
| `transfer_20261010T075811Z.json` | transfer between two tiny random models |
| `pair_20261011T172352Z.json` | PAIR loop on the tiny random model |

All six were re-run on 2026-10-13 through the fixed runners; the replacements are
in the parent directory and carry a `model` block with
`model_kind: "tiny_random"`, `random_weights: true`, `vocab_size: 512`, and the
`ln(vocab)` loss floor.

## What changed

`src/provenance.py` now records what was loaded:

```json
"model": {
  "model_label": "tiny(offline random weights), seed=42",
  "model_kind": "tiny_random",
  "hf_model_id": null,
  "random_weights": true,
  "vocab_size": 512,
  "note": "…Loss is floored at ln(vocab) = ln(512) = 6.238."
}
```

and `tests/test_provenance.py` fails the build if a tiny run ever records an HF
model id again.

The general lesson, worth keeping in the writeup: **a run record should be
written by the code that loaded the model, not by the person who typed the
command.** Any field that is copied from a config rather than observed is a field
that can lie.
