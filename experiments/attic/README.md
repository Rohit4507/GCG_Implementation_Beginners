# Attic

Superseded development scripts, kept for the record. Nothing here is imported
by the test suite or by any runner, and nothing here is covered by CI.

| File | Why it is here |
|---|---|
| `check_algorithm1.py` | The first end-to-end exercise of Algorithm 1, written against a real model before `experiments/run_algorithm1.py` existed. It has no KV-cache, no cost accounting and no run record, and it asserts `suffix_ids.shape == (8,)` for the debug preset, so it cannot substitute for the runner. **It was not thrown away because it is not redundant.** Its six checks were the only place the gradient path was exercised directly, and one of them — that `model(input_ids=ids)` and `model(inputs_embeds=embed(ids))` agree — is the identity GCG's one-hot surrogate depends on. That check is now a real test in `tests/test_algorithm1_core.py`; it had previously never run in CI, because a file nothing imports never runs. |

Read it to see the raw pieces (`gradient -> candidates -> score`) laid out
without the surrounding machinery.

It is deliberately **not** a package: there is no `__init__.py` here, so
`python -m unittest discover -s tests` never tries to import it.
