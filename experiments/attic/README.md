# Attic

Superseded development scripts, kept for the record. Nothing here is imported
by the test suite or by any runner, and nothing here is covered by CI.

| File | Why it is here |
|---|---|
| `check_algorithm1.py` | The first end-to-end exercise of Algorithm 1, written before `experiments/run_algorithm1.py` existed. Hardcodes a model id, has no KV-cache, no cost accounting, no run record, and asserts `suffix_ids.shape == (8,)` for the debug preset. Kept as a readable "raw pieces" reference (`gradient -> candidates -> score`) — use the runner for anything real. |

It is deliberately **not** a package: `experiments/attic/__init__.py` is empty
only so that `python -m unittest discover -s tests` does not try to import it.
