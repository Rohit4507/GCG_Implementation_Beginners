"""
Run an experiment module in a throwaway copy of the repo.

Why this exists
---------------
Several tests need to execute a real runner end to end. Runners write to
`results/reports/generated/*.json` and `results/plots/*.png` with **relative
paths**, and the dashboard counts every record in `results/reports/generated/`.
So a test that "just runs the runner" does not merely check behaviour — it
injects synthetic runs into the published dataset.

That is not hypothetical. Writing these tests took the record count from 15 to
47, moved the dashboard's pooled denominator from 78 to 102 and changed the
headline from 42% to 32%. The numbers were regenerated from real records, but
the inputs had been contaminated by test fixtures. For a repository whose whole
argument is "every number is re-derivable from the records", that is the worst
possible failure mode: the gate that checks docs-are-in-sync would happily
commit it, because the docs *did* match the records.

So: copy the code into a temporary directory, run there, discard. The tests
still cross the config -> optimiser -> record seam, which is the part that
matters, and the working tree is untouched.
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Code and data needed for a runner to work; deliberately NOT results/ or docs/
# so there is nothing to pollute and nothing to compare against.
_COPY = ("src", "experiments", "data", "requirements.txt")


def run_module(module, args, timeout=900):
    """
    Run `python -m experiments.<module> <args>` in a temp copy of the repo.

    Returns a `subprocess.CompletedProcess`. `stdout`/`stderr` are captured.
    """
    tmp = tempfile.mkdtemp(prefix="gcg-test-")
    try:
        for name in _COPY:
            src = os.path.join(REPO_ROOT, name)
            dst = os.path.join(tmp, name)
            if os.path.isdir(src):
                shutil.copytree(
                    src, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            elif os.path.isfile(src):
                shutil.copy2(src, dst)

        # A runner imports `experiments.<mod>`, which needs the repo on the path
        # but must not pick up the outer working tree.
        env = dict(os.environ)
        env["PYTHONPATH"] = tmp
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        return subprocess.run(
            [sys.executable, "-m", f"experiments.{module}", *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=tmp, env=env,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
