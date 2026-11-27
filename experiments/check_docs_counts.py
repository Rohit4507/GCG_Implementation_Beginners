"""
Fail if the README's self-reported test / smoke-check counts drift.

`README.md` states "111 unit tests" and "7/7 smoke checks" in four places. Those
numbers were typed by hand, so every test added since then has made the README
slightly wrong — and a repo whose selling point is checkable evidence cannot
have uncheckable numbers on its front page.

This script derives both numbers from the code and compares them with the
README. Run it in CI (and before a release).

    python -m experiments.check_docs_counts
    python -m experiments.check_docs_counts --fix   # rewrite the README

Exit code is 1 on drift, so it works as a CI gate.
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

README = "README.md"
TESTS_GLOB = "tests/test_*.py"
SMOKE_FILE = "experiments/run_smoke.py"


def count_tests() -> int:
    """
    Number of `test_*` methods in the suite.

    Counted from the AST rather than by running unittest, so this stays fast and
    works without torch installed.
    """
    total = 0
    for path in sorted(glob.glob(TESTS_GLOB)):
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total += 1
    return total


def count_smoke_checks() -> int:
    """
    Number of checks `run_smoke.py` records.

    Read from the keys it assigns into `results["checks"]`, which is what the
    "N/N checks passed" line reports at the end of a run.
    """
    source = open(SMOKE_FILE, encoding="utf-8").read()
    return len(re.findall(r'results\["checks"\]\[[^\]]+\]\s*=', source))


def readme_claims() -> dict:
    text = open(README, encoding="utf-8").read()
    tests = sorted({int(m) for m in re.findall(r"(\d+)\s+unit tests", text)})
    smoke = sorted({int(m) for m in re.findall(r"(\d+)/(\d+)\s+smoke checks", text)
                    for m in (m[0],)} | {
        int(a) for a, b in re.findall(r"(\d+)/(\d+)\s+smoke checks", text)
    })
    return {"unit_tests": tests, "smoke_checks": smoke, "text": text}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true",
                        help="rewrite the stale numbers in README.md")
    args = parser.parse_args()

    actual_tests = count_tests()
    actual_smoke = count_smoke_checks()
    claims = readme_claims()

    problems = []
    for claimed in claims["unit_tests"]:
        if claimed != actual_tests:
            problems.append(f"README says {claimed} unit tests; there are {actual_tests}")
    for claimed in claims["smoke_checks"]:
        if claimed != actual_smoke:
            problems.append(f"README says {claimed} smoke checks; there are {actual_smoke}")

    if not problems:
        print(f"README counts OK: {actual_tests} unit tests, {actual_smoke} smoke checks")
        return 0

    for problem in problems:
        print(f"DRIFT: {problem}", file=sys.stderr)

    if args.fix:
        text = claims["text"]
        text = re.sub(r"\b\d+(\s+unit tests)", rf"{actual_tests}\1", text)
        text = re.sub(r"\b\d+/\d+(\s+smoke checks)", rf"{actual_smoke}/{actual_smoke}\1", text)
        with open(README, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"rewrote {README} with {actual_tests} / {actual_smoke}")
        return 0

    print("run with --fix to update the README", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
