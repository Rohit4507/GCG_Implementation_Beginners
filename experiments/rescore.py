"""
Re-score finished run records under the multi-criterion judge.

Runs offline. No model, no tokenizer, no GPU — it only needs the raw responses
that `evaluate_pairs` already stores in every record. That matters because the
real-model records live on Kaggle: this is how a failed-looking 0% gets turned
into an explanation without spending another GPU-hour.

    python -m experiments.rescore                       # all generated records
    python -m experiments.rescore results/reports/generated/transfer_*.json
    python -m experiments.rescore --json                # machine-readable only

Writes `results/reports/scorecards/<record>.json` for anything it scores, and
prints a ladder table to stdout.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import iter_samples, score_response, summarise  # noqa: E402

DEFAULT_GLOB = "results/reports/generated/*.json"
OUT_DIR = "results/reports/scorecards"


def _pct(x, width: int = 6) -> str:
    return "  —   " if x is None else f"{x * 100:>{width - 1}.0f}%"


def _ci(entry) -> str:
    if not entry or entry["rate"] is None:
        return "—"
    return f"[{entry['ci95_low'] * 100:.0f}–{entry['ci95_high'] * 100:.0f}%]"


def score_record(path: str) -> Dict:
    with open(path) as fh:
        record = json.load(fh)

    samples = []
    for s in iter_samples(record):
        card = score_response(
            response=s.get("response", ""),
            goal=s.get("goal", ""),
            target=s.get("target", ""),
            prompt=s.get("goal", ""),  # the goal is the user turn; used for echo detection
        )
        card["_path"] = s.get("_path", "")
        samples.append(card)

    return {
        "record": os.path.basename(path),
        "summary": summarise(samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None,
                        help=f"record files or globs (default: {DEFAULT_GLOB})")
    parser.add_argument("--json", action="store_true",
                        help="print the scorecards as JSON and nothing else")
    parser.add_argument("--no-write", action="store_true",
                        help="do not write results/reports/scorecards/")
    args = parser.parse_args()

    patterns = args.paths or [DEFAULT_GLOB]
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or ([pattern] if os.path.exists(pattern) else []))

    if not paths:
        print(f"no records matched: {patterns}", file=sys.stderr)
        return 1

    scorecards = []
    for path in paths:
        try:
            scorecards.append(score_record(path))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skipped {path}: {exc}", file=sys.stderr)

    if not scorecards:
        print("nothing scoreable", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(scorecards, indent=2))
    else:
        print("=" * 92)
        print("SCORECARD LADDER — each rung requires the ones above it")
        print("=" * 92)
        print(f"{'record':<40} {'n':>4} {'strict_prefix':>14} "
              f"{'not_refused':>14} {'engaged':>14}")
        print("-" * 92)

        for card in scorecards:
            s = card["summary"]
            print(f"{card['record']:<40} {s['n']:>4} "
                  f"{_pct(s['strict_prefix']['rate']):>8} {_ci(s['strict_prefix']):>5} "
                  f"{_pct(s['not_refused']['rate']):>8} {_ci(s['not_refused']):>5} "
                  f"{_pct(s['engaged']['rate']):>8} {_ci(s['engaged']):>5}")

        print()
        for card in scorecards:
            s = card["summary"]
            if not s["n"]:
                continue
            reasons = ", ".join(f"{k}={v}" for k, v in s["degenerate_reasons"].items()) or "none"
            print(f"{card['record']}")
            print(f"    refusals {s['refusals']}/{s['n']} | degenerate: {reasons} | "
                  f"mean coverage {s.get('mean_goal_coverage')} | "
                  f"mean novelty {s.get('mean_novel_content')}")

    if not args.no_write:
        os.makedirs(OUT_DIR, exist_ok=True)
        for card in scorecards:
            name = os.path.splitext(card["record"])[0] + ".scorecard.json"
            with open(os.path.join(OUT_DIR, name), "w") as fh:
                json.dump(card, fh, indent=2)
        if not args.json:
            print(f"\nwrote {len(scorecards)} scorecard(s) to {OUT_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
