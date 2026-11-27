"""
Tests for the benchmark loader in `src/data.py`.

`load_benchmark_pairs` decides *which goals an experiment attacks*, and it has
two paths that must agree on shape: the Hugging Face hub path and the bundled
CSV/JSONL fallback. It was previously uncovered, so a change in either branch
could have altered the sample silently. These tests pin:

  * the fallback works with no network (the CI default),
  * `n` truncates from the front, deterministically,
  * `return_source` names where the goals came from (records quote it),
  * a hub failure is swallowed into the fallback rather than crashing a run.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src import data


class TestLoadPairs(unittest.TestCase):
    """The plain file reader both paths funnel through."""

    def _write(self, lines):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8",
        )
        with handle as f:
            f.write("".join(lines))
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_reads_goal_and_target_columns(self):
        path = self._write([
            json.dumps({"goal": "g1", "target": "t1"}) + "\n",
            json.dumps({"goal": "g2", "target": "t2"}) + "\n",
        ])
        goals, targets = data.load_pairs(path)
        self.assertEqual(goals, ["g1", "g2"])
        self.assertEqual(targets, ["t1", "t2"])

    def test_blank_lines_are_skipped(self):
        path = self._write([
            json.dumps({"goal": "g1", "target": "t1"}) + "\n",
            "\n",
            "   \n",
            json.dumps({"goal": "g2", "target": "t2"}) + "\n",
        ])
        goals, _ = data.load_pairs(path)
        self.assertEqual(goals, ["g1", "g2"])

    def test_missing_key_is_an_error_not_a_silent_skip(self):
        path = self._write([json.dumps({"goal": "g1"}) + "\n"])
        with self.assertRaises(KeyError):
            data.load_pairs(path)

    def test_returns_parallel_lists(self):
        goals, targets = data.load_benign_pairs()
        self.assertEqual(len(goals), len(targets))
        self.assertGreater(len(goals), 0)


class TestLoadBenchmarkPairs(unittest.TestCase):
    """The hub/fallback loader used by every runner's --benchmark flag."""

    def test_fallback_works_offline(self):
        goals, targets = data.load_benchmark_pairs(prefer_hub=False)
        self.assertEqual(len(goals), len(targets))
        self.assertGreater(len(goals), 0)
        for goal, target in zip(goals, targets):
            self.assertIsInstance(goal, str)
            self.assertIsInstance(target, str)

    def test_n_truncates_from_the_front(self):
        full, _ = data.load_benchmark_pairs(prefer_hub=False)
        head, _ = data.load_benchmark_pairs(n=3, prefer_hub=False)
        self.assertEqual(head, full[:3])

    def test_none_returns_everything(self):
        with_none, _ = data.load_benchmark_pairs(n=None, prefer_hub=False)
        with_big, _ = data.load_benchmark_pairs(n=10_000, prefer_hub=False)
        self.assertEqual(with_none, with_big)

    def test_return_source_names_the_bundled_file(self):
        source = data.load_benchmark_pairs(prefer_hub=False, return_source=True)[2]
        self.assertIn("bundled sample", source)

    def test_hub_failure_falls_back_instead_of_raising(self):
        """
        On Kaggle the hub is reachable; in CI it is not. A missing `datasets`
        package or a network error must not take down a run that would have
        worked from the bundled sample.
        """
        with patch.dict("sys.modules", {"datasets": None}):
            goals, targets, source = data.load_benchmark_pairs(
                prefer_hub=True, return_source=True,
            )
        self.assertGreater(len(goals), 0)
        self.assertIn("bundled sample", source)

    def test_default_prefers_hub_but_degrades(self):
        """`prefer_hub=True` is the default; unpatched it must still return."""
        goals, targets = data.load_benchmark_pairs()
        self.assertEqual(len(goals), len(targets))
        self.assertGreater(len(goals), 0)


if __name__ == "__main__":
    unittest.main()
