"""
Tests for multi-turn helpers (src/multiturn.py).

The token-counting is easy to get wrong in a way that quietly mislabels an
experiment (e.g. calling a 3-user-turn attack "turn 5" because the context has
4 messages including assistant replies). These tests pin the definitions.

    python -m unittest discover -s tests -v
"""
import unittest

from src.multiturn import (
    count_context_messages,
    count_prior_turns,
    summarise_comparison,
)

PRIOR = [
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
]


class TestTurnCounting(unittest.TestCase):
    def test_counts_user_turns_not_messages(self):
        self.assertEqual(count_prior_turns(PRIOR), 2)

    def test_counts_all_context_messages(self):
        self.assertEqual(count_context_messages(PRIOR), 4)

    def test_none_is_zero(self):
        self.assertEqual(count_prior_turns(None), 0)
        self.assertEqual(count_context_messages(None), 0)

    def test_attacked_turn_is_prior_user_turns_plus_one(self):
        """Two escalation turns means the attacked message is user turn 3."""
        self.assertEqual(count_prior_turns(PRIOR) + 1, 3)


class TestSummariseComparison(unittest.TestCase):
    @staticmethod
    def _side(asr, loss=6.2, refusal=0.0):
        return {
            "loss_last": loss,
            "evaluation": {
                "summary": {"asr_prefix_match": asr, "refusal_rate": refusal},
            },
        }

    def test_both_zero_is_reported_as_uninformative(self):
        out = summarise_comparison(self._side(0.0), self._side(0.0))
        self.assertIn("no attack succeeded", out["verdict"])
        self.assertNotIn("asr_delta_multi_minus_single", out)

    def test_multi_turn_higher(self):
        out = summarise_comparison(self._side(0.1), self._side(0.3))
        self.assertAlmostEqual(out["asr_delta_multi_minus_single"], 0.2)
        self.assertIn("increased", out["verdict"])

    def test_multi_turn_lower(self):
        out = summarise_comparison(self._side(0.3), self._side(0.1))
        self.assertAlmostEqual(out["asr_delta_multi_minus_single"], -0.2)
        self.assertIn("decreased", out["verdict"])

    def test_equal_nonzero_is_not_a_difference(self):
        out = summarise_comparison(self._side(0.2), self._side(0.2))
        self.assertIn("no difference", out["verdict"])

    def test_missing_summary_is_flagged_incomplete(self):
        out = summarise_comparison({}, {})
        self.assertIn("incomplete", out["verdict"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
