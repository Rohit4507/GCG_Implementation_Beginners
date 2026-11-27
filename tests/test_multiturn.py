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


class TestBuildPriorTurns(unittest.TestCase):
    """
    `build_prior_turns` runs a scripted escalation conversation and returns the
    messages to prepend plus a transcript. It was uncovered, and it is the only
    thing that decides how many turns a "3-turn attack" actually has.
    """

    @classmethod
    def setUpClass(cls):
        from src.tiny_model import build_tiny_model_and_tokenizer
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(seed=0)

    def _build(self, n_prior_turns):
        from src.multiturn import build_prior_turns
        return build_prior_turns(
            self.model, self.tokenizer, "cpu",
            n_prior_turns=n_prior_turns, max_new_tokens=4, verbose=False,
        )

    def test_returns_alternating_user_and_assistant_messages(self):
        prior_turns, transcript = self._build(2)
        self.assertEqual(len(prior_turns), 4)
        self.assertEqual([m["role"] for m in prior_turns],
                         ["user", "assistant", "user", "assistant"])

    def test_transcript_matches_the_message_count(self):
        prior_turns, transcript = self._build(2)
        self.assertEqual(len(transcript), 2)
        for entry, message in zip(transcript, prior_turns[::2]):
            self.assertEqual(entry["user"], message["content"])

    def test_n_prior_turns_is_respected(self):
        for n in (0, 1, 2):
            prior_turns, transcript = self._build(n)
            self.assertEqual(len(transcript), n)
            self.assertEqual(len(prior_turns), 2 * n)

    def test_user_turn_count_matches_the_documented_attack_shape(self):
        """
        `count_prior_turns` counts user turns only, so a "3-turn attack" with
        2 prior turns reports 2 and the attacked message is turn 3.
        """
        from src.multiturn import count_context_messages, count_prior_turns
        prior_turns, _ = self._build(2)
        self.assertEqual(count_prior_turns(prior_turns), 2)
        self.assertEqual(count_context_messages(prior_turns), 4)

    def test_assistant_replies_are_non_empty_strings(self):
        prior_turns, _ = self._build(1)
        self.assertIsInstance(prior_turns[1]["content"], str)
        self.assertGreater(len(prior_turns[1]["content"]), 0)

    def test_counters_handle_none(self):
        from src.multiturn import count_context_messages, count_prior_turns
        self.assertEqual(count_prior_turns(None), 0)
        self.assertEqual(count_prior_turns([]), 0)
        self.assertEqual(count_context_messages(None), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
