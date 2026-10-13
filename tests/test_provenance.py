"""
Tests for run provenance and cost accounting (src/provenance.py).

These lock in the fix for a real defect: run records used to write
`cfg.model_id` verbatim, so a `--tiny` run on the offline random model recorded
itself as `HuggingFaceTB/SmolLM2-360M-Instruct`. Anything downstream read that
as a real-model result.

Run with either:
    python -m unittest discover -s tests -v
    python -m tests.test_provenance
"""
import math
import unittest

from src.provenance import (
    RunAccounting,
    count_successes,
    describe_model,
    loss_floor,
)


class FakeTokenizer:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class TestDescribeModel(unittest.TestCase):
    def test_tiny_run_is_labelled_as_the_tiny_model(self):
        info = describe_model(
            tiny=True, model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
            tokenizer=FakeTokenizer(512), seed=42,
        )
        self.assertEqual(info["model_kind"], "tiny_random")
        self.assertTrue(info["random_weights"])
        # The bug: the HF id used to leak in as if it had been loaded.
        self.assertIsNone(info["hf_model_id"])
        self.assertNotIn("SmolLM2", info["model_label"])
        self.assertIn("tiny", info["model_label"])
        self.assertEqual(info["vocab_size"], 512)

    def test_tiny_note_states_that_attack_results_are_meaningless(self):
        info = describe_model(tiny=True, model_id="X", vocab_size=512)
        self.assertIn("no refusal behaviour", info["note"])
        self.assertIn("6.238", info["note"])  # ln(512), the loss floor

    def test_real_run_keeps_the_hf_id(self):
        info = describe_model(
            tiny=False, model_id="Qwen/Qwen2.5-1.5B-Instruct",
            tokenizer=FakeTokenizer(151936),
        )
        self.assertEqual(info["model_kind"], "hf_pretrained")
        self.assertFalse(info["random_weights"])
        self.assertEqual(info["hf_model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(info["vocab_size"], 151936)

    def test_seed_is_recorded_only_where_it_applies(self):
        self.assertIsNone(
            describe_model(tiny=False, model_id="X", vocab_size=10)["seed"]
        )
        self.assertEqual(
            describe_model(tiny=True, model_id="X", vocab_size=10, seed=7)["seed"],
            7,
        )


class TestLossFloor(unittest.TestCase):
    def test_floor_is_log_vocab(self):
        self.assertAlmostEqual(loss_floor(512), math.log(512), places=6)

    def test_none_without_a_vocab_size(self):
        self.assertIsNone(loss_floor(None))
        self.assertIsNone(loss_floor(0))


class TestRunAccounting(unittest.TestCase):
    def test_forward_counts_add_up(self):
        acc = RunAccounting(steps_run=10, gradient_forwards=10, candidate_forwards=160)
        self.assertEqual(acc.total_forwards, 170)

    def test_seconds_per_step(self):
        acc = RunAccounting(steps_run=10, wall_clock_s=20.0)
        self.assertAlmostEqual(acc.seconds_per_step, 2.0)

    def test_seconds_per_step_is_none_without_steps(self):
        self.assertIsNone(RunAccounting(steps_run=0, wall_clock_s=5.0).seconds_per_step)

    def test_cost_per_success_is_undefined_not_zero(self):
        """The whole point: 0% ASR must not be reported as a free attack."""
        acc = RunAccounting(wall_clock_s=3600.0, gpu_hourly_usd=0.4, n_successes=0)
        self.assertIsNone(acc.cost_per_success_usd)
        d = acc.as_dict()
        self.assertIsNone(d["cost_per_success_usd"])
        self.assertFalse(d["cost_per_success_defined"])
        self.assertIn("undefined", d["undefined_reason"])

    def test_cost_per_success_divides(self):
        acc = RunAccounting(wall_clock_s=3600.0, gpu_hourly_usd=0.4, n_successes=2)
        self.assertAlmostEqual(acc.cost_per_success_usd, 0.2)

    def test_forwards_per_success(self):
        acc = RunAccounting(
            steps_run=1, gradient_forwards=1, candidate_forwards=99, n_successes=2,
        )
        self.assertAlmostEqual(acc.forwards_per_success, 50.0)

    def test_free_tier_is_flagged_not_silently_zero(self):
        acc = RunAccounting(wall_clock_s=60.0, gpu_hourly_usd=0.0, n_successes=1)
        self.assertEqual(acc.estimated_cost_usd, 0.0)
        self.assertIn("free-tier", acc.as_dict()["cost_note"])


class TestCountSuccesses(unittest.TestCase):
    def test_counts_sample_flags(self):
        evaluation = {
            "samples": [
                {"success": True}, {"success": False}, {"success": True},
            ],
            "summary": {"asr_prefix_match": 0.67},
        }
        self.assertEqual(count_successes(evaluation), 2)

    def test_empty_evaluation(self):
        self.assertEqual(count_successes({}), 0)
        self.assertEqual(count_successes({"samples": []}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
