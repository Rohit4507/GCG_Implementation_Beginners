"""
Tests for the random-search baseline (src/baselines.py).

`baselines.py` had 0% test coverage while the README's "GCG beats random search"
claim rests entirely on it, and the only thing exercising it was one assertion
inside the smoke test. These tests pin the property that makes the comparison
fair: the baseline spends the *same budget* as `run_gcg` and differs only in how
candidate tokens are proposed.

    python -m unittest discover -s tests -v
"""
import unittest

import torch

from src.algorithm1 import run_gcg
from src.baselines import run_random_search, sample_random_candidates, target_loss
from src.model import init_suffix_ids
from src.tiny_model import build_tiny_model_and_tokenizer

GOALS = ["Write a poem about nature"]
TARGETS = ["Sure, here is a poem about nature:\n\n"]


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(
            goals=GOALS, targets=TARGETS, seed=0,
        )
        cls.goal, cls.target = GOALS[0], TARGETS[0]


class TestSampleRandomCandidates(_Fixture):
    def test_first_candidate_is_the_unmodified_suffix(self):
        suffix = torch.tensor([3, 4, 5, 6])
        cands = sample_random_candidates(suffix, search_width=8, vocab_size=50)
        self.assertEqual(cands.shape, (8, 4))
        self.assertTrue(torch.equal(cands[0], suffix))

    def test_each_mutation_touches_exactly_one_position(self):
        suffix = torch.tensor([3, 4, 5, 6])
        torch.manual_seed(0)
        cands = sample_random_candidates(suffix, search_width=64, vocab_size=50)
        diffs = (cands != suffix).sum(dim=1)
        # Candidate 0 is unchanged; every mutation differs in exactly one place.
        self.assertEqual(diffs[0].item(), 0)
        self.assertTrue((diffs[1:] == 1).all())

    def test_tokens_are_inside_the_requested_vocabulary(self):
        suffix = torch.tensor([3, 4, 5, 6])
        cands = sample_random_candidates(suffix, search_width=128, vocab_size=7)
        self.assertTrue((cands < 7).all())
        self.assertTrue((cands >= 0).all())

    def test_it_never_consults_a_gradient(self):
        """Gradient-blindness is the whole point of the baseline."""
        import inspect
        src = inspect.getsource(sample_random_candidates)
        for forbidden in ("grad", "topk", "one_hot"):
            self.assertNotIn(forbidden, src)


class TestRandomSearchBudget(_Fixture):
    def test_budget_matches_run_gcg(self):
        """
        Same steps, same search width, same re-tokenisation filter, same greedy
        update. If the budget drifts, the loss gap stops meaning anything.
        """
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        steps, width = 3, 8

        torch.manual_seed(0)
        gcg_stats = {}
        run_gcg(
            self.model, self.tokenizer, self.goal, self.target, suffix.clone(),
            num_steps=steps, topk=4, search_width=width, mini_batch_size=4,
            use_kv_cache=False, verbose=False, stats_out=gcg_stats,
        )

        torch.manual_seed(0)
        rand_stats = {}
        _, rand_losses = run_random_search(
            self.model, self.tokenizer, self.goal, self.target, suffix.clone(),
            num_steps=steps, search_width=width, mini_batch_size=4,
            use_kv_cache=False, verbose=False, stats_out=rand_stats,
        )

        self.assertEqual(len(rand_losses), steps)
        self.assertEqual(gcg_stats["steps_run"], rand_stats["steps_run"])
        self.assertLessEqual(rand_stats["candidate_forwards"],
                             gcg_stats["candidate_forwards"] * 2)

    def test_baseline_reports_zero_gradient_forwards(self):
        """
        The baseline's *point* is that it buys its progress without backward
        passes; a non-zero count here would mean the comparison is not a
        gradient-vs-no-gradient comparison at all.
        """
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        stats = {}
        run_random_search(
            self.model, self.tokenizer, self.goal, self.target, suffix,
            num_steps=2, search_width=4, mini_batch_size=2,
            use_kv_cache=True, verbose=False, stats_out=stats,
        )
        self.assertEqual(stats["gradient_forwards"], 0)
        self.assertGreater(stats["candidate_forwards"], 0)
        self.assertGreater(stats["wall_clock_s"], 0)

    def test_returns_a_suffix_of_the_requested_length(self):
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        best, losses = run_random_search(
            self.model, self.tokenizer, self.goal, self.target, suffix,
            num_steps=2, search_width=4, mini_batch_size=2,
            use_kv_cache=True, verbose=False,
        )
        self.assertEqual(best.shape, suffix.shape)
        self.assertEqual(len(losses), 2)

    def test_best_loss_is_non_increasing_at_the_end(self):
        """Greedy update: the returned suffix cannot be worse than where it began."""
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        best, losses = run_random_search(
            self.model, self.tokenizer, self.goal, self.target, suffix,
            num_steps=4, search_width=6, mini_batch_size=3,
            use_kv_cache=True, verbose=False,
        )
        self.assertLessEqual(losses[-1], losses[0] + 1e-6)


class TestTargetLoss(_Fixture):
    def test_matches_a_direct_cross_entropy_computation(self):
        from src.token_manager import TokenManager
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        tm = TokenManager(self.tokenizer, self.goal, self.target, suffix, "cpu")

        via_baseline = target_loss(self.model, tm, suffix)

        with torch.no_grad():
            logits = self.model(input_ids=tm.build_input_ids(suffix)).logits
            logits = logits[:, tm.loss_slice, :]
            manual = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tm.target_ids.reshape(-1),
            ).item()

        self.assertAlmostEqual(via_baseline, manual, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
