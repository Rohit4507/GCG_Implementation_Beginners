"""
Tests for attack-scope accounting: which prompts were actually attacked.

The first real-model run optimised one suffix on pair 0 of 10, evaluated all
10, and wrote a single ASR over 10. That record cannot be read back correctly:
the in-domain rate (n=1) and the transfer rate (n=9) are averaged into one
number whose n is wrong for both claims.

These tests pin the fix, and in particular pin the pooling behaviour, because
averaging per-call summaries is the specific mistake that produced the
unreadable record.

    python -m unittest discover -s tests -v
"""
import unittest

from src.evaluate import pool_evaluations, summarise_samples
from experiments.build_report import _kind, extract


def _samples(prefix_matches, refusals=None, start=0):
    refusals = refusals if refusals is not None else [False] * len(prefix_matches)
    return [
        {"pair_index": start + i, "goal": f"g{i}", "target": f"t{i}",
         "response": f"r{i}", "prefix_match": bool(p), "refusal": bool(r),
         "success": bool(p)}
        for i, (p, r) in enumerate(zip(prefix_matches, refusals))
    ]


def _evaluation(prefix_matches, refusal_count=0, start=0):
    samples = _samples(
        prefix_matches,
        refusals=[i < refusal_count for i in range(len(prefix_matches))],
        start=start,
    )
    return {"samples": samples, "summary": summarise_samples(samples)}


class TestSummariseSamples(unittest.TestCase):
    def test_rate_and_n(self):
        summary = summarise_samples(_samples([True, False, False, False]))
        self.assertEqual(summary["n_pairs"], 4)
        self.assertAlmostEqual(summary["asr_prefix_match"], 0.25)

    def test_refusal_rate_is_separate_from_asr(self):
        """
        The two must not be conflated: a model that neither refuses nor complies
        scores 0% ASR with 0% refusals, which is a real state the first run hit.
        """
        summary = summarise_samples(_samples([False, False], refusals=[True, False]))
        self.assertEqual(summary["asr_prefix_match"], 0.0)
        self.assertAlmostEqual(summary["refusal_rate"], 0.5)

    def test_empty_is_zero_not_a_crash(self):
        summary = summarise_samples([])
        self.assertEqual(summary["n_pairs"], 0)
        self.assertEqual(summary["asr_prefix_match"], 0.0)


class TestPoolEvaluations(unittest.TestCase):
    def test_pooling_recomputes_rather_than_averaging_summaries(self):
        """
        The regression that matters.

        Pooling a 1-sample call with 1 success and a 9-sample call with none
        gives 1/10 = 10%. Averaging the two summaries gives (1.0 + 0.0)/2 = 50%.
        The second number is what an unreadable record looks like.
        """
        pooled = pool_evaluations([_evaluation([True]), _evaluation([False] * 9)])

        self.assertEqual(pooled["summary"]["n_pairs"], 10)
        self.assertAlmostEqual(pooled["summary"]["asr_prefix_match"], 0.1)
        self.assertNotAlmostEqual(pooled["summary"]["asr_prefix_match"], 0.5)

    def test_pooling_keeps_every_sample(self):
        pooled = pool_evaluations([_evaluation([True, False]), _evaluation([False])])
        self.assertEqual(len(pooled["samples"]), 3)

    def test_none_for_nothing_to_pool(self):
        """None, so a report can say "no held-out prompts" instead of 0%."""
        self.assertIsNone(pool_evaluations([]))
        self.assertIsNone(pool_evaluations([None, None]))

    def test_none_entries_are_skipped_not_counted(self):
        pooled = pool_evaluations([None, _evaluation([True]), None])
        self.assertEqual(pooled["summary"]["n_pairs"], 1)
        self.assertEqual(pooled["summary"]["asr_prefix_match"], 1.0)

    def test_per_prompt_pool_is_the_real_asr(self):
        """
        Three prompts, each attacked, each scored on its own prompt: the pooled
        in-domain rate is over n=3, which is what "ASR over 3 prompts" means.
        """
        per_prompt = [_evaluation([m], start=i) for i, m in enumerate([True, True, False])]
        pooled = pool_evaluations(per_prompt)
        self.assertEqual(pooled["summary"]["n_pairs"], 3)
        self.assertAlmostEqual(pooled["summary"]["asr_prefix_match"], 2 / 3)


class TestLegacyKindRepair(unittest.TestCase):
    def test_legacy_record_is_read_as_single_suffix(self):
        """
        Old records say `algorithm1_per_prompt` but were produced by the
        single-suffix runner. They have no `n_suffixes_optimised`, and must not
        be relabelled as the per-prompt mode they never ran.
        """
        legacy = {"run_kind": "algorithm1_per_prompt", "evaluation": _evaluation([False])}
        self.assertEqual(_kind(legacy), "algorithm1_single_suffix")

    def test_new_per_prompt_record_stays_per_prompt(self):
        record = {"run_kind": "algorithm1_per_prompt",
                  "n_suffixes_optimised": 3,
                  "in_domain": _evaluation([False, False, False])}
        self.assertEqual(_kind(record), "algorithm1_per_prompt")

    def test_seed_variance_still_inferred_from_its_own_field(self):
        self.assertEqual(_kind({"mean_paired_diff_nats": 0.01}), "seed_variance")


class TestExtractAttackScope(unittest.TestCase):
    def test_single_suffix_reports_in_domain_n_not_total(self):
        """
        The headline n must be the number of prompts that were attacked, which
        is 1, not the 10 prompts that were merely evaluated.
        """
        record = {
            "run_kind": "algorithm1_single_suffix",
            "n_suffixes_optimised": 1,
            "model": {"model_label": "m"},
            "in_domain": _evaluation([False]),
            "transfer": _evaluation([False] * 9),
        }
        row = extract(record)
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["asr"], 0.0)
        self.assertEqual(row["headline_value"], "0% → 0%")

    def test_per_prompt_reports_pooled_in_domain_n(self):
        record = {
            "run_kind": "algorithm1_per_prompt",
            "n_suffixes_optimised": 3,
            "model": {"model_label": "m"},
            "in_domain": _evaluation([True, True, False]),
            "transfer": _evaluation([False] * 6),
        }
        row = extract(record)
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["asr"], 2 / 3)

    def test_transfer_only_record_shows_a_dash_not_a_zero(self):
        """"No held-out prompts" must not render as "0% on held-out prompts"."""
        record = {
            "run_kind": "algorithm1_single_suffix",
            "n_suffixes_optimised": 1,
            "model": {"model_label": "m"},
            "in_domain": _evaluation([False]),
            "transfer": None,
        }
        self.assertEqual(extract(record)["headline_value"], "0% → —")

    def test_legacy_record_without_the_split_still_extracts(self):
        record = {
            "run_kind": "algorithm1_per_prompt",
            "model": {"model_label": "m"},
            "evaluation": _evaluation([False] * 10),
        }
        row = extract(record)
        self.assertEqual(row["n"], 10)
        self.assertEqual(row["asr"], 0.0)


class TestExtractControl(unittest.TestCase):
    def test_control_headline_is_the_lift_not_the_compliance_rate(self):
        record = {
            "run_kind": "no_attack_control",
            "model": {"model_label": "m"},
            "evaluation": _evaluation([False] * 10),
            "lift_vs_no_attack": {"optimised": {"engaged_delta": 0.4}},
        }
        row = extract(record)
        self.assertEqual(row["n"], 10)
        self.assertEqual(row["headline_value"], "+40%")
        self.assertEqual(row["asr_label"], "ASR (no attack)")

    def test_no_optimised_condition_gives_a_dash(self):
        record = {
            "run_kind": "no_attack_control",
            "model": {"model_label": "m"},
            "evaluation": _evaluation([False] * 3),
        }
        self.assertEqual(extract(record)["headline_value"], "—")


if __name__ == "__main__":
    unittest.main()
