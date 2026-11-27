"""
Tests for the seed-variance confidence interval and its provenance.

Two defects, both of which made the repo's one quantitative effect-size claim
weaker than it looked:

1. `t_crit = {4: 2.776, 9: 2.262, 19: 2.093, 29: 2.045}.get(n - 1, 1.96)`
   silently used the *normal* critical value for every seed count not in the
   table. 7 seeds got 1.96 instead of 2.447 — the interval came out ~20% too
   narrow, in exactly the small-n regime where it matters.

2. The record wrote `"model": args.model or "tiny(...)"` as a bare string, so
   `build_report`'s `model_kind == "hf_pretrained"` count and the pooled ladder
   could not tell a seed-variance record apart from any other run.

    python -m unittest discover -s tests -v
"""
import unittest

from experiments.run_seed_variance import t_critical_95


class TestTCritical(unittest.TestCase):
    def test_tabulated_values_are_exact(self):
        for df, expected in (
            (1, 12.706), (4, 2.776), (9, 2.262), (19, 2.093), (29, 2.045),
            (30, 2.042), (100, 1.984),
        ):
            self.assertAlmostEqual(t_critical_95(df), expected, places=3, msg=df)

    def test_the_previously_wrong_case_is_fixed(self):
        """df=6 (7 seeds) used to fall through to 1.96."""
        t = t_critical_95(6)
        self.assertNotAlmostEqual(t, 1.96, places=3)
        self.assertAlmostEqual(t, 2.447, places=3)

    def test_every_small_df_is_above_the_normal_value(self):
        for df in range(1, 31):
            self.assertGreater(t_critical_95(df), 1.96, df)

    def test_is_monotonically_decreasing(self):
        values = [t_critical_95(df) for df in range(1, 121)]
        for smaller_df, larger_df in zip(values, values[1:]):
            self.assertLessEqual(larger_df, smaller_df + 1e-12)

    def test_large_df_converges_to_the_normal_value(self):
        self.assertEqual(t_critical_95(1000), 1.96)

    def test_interpolates_between_tabulated_points(self):
        # 31-39 fall between df=30 (2.042) and df=40 (2.021).
        t = t_critical_95(35)
        self.assertGreater(t, 2.021)
        self.assertLess(t, 2.042)

    def test_degenerate_df_returns_infinity_not_a_number(self):
        """One seed gives no standard error and therefore no finite interval."""
        self.assertEqual(t_critical_95(0), float("inf"))
        self.assertEqual(t_critical_95(-3), float("inf"))


class TestRecordProvenance(unittest.TestCase):
    def test_runner_reports_provenance_as_a_dict(self):
        """
        The record must carry the same `model` block every other runner writes,
        not a bare string, or downstream consumers cannot classify it.
        """
        import inspect
        from experiments import run_seed_variance

        source = inspect.getsource(run_seed_variance.main)
        self.assertIn("describe_model", source)
        self.assertIn('"run_kind": "seed_variance"', source)
        self.assertNotIn('"model": args.model or', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
