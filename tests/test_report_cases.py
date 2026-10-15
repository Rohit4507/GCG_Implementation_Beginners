"""
Tests for the ladder wording on the generated page.

`build_report._ladder_html` has to describe four different situations that all
produce the same "0% ASR" headline. Getting this wrong is a silent failure in
the worst direction: the page would confidently explain a measurement gap that
is not there, or hand-wave away a real one. These tests pin each case.

    python -m unittest discover -s tests -v
"""
import re
import unittest

from experiments.build_report import _ladder_html


def _ladder(strict=0, not_refused=0, engaged=0, n=10, refuse=0, reasons=None):
    ladder = {"n": n, "degenerate_reasons": reasons or {}, "refusals": refuse}
    for key, hits in (("strict_prefix", strict),
                      ("not_refused", not_refused),
                      ("engaged", engaged)):
        ladder[key] = {
            "successes": hits,
            "rate": (hits / n) if n else None,
            "ci95_low": 0.0,
            "ci95_high": 0.2,
        }
    return ladder


def _text(html_body):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_body)).strip()


class TestLadderWording(unittest.TestCase):

    def test_no_samples_says_so_instead_of_printing_zeros(self):
        out = _text(_ladder_html({"n": 0}))
        self.assertIn("nothing to re-score", out.lower())
        self.assertNotIn("0%", out)

    def test_success_case_leads_with_the_successes(self):
        out = _text(_ladder_html(_ladder(strict=3, not_refused=8, engaged=6, n=10)))
        self.assertIn("3 of 10", out)
        self.assertIn("verbatim", out)

    def test_measurement_gap_case_is_called_out(self):
        """
        Engagement without a verbatim match: the case the project actually hit.
        The page must name it as a scoring artefact, not as model robustness.
        """
        out = _text(_ladder_html(_ladder(strict=0, not_refused=10, engaged=4, n=10)))
        self.assertIn("measurement-gap", out.lower())
        self.assertIn("4 of 10", out)

    def test_vacuous_ceiling_is_not_oversold(self):
        """
        Every generation degenerate is NOT evidence of a scoring problem, and
        must not be described as one. A garbage generation was never in a
        position to refuse, so the ceiling proves nothing.
        """
        out = _text(_ladder_html(
            _ladder(strict=0, not_refused=10, engaged=0, n=10,
                    reasons={"repetition": 10})))
        self.assertIn("vacuous", out.lower())
        self.assertIn("not a refusal", out.lower())
        self.assertNotIn("measurement-gap", out.lower())

    def test_real_zero_case_quotes_the_upper_bound(self):
        """Refusals only: 0% is real, but n bounds it and the page must say so."""
        out = _text(_ladder_html(
            _ladder(strict=0, not_refused=0, engaged=0, n=10, refuse=10)))
        self.assertIn("real measurement", out.lower())
        self.assertNotIn("vacuous", out.lower())

    def test_every_case_reports_the_interval_rule(self):
        for kwargs in ({"strict": 3, "not_refused": 8, "engaged": 6},
                       {"strict": 0, "not_refused": 10, "engaged": 4},
                       {"strict": 0, "not_refused": 10, "engaged": 0,
                        "reasons": {"repetition": 10}},
                       {"strict": 0, "not_refused": 0, "engaged": 0, "refuse": 10}):
            out = _text(_ladder_html(_ladder(n=10, **kwargs)))
            self.assertIn("wilson", out.lower())
            self.assertIn("is not a 0% rate", out.lower())

    def test_the_four_cases_are_mutually_distinct(self):
        """No two situations may produce the same explanation."""
        texts = [
            _text(_ladder_html(_ladder(strict=3, not_refused=8, engaged=6))),
            _text(_ladder_html(_ladder(strict=0, not_refused=10, engaged=4))),
            _text(_ladder_html(_ladder(strict=0, not_refused=10, engaged=0,
                                       reasons={"repetition": 10}))),
            _text(_ladder_html(_ladder(strict=0, not_refused=0, engaged=0, refuse=10))),
        ]
        self.assertEqual(len(set(texts)), 4, "cases collapsed into one another")


class TestLadderRungsPresent(unittest.TestCase):
    def test_all_three_rungs_are_rendered(self):
        out = _text(_ladder_html(_ladder(strict=0, not_refused=10, engaged=4)))
        for rung in ("strict_prefix", "not_refused", "engaged"):
            self.assertIn(rung, out)

    def test_ceiling_is_described_as_a_ceiling(self):
        out = _text(_ladder_html(_ladder(strict=0, not_refused=10, engaged=4)))
        self.assertIn("ceiling", out.lower())

    def test_degenerate_reasons_are_surfaced(self):
        out = _text(_ladder_html(_ladder(
            strict=0, not_refused=10, engaged=0, reasons={"echo_of_prompt": 7})))
        self.assertIn("echo_of_prompt=7", out)


if __name__ == "__main__":
    unittest.main()
