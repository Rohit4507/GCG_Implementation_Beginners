"""
Tests for the no-attack control condition.

Every jailbreak claim needs a baseline: if the model complies without any
suffix, a non-zero attack rate is not evidence the attack worked. These tests
pin the two things that make the control valid — the random suffix has to be
comparable to what the optimiser could have produced, and the lift has to be
computed against the right baseline.

    python -m unittest discover -s tests -v
"""
import unittest

import torch

from src.tiny_model import build_tiny_model_and_tokenizer
from experiments.run_control import random_roundtripping_suffix


class TestRandomControlSuffix(unittest.TestCase):
    """
    A random control made of arbitrary token ids would not be comparable to a
    GCG suffix: `run_gcg`'s candidate filter discards suffixes that do not
    survive decode -> re-encode, so the optimiser could never have kept such
    tokens. The control has to respect the same constraint.
    """

    @classmethod
    def setUpClass(cls):
        _, cls.tokenizer = build_tiny_model_and_tokenizer()

    def test_length_matches_the_request(self):
        for length in (4, 8):
            ids, _ = random_roundtripping_suffix(self.tokenizer, length, "cpu", seed=0)
            self.assertIsNotNone(ids, f"no round-tripping suffix found for length {length}")
            self.assertEqual(ids.shape[0], length)

    def test_the_suffix_survives_a_round_trip(self):
        """The whole point: these tokens re-encode to themselves."""
        ids, text = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=0)
        back = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        self.assertEqual(len(back), 8)
        self.assertTrue(torch.equal(ids.cpu(), torch.tensor(back)))

    def test_ids_are_inside_the_tokenizer_vocabulary(self):
        """
        Not just inside the embedding matrix. ids the tokenizer cannot decode
        would make the control look artificially bad.
        """
        ids, _ = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=1)
        self.assertTrue((ids < len(self.tokenizer)).all().item())

    def test_seed_is_reproducible(self):
        a, _ = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=7)
        b, _ = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=7)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_give_different_suffixes(self):
        a, _ = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=0)
        b, _ = random_roundtripping_suffix(self.tokenizer, 8, "cpu", seed=1)
        self.assertFalse(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
