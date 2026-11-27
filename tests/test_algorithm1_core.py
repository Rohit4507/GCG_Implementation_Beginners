"""
Tests for the algorithmic core of Algorithm 1 (`src/algorithm1.py`).

These pin the identity that the whole gradient path rests on:

    ``model(inputs_embeds=embed(ids))`` and ``model(input_ids=ids)`` must
    produce the same logits.

GCG works by replacing the one-hot suffix ids with a *differentiable*
``one_hot @ embedding_matrix`` product and differentiating the loss w.r.t.
that matrix. That is only a valid surrogate for "which token should replace
this position" if the embedding path and the id path are numerically the same
forward pass. If they ever diverge (a transformers refactor changing how
position ids or attention masks are derived from embeds, for instance), the
gradient would still be computed, would still look plausible, and would point
at the wrong tokens.

This check used to live only in the pre-runner scratch script
(`experiments/attic/check_algorithm1.py`, formerly `experiments/check_algorithm1.py`).
It was never part of the test suite, so it did not run in CI. It does now.
"""
import unittest

import torch

from src.algorithm1 import compute_token_gradient, evaluate_candidates, sample_candidates
from src.token_manager import TokenManager
from src.tiny_model import build_tiny_model_and_tokenizer


def _default_suffix(tokenizer):
    """
    The debug preset's suffix, as ids.

    Module-level on purpose: a class body cannot reference another attribute of
    the same class while that class is still being defined, which is the trap
    that broke the first draft of this file.
    """
    return tokenizer(
        "! ! ! ! ! ! ! !", return_tensors="pt", add_special_tokens=False,
    ).input_ids.squeeze(0).long()


def _token_manager(tokenizer):
    return TokenManager(
        tokenizer=tokenizer,
        goal="Write a poem about the sea",
        target="Sure, here is a poem about the sea",
        suffix_ids=_default_suffix(tokenizer),
        device="cpu",
    )


class TestInputIdsEqualsInputsEmbeds(unittest.TestCase):
    """`inputs_embeds=embed(ids)` must equal `input_ids=ids`."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(seed=0)
        cls.tm = _token_manager(cls.tokenizer)

    def test_logits_match_on_the_full_prompt(self):
        """
        The precondition for the one-hot surrogate. The tolerance is loose
        because the two paths are different kernels; what matters is that they
        agree to float32 noise, not that they are bit-identical.
        """
        prompt_ids = self.tm.build_prompt_ids(_default_suffix(self.tokenizer))
        with torch.no_grad():
            logits_ids = self.model(input_ids=prompt_ids).logits
            embeds = self.model.get_input_embeddings()(prompt_ids)
            logits_embeds = self.model(inputs_embeds=embeds).logits

        self.assertEqual(logits_ids.shape, logits_embeds.shape)
        max_diff = torch.max(torch.abs(logits_ids - logits_embeds)).item()
        self.assertLess(
            max_diff, 1e-4,
            "input_ids and inputs_embeds paths diverged; the one-hot gradient "
            "surrogate in compute_token_gradient is no longer valid",
        )

    def test_discrimination_floor(self):
        """
        Guards the guard. An assertion of the form ``assert diff < 1e-4`` is
        vacuous if the model is insensitive to its inputs, so this test proves
        the quantity above is meaningful: perturbing the embeddings *off* the
        id path must move the logits by far more than the tolerance.

        Two notes, because both are easy to get wrong and I got the first one
        wrong while writing this file:

        * A **constant** offset added to every embedding does NOT count. The
          tiny model's first LayerNorm normalises it away, so ``emb + 0.5``
          moves the logits by only ~1e-06 and a mutation built that way looks
          like evidence that the main test is vacuous when it is not.
        * Per-position noise and swapping a single token's embedding are real
          corruptions, and they move the logits by ~0.7 and ~1.1 respectively
          (measured on the seed-0 tiny model).
        """
        prompt_ids = self.tm.build_prompt_ids(_default_suffix(self.tokenizer))
        emb_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            reference = self.model(input_ids=prompt_ids).logits
            embeds = emb_layer(prompt_ids)

            torch.manual_seed(0)
            noisy = self.model(
                inputs_embeds=embeds + 0.05 * torch.randn_like(embeds)
            ).logits

            swapped = embeds.clone()
            swapped[0, 3] = emb_layer.weight[100]
            swapped_out = self.model(inputs_embeds=swapped).logits

        noise_diff = torch.max(torch.abs(reference - noisy)).item()
        swap_diff = torch.max(torch.abs(reference - swapped_out)).item()
        self.assertGreater(noise_diff, 1e-3, "model ignores per-position noise")
        self.assertGreater(swap_diff, 1e-3, "model ignores a swapped token")

    def test_loss_matches_too(self):
        """The same check expressed on the loss, which is what the optimiser sees."""
        prompt_ids = self.tm.build_prompt_ids(_default_suffix(self.tokenizer))
        with torch.no_grad():
            ids_loss = self.model(input_ids=prompt_ids, labels=prompt_ids).loss
            embeds = self.model.get_input_embeddings()(prompt_ids)
            embeds_loss = self.model(inputs_embeds=embeds, labels=prompt_ids).loss
        self.assertLess(abs(ids_loss.item() - embeds_loss.item()), 1e-4)


class TestGradientShapeAndFiniteness(unittest.TestCase):
    """Step 1 of Algorithm 1: the gradient must be usable, not just present."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(seed=0)
        cls.tm = _token_manager(cls.tokenizer)
        cls.suffix = _default_suffix(cls.tokenizer)

    def test_gradient_is_suffix_len_by_vocab(self):
        grad, _ = compute_token_gradient(self.model, self.tm, self.suffix)
        self.assertEqual(
            tuple(grad.shape),
            (len(self.suffix), self.tokenizer.vocab_size),
        )

    def test_gradient_has_no_nan_or_inf(self):
        """
        A NaN here silently poisons `topk`, which yields arbitrary candidate
        ids, which produce a flat loss curve that looks like a weak attack
        rather than a broken one.
        """
        grad, _ = compute_token_gradient(self.model, self.tm, self.suffix)
        self.assertFalse(torch.isnan(grad).any().item())
        self.assertFalse(torch.isinf(grad).any().item())

    def test_every_optimised_position_gets_a_nonzero_gradient(self):
        """
        Only suffix positions are differentiable, and each of them must receive
        gradient -- a uniformly dead row would mean that position can never
        improve, so `topk` there is just noise.
        """
        grad, _ = compute_token_gradient(self.model, self.tm, self.suffix)
        row_sums = grad.abs().sum(dim=1)
        self.assertEqual(row_sums.shape[0], len(self.suffix))
        self.assertTrue(
            all(s > 0 for s in row_sums.tolist()),
            f"dead gradient rows: {[i for i, s in enumerate(row_sums.tolist()) if s == 0]}",
        )

    def test_initial_loss_is_a_finite_scalar(self):
        _, loss = compute_token_gradient(self.model, self.tm, self.suffix)
        self.assertIsInstance(loss, float)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())


class TestCandidatePipeline(unittest.TestCase):
    """Steps 2-3: sample from top-k, then evaluate the surviving batch."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(seed=0)
        cls.tm = _token_manager(cls.tokenizer)
        cls.suffix = _default_suffix(cls.tokenizer)

    def test_candidate_shape_and_validity(self):
        grad, _ = compute_token_gradient(self.model, self.tm, self.suffix)
        candidates = sample_candidates(self.suffix, grad, topk=10, search_width=16)
        self.assertEqual(candidates.shape, (16, len(self.suffix)))
        self.assertTrue((candidates >= 0).all())
        self.assertTrue((candidates < self.tokenizer.vocab_size).all())

    def test_evaluation_returns_one_finite_loss_per_candidate(self):
        grad, _ = compute_token_gradient(self.model, self.tm, self.suffix)
        candidates = sample_candidates(self.suffix, grad, topk=10, search_width=16)
        losses = evaluate_candidates(self.model, self.tm, candidates, mini_batch_size=8)
        self.assertEqual(losses.shape[0], candidates.shape[0])
        self.assertFalse(torch.isnan(losses).any())
        self.assertFalse(torch.isinf(losses).any())


if __name__ == "__main__":
    unittest.main()
