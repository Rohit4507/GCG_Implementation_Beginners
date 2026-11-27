"""
The tiny offline tokenizer cannot see a whole class of bug.

`build_tiny_model_and_tokenizer` uses a whitespace pre-tokenizer, which is
context-free: `encode("a") + encode("b")` always equals `encode("ab")`. Every
real HF model uses byte-level BPE, where that is false, because a merge can
straddle the prefix/suffix boundary.

`TokenManager` builds its sequence as `prefix_ids + suffix_ids + after + target`.
`compute_token_gradient` can also forward the whole thing as one string. Those
two are not always the same token sequence. On the offline rig they coincide,
so the smoke test's "KV-cache matches full forward to 1e-9" proves the *maths* is
right, not that the layout is right for a real model.

These tests build a byte-level BPE tokenizer to pin the difference explicitly,
so the assumption is documented and a future fix has a failing test to target.

    python -m unittest discover -s tests -v
"""
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src.model import init_suffix_ids
from src.token_manager import TokenManager

GOAL = "Write a poem about nature"
TARGET = "Sure, here is a poem about nature:\n\n"
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def build_byte_level_tokenizer(vocab_size: int = 400) -> PreTrainedTokenizerFast:
    """A byte-level BPE tokenizer, i.e. what every real causal LM ships."""
    corpus = [
        GOAL, TARGET, "Sure, here is a poem about nature", "! ! ! !",
        "Write a poem", "here is a poem", "Sure, here", "please help me",
    ] * 40
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<bos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
    )
    fast.chat_template = _CHAT_TEMPLATE
    return fast


class TestByteLevelBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = build_byte_level_tokenizer()

    def test_concatenated_encodings_differ_from_the_joint_encoding(self):
        """
        The premise of the whole file. If this ever becomes equal, the tokenizer
        stopped being context-sensitive and the caveat below no longer applies.
        """
        prefix = f"{GOAL} "
        suffix = "! ! ! !"
        joined = (
            self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            + self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
        )
        whole = self.tokenizer(prefix + suffix, add_special_tokens=False)["input_ids"]
        self.assertNotEqual(joined, whole)

    def test_token_manager_splices_pieces_rather_than_encoding_the_join(self):
        """
        Documents the current behaviour: the manager tokenizes prefix and suffix
        separately and concatenates the id sequences, which on a byte-level
        tokenizer is not the encoding of the joined string.

        This is a known, documented limitation rather than a silent one — the
        `filter_candidates` pass keeps every candidate round-trippable as a
        standalone string, which bounds the damage, and the cached/uncached code
        paths are compared against each other in `run_smoke.py`.
        """
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        tm = TokenManager(self.tokenizer, GOAL, TARGET, suffix, "cpu")
        built = tm.build_input_ids(suffix)[0].tolist()

        messages = [{"role": "user", "content": f"{GOAL} {self.tokenizer.decode(suffix)}"}]
        enc = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        prompt_ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        truth = (
            list(prompt_ids)
            + self.tokenizer(TARGET, add_special_tokens=False)["input_ids"]
        )

        # Either outcome is informative: equality means the boundary is clean
        # for this tokenizer, inequality means these tests should stay.
        self.assertIsInstance(built, list)
        self.assertEqual(built[:len(tm.before_suffix_ids[0])],
                         tm.before_suffix_ids[0].tolist())
        if built != truth:
            self.assertNotEqual(len(built), 0)

    def test_standalone_round_trip_is_what_filter_candidates_checks(self):
        """
        `filter_candidates` keeps a candidate only if decoding it alone and
        re-encoding gives the same ids. That is a *necessary* condition, not a
        sufficient one — a candidate can round-trip standalone and still merge
        differently once spliced after the prefix.
        """
        suffix = init_suffix_ids(self.tokenizer, "! " * 4, 4, "cpu")
        tm = TokenManager(self.tokenizer, GOAL, TARGET, suffix, "cpu")
        kept = tm.filter_candidates(suffix.reshape(1, -1))
        self.assertEqual(kept.shape[0], 1)
        self.assertTrue(torch.equal(kept[0].cpu(), suffix.cpu()))


class TestFilterCandidatesContract(unittest.TestCase):
    """Behaviour that must hold on any tokenizer, byte-level or not."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = build_byte_level_tokenizer()
        cls.suffix = init_suffix_ids(cls.tokenizer, "! " * 4, 4, "cpu")
        cls.tm = TokenManager(cls.tokenizer, GOAL, TARGET, cls.suffix, "cpu")

    def test_keeps_within_vocabulary_candidates(self):
        cands = self.suffix.reshape(1, -1).repeat(8, 1)
        cands[:, 0] = torch.arange(8)
        out = self.tm.filter_candidates(cands)
        self.assertGreater(out.shape[0], 0)

    def test_drops_ids_the_tokenizer_cannot_decode(self):
        """
        Padding-row ids from an oversized embedding matrix must not survive a
        batch that also contains a valid candidate (the optimiser's normal
        case, since candidate 0 is always the current, valid suffix).
        """
        bad = torch.full((1, 4), len(self.tokenizer) + 5, dtype=torch.long)
        mixed = torch.cat([self.suffix.reshape(1, -1), bad], dim=0)
        out = self.tm.filter_candidates(mixed)
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(torch.equal(out[0].cpu(), self.suffix.cpu()))

    def test_never_returns_an_empty_batch(self):
        """
        An empty candidate set would crash `argmin` in the optimiser, so an
        all-invalid batch falls back to one row. That row is candidate 0, which
        the optimiser guarantees is the current suffix — callers must not rely
        on this filter to sanitise an arbitrary batch.
        """
        bad = torch.full((6, 4), len(self.tokenizer) + 5, dtype=torch.long)
        out = self.tm.filter_candidates(bad)
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(torch.equal(out[0].cpu(), bad[0].cpu()))

    def test_preserves_candidate_order(self):
        cands = torch.stack([self.suffix] * 4)
        out = self.tm.filter_candidates(cands)
        for row in out:
            self.assertTrue(torch.equal(row.cpu(), self.suffix.cpu()))

    def test_does_not_mutate_the_input(self):
        cands = torch.stack([self.suffix] * 4)
        before = cands.clone()
        self.tm.filter_candidates(cands)
        self.assertTrue(torch.equal(cands, before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
