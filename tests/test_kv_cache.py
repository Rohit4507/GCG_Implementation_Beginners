"""
Tests for the KV-cache format layer (src/kv_cache.py).

`kv_cache.py` existed to paper over a real cross-version break: transformers
v4 exposes `DynamicCache.key_cache` / `.to_legacy_cache()`, and v5 rejects
legacy tuples on input entirely. That adapter chain was the least covered part
of the module (49%) and is exactly what a dependency bump breaks — while the
smoke test, which only exercises one code path, would still pass.

These tests pin each format branch without needing several transformers
versions installed.

    python -m unittest discover -s tests -v
"""
import unittest
from unittest import mock

import torch
from transformers import cache_utils

from src import kv_cache
from src.kv_cache import (
    PrefixCache,
    batchify_pairs,
    build_cache_from_pairs,
    cache_to_pairs,
)
from src.model import init_suffix_ids
from src.token_manager import TokenManager
from src.tiny_model import build_tiny_model_and_tokenizer

GOAL = "Write a poem about nature"
TARGET = "Sure, here is a poem about nature:\n\n"


class _FakeLayer:
    def __init__(self, k, v):
        self.keys = k
        self.values = v


class _FakeV5Cache:
    """Shape of a transformers v5 DynamicCache: .layers[i].keys/.values."""

    def __init__(self, pairs):
        self.layers = [_FakeLayer(k, v) for k, v in pairs]


class _FakeV4Cache:
    """Shape of a transformers v4 DynamicCache: .key_cache/.value_cache."""

    def __init__(self, pairs):
        self.key_cache = [k for k, _ in pairs]
        self.value_cache = [v for _, v in pairs]

    def to_legacy_cache(self):
        return tuple((k, v) for k, v in zip(self.key_cache, self.value_cache))


def _pairs(n_layers=2, shape=(1, 2, 3, 4)):
    return [(torch.randn(*shape), torch.randn(*shape)) for _ in range(n_layers)]


class TestCacheToPairs(unittest.TestCase):
    def test_legacy_tuple_of_tuples(self):
        pairs = _pairs()
        self.assertEqual(len(cache_to_pairs(tuple(pairs))), 2)

    def test_legacy_list_of_lists(self):
        pairs = _pairs()
        self.assertEqual(len(cache_to_pairs([list(p) for p in pairs])), 2)

    def test_v5_layers_format(self):
        pairs = _pairs()
        out = cache_to_pairs(_FakeV5Cache(pairs))
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0][0], pairs[0][0]))

    def test_v4_key_value_cache_format(self):
        pairs = _pairs()
        out = cache_to_pairs(_FakeV4Cache(pairs))
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[1][1], pairs[1][1]))

    def test_unknown_object_raises_instead_of_returning_garbage(self):
        with self.assertRaises(TypeError):
            cache_to_pairs(object())

    def test_bad_layer_type_raises_with_the_type_in_the_message(self):
        with self.assertRaises(TypeError) as ctx:
            cache_to_pairs([object()])
        self.assertIn("object", str(ctx.exception))


class TestBatchifyPairs(unittest.TestCase):
    def test_batch_size_one_is_a_passthrough(self):
        pairs = _pairs()
        self.assertIs(batchify_pairs(pairs, 1), pairs)

    def test_repeat_expands_the_leading_dimension_only(self):
        pairs = _pairs(n_layers=2, shape=(1, 2, 3, 4))
        out = batchify_pairs(pairs, 5)
        for (k, v) in out:
            self.assertEqual(k.shape, (5, 2, 3, 4))
            self.assertEqual(v.shape, (5, 2, 3, 4))

    def test_repeat_does_not_alias_the_original(self):
        pairs = _pairs()
        out = batchify_pairs(pairs, 3)
        out[0][0].add_(1.0)
        self.assertFalse(torch.equal(out[0][0][0], pairs[0][0][0]))

    def test_rows_are_identical_copies(self):
        pairs = _pairs()
        out = batchify_pairs(pairs, 4)
        k = out[0][0]
        for i in range(1, 4):
            self.assertTrue(torch.equal(k[0], k[i]))


class TestBuildCacheFromPairs(unittest.TestCase):
    def test_returns_something_the_model_accepts(self):
        pairs = _pairs()
        cache = build_cache_from_pairs(pairs)
        self.assertIsNotNone(cache)

    def test_round_trips_through_cache_to_pairs(self):
        pairs = _pairs()
        out = cache_to_pairs(build_cache_from_pairs(pairs))
        self.assertEqual(len(out), len(pairs))
        self.assertTrue(torch.equal(out[0][0], pairs[0][0]))


class TestBuildCacheConstructorFallbacks(unittest.TestCase):
    """
    `build_cache_from_pairs` probes up to four `DynamicCache` constructors
    because the signature changed across transformers v4.x and v5.x. Only the
    branch that happens to match the installed version is normally exercised,
    so the others were dead in practice. These tests drive each one with a fake
    class, because a wrong constructor here surfaces as a confusing TypeError
    deep inside a model forward pass rather than at the call site.
    """

    def _pairs(self):
        gen = torch.Generator().manual_seed(0)
        return [
            (torch.randn(1, 2, 3, 4, generator=gen),
             torch.randn(1, 2, 3, 4, generator=gen))
            for _ in range(2)
        ]

    def test_prefers_ddp_cache_data_kwarg(self):
        calls = {}

        class _Cache:
            def __init__(self, ddp_cache_data=None):
                calls["kwarg"] = ddp_cache_data

        with mock.patch.dict("sys.modules", {}):
            with mock.patch.object(cache_utils, "DynamicCache", _Cache):
                out = kv_cache.build_cache_from_pairs(self._pairs())
        self.assertIsInstance(out, _Cache)
        self.assertEqual(len(calls["kwarg"]), 2)

    def test_falls_back_to_legacy_cache_kwarg(self):
        calls = {}

        class _Cache:
            def __init__(self, legacy_cache=None):
                calls["legacy"] = legacy_cache

        with mock.patch.object(cache_utils, "DynamicCache", _Cache):
            out = kv_cache.build_cache_from_pairs(self._pairs())
        self.assertIsInstance(out, _Cache)
        self.assertEqual(len(calls["legacy"]), 2)

    def test_falls_back_to_from_legacy_cache_classmethod(self):
        seen = {}

        class _Cache:
            def __init__(self):          # rejects both kwargs
                raise TypeError("no")

            @classmethod
            def from_legacy_cache(cls, legacy):
                seen["legacy"] = legacy
                return cls.__new__(cls)

        with mock.patch.object(cache_utils, "DynamicCache", _Cache):
            out = kv_cache.build_cache_from_pairs(self._pairs())
        self.assertIsInstance(out, _Cache)
        self.assertEqual(len(seen["legacy"]), 2)

    def test_last_resort_fills_an_empty_cache_layer_by_layer(self):
        updates = []

        class _Cache:
            def __init__(self):
                self.layers = []

            def update(self, k, v, i):
                updates.append(i)
                self.layers.append((k, v))
                return k, v

        with mock.patch.object(cache_utils, "DynamicCache", _Cache):
            out = kv_cache.build_cache_from_pairs(self._pairs())
        self.assertEqual(updates, [0, 1])
        self.assertEqual(len(out.layers), 2)

    def test_returns_raw_tuples_when_no_constructor_works(self):
        """
        Very old transformers: `cache_utils` has no DynamicCache at all, and the
        raw tuple of (k, v) pairs is the native format. Returning it is correct;
        raising would break those versions for no reason.
        """
        with mock.patch.object(cache_utils, "DynamicCache",
                               side_effect=ImportError("gone")):
            out = kv_cache.build_cache_from_pairs(self._pairs())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)


class TestPrefixCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = build_tiny_model_and_tokenizer(
            goals=[GOAL], targets=[TARGET], seed=0,
        )
        cls.suffix = init_suffix_ids(cls.tokenizer, "! " * 4, 4, "cpu")
        cls.tm = TokenManager(cls.tokenizer, GOAL, TARGET, cls.suffix, "cpu")
        cls.cache = PrefixCache(cls.model, cls.tm.before_suffix_ids)

    def test_prefix_len_matches_the_prefix(self):
        self.assertEqual(self.cache.prefix_len, self.tm.before_len)

    def test_get_returns_a_fresh_object_each_call(self):
        """
        A shared, mutable cache object would be corrupted by the first forward
        pass that updates it in place — the exact bug this guards.
        """
        a = self.cache.get(1)
        b = self.cache.get(1)
        self.assertIsNot(a, b)

    def test_get_batches_without_disturbing_the_stored_pairs(self):
        before = [k.clone() for k, _ in self.cache.pairs]
        self.cache.get(4)
        for original, (k, _) in zip(before, self.cache.pairs):
            self.assertTrue(torch.equal(original, k))

    def test_batch_size_is_honoured(self):
        out = cache_to_pairs(self.cache.get(3))
        self.assertEqual(out[0][0].shape[0], 3)

    def test_accepts_a_one_dimensional_prefix(self):
        one_d = self.tm.before_suffix_ids.squeeze(0)
        self.assertEqual(
            PrefixCache(self.model, one_d).prefix_len, self.tm.before_len,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
