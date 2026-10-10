"""
KV-cache utilities for the *fixed* part of the prompt.

Everything before the adversarial suffix never changes during optimization,
so we can forward it once and reuse its key/value tensors for every gradient
and candidate-evaluation step (the trick used in Brian Pulfer's GCG write-up).

Cache formats differ across transformers versions:
  - v4.x: DynamicCache with .key_cache/.value_cache (or to_legacy_cache),
          legacy tuples accepted on input.
  - v5.x: DynamicCache with .layers[i].keys/.values; legacy tuples are
          REJECTED on input.

We therefore store raw (key, value) tensor pairs and rebuild a cache object
in whatever format the installed version accepts.
"""
from typing import Any, List, Tuple

import torch


def cache_to_pairs(cache: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract [(k, v), ...] from any past_key_values format."""
    if isinstance(cache, (tuple, list)):
        pairs = []
        for layer in cache:
            if isinstance(layer, (tuple, list)):
                pairs.append((layer[0], layer[1]))
            else:
                raise TypeError(f"Unexpected layer type in cache: {type(layer)}")
        return pairs
    if hasattr(cache, "layers"):  # transformers v5
        return [(lay.keys, lay.values) for lay in cache.layers]
    if hasattr(cache, "key_cache"):  # transformers v4 DynamicCache
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "to_legacy_cache"):
        return [(k, v) for k, v in cache.to_legacy_cache()]
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def build_cache_from_pairs(pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Build a past_key_values object from (k, v) pairs, using whichever
    constructor the installed transformers version supports. Falls back to
    raw legacy tuples (fine on transformers v4).
    """
    # transformers v5 (and late v4): explicit iterable of (k, v) tuples
    try:
        from transformers.cache_utils import DynamicCache
        try:
            return DynamicCache(ddp_cache_data=tuple(pairs))
        except TypeError:
            pass
        # Older v4 signature: DynamicCache(legacy_cache=...)
        try:
            return DynamicCache(legacy_cache=tuple(pairs))
        except TypeError:
            pass
        # v4 classmethod
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(tuple(pairs))
        # Last resort: fill an empty cache layer by layer
        cache = DynamicCache()
        for i, (k, v) in enumerate(pairs):
            cache.update(k, v, i)
        return cache
    except Exception:
        # Very old versions: raw tuples are the native format
        return tuple(pairs)


def batchify_pairs(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Repeat (B=1) (k, v) pairs along the batch dimension."""
    if batch_size == 1:
        return pairs
    return [
        (
            k.repeat(batch_size, *([1] * (k.dim() - 1))),
            v.repeat(batch_size, *([1] * (v.dim() - 1))),
        )
        for k, v in pairs
    ]


class PrefixCache:
    """
    KV cache of the prompt prefix (everything before the suffix).

    Built once per TokenManager, then reused for:
      - the gradient forward pass (batch size 1), and
      - candidate evaluation (batch size = mini-batch width).

    `get()` always returns a FRESH cache object, so in-place updates by
    any version of the model forward can never corrupt the stored state.
    """

    def __init__(self, model, prefix_ids: torch.Tensor):
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            out = model(
                inputs_embeds=embed_layer(prefix_ids),
                use_cache=True,
            )
        self.pairs = cache_to_pairs(out.past_key_values)
        self.prefix_len = prefix_ids.shape[1]

    def get(self, batch_size: int = 1):
        return build_cache_from_pairs(batchify_pairs(self.pairs, batch_size))
