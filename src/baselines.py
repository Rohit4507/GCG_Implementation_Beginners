"""
Baselines for GCG comparison.

`run_random_search` mirrors the GCG loop's budget exactly — same
`search_width` candidate mutations per step, same re-tokenization filter,
same greedy update — but substitutes tokens uniformly at random from the
vocabulary instead of using the gradient top-k. The difference between the
two loss curves is precisely the value of the gradient signal.
"""
from typing import Tuple

import torch
from torch import Tensor
from tqdm import trange

from .algorithm1 import evaluate_candidates
from .kv_cache import PrefixCache
from .token_manager import TokenManager


def sample_random_candidates(
    suffix_ids: Tensor,
    search_width: int,
    vocab_size: int,
) -> Tensor:
    """
    Gradient-blind version of algorithm1.sample_candidates: one candidate
    keeps the current suffix; the rest each mutate one random position with
    a uniformly random vocabulary token.
    """
    suffix_len = suffix_ids.shape[-1]
    device = suffix_ids.device

    candidates = suffix_ids.reshape(1, -1).repeat(search_width, 1)
    positions = torch.randint(0, suffix_len, (search_width - 1,), device=device)
    tokens = torch.randint(0, vocab_size, (search_width - 1,), device=device)
    candidates[torch.arange(1, search_width, device=device), positions] = tokens
    return candidates


def run_random_search(
    model,
    tokenizer,
    goal: str,
    target: str,
    suffix_ids: Tensor,
    num_steps: int = 250,
    search_width: int = 512,
    mini_batch_size: int = 64,
    use_kv_cache: bool = True,
    verbose: bool = True,
) -> Tuple[Tensor, list]:
    """
    Random-search baseline of Algorithm 1.

    Returns (best_suffix_ids, loss_history) exactly like run_gcg, so the two
    are directly comparable.
    """
    device = suffix_ids.device
    vocab_size = model.get_input_embeddings().weight.shape[0]

    token_manager = TokenManager(
        tokenizer=tokenizer,
        goal=goal,
        target=target,
        suffix_ids=suffix_ids,
        device=str(device),
    )
    prefix_cache = PrefixCache(model, token_manager.before_suffix_ids) if use_kv_cache else None

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    iterator = trange(num_steps, desc="Random-search baseline", disable=not verbose)

    for step in iterator:
        candidates = sample_random_candidates(suffix_ids, search_width, vocab_size)
        candidates = token_manager.filter_candidates(candidates)

        losses = evaluate_candidates(
            model, token_manager, candidates, mini_batch_size, prefix_cache,
        )

        best_idx = losses.argmin().item()
        new_loss = losses[best_idx].item()

        suffix_ids = candidates[best_idx].detach().clone()
        loss_history.append(new_loss)

        if new_loss < best_loss:
            best_loss = new_loss
            best_suffix = suffix_ids.clone()

        if verbose:
            iterator.set_postfix(step_loss=f"{new_loss:.4f}", best_loss=f"{best_loss:.4f}")

    return best_suffix, loss_history


def target_loss(
    model,
    token_manager: TokenManager,
    suffix_ids: Tensor,
    prefix_cache: PrefixCache = None,
) -> float:
    """Target cross-entropy of a single suffix (via the candidate scorer)."""
    losses = evaluate_candidates(
        model, token_manager, suffix_ids.reshape(1, -1), 1, prefix_cache,
    )
    return losses[0].item()
