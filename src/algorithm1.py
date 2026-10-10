"""
Algorithm 1 of the GCG paper (Zou et al., 2023) — per-prompt attack.

Pipeline per step:
  1. Gradient of the target-token cross-entropy w.r.t. a one-hot proxy of the
     suffix tokens (equivalent to w.r.t. the suffix embeddings).
  2. Top-k candidate substitutions per suffix position from -grad.
  3. Sample `search_width` single-token mutations.
  4. Re-tokenization validity filter (done by TokenManager.filter_candidates).
  5. Mini-batched forward pass to score every candidate.
  6. Greedy update: keep the candidate with the lowest loss.

With `use_kv_cache=True` the fixed prompt prefix is forwarded once (see
kv_cache.PrefixCache) and reused every step.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from tqdm import trange

from .kv_cache import PrefixCache
from .token_manager import TokenManager


def compute_token_gradient(
    model,
    token_manager: TokenManager,
    suffix_ids: Tensor,
    prefix_cache: Optional[PrefixCache] = None,
) -> Tuple[Tensor, float]:
    """
    Step 1 of Algorithm 1:
    Computes gradients using the one-hot mutation proxy matrix.
    """
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight  # [vocab_size, embed_dim]
    vocab_size = embedding_matrix.shape[0]

    # Explicit memory reset to avoid gradient accumulation
    model.zero_grad(set_to_none=True)

    # One-hot encode the suffix tokens
    one_hot = F.one_hot(
        suffix_ids.long(),
        num_classes=vocab_size,
    ).to(dtype=embedding_matrix.dtype)

    one_hot = one_hot.clone().detach().requires_grad_(True)

    # Suffix embeddings via matrix multiplication (this keeps the
    # computation differentiable w.r.t. the one-hot matrix).
    suffix_embeds = one_hot @ embedding_matrix  # [suffix_len, embed_dim]
    suffix_embeds = suffix_embeds.unsqueeze(0)  # [1, suffix_len, embed_dim]

    if prefix_cache is not None:
        # Only forward suffix + after-suffix + target; the prefix KV is cached.
        after_embeds = embedding_layer(token_manager.after_suffix_ids)
        target_embeds = embedding_layer(token_manager.target_ids)
        combined_embeds = torch.cat(
            [suffix_embeds, after_embeds, target_embeds], dim=1
        )
        outputs = model(
            inputs_embeds=combined_embeds,
            past_key_values=prefix_cache.get(1),
            use_cache=False,
        )
        # Loss slice is absolute; shift it to the forwarded chunk.
        rel = slice(
            token_manager.loss_slice.start - prefix_cache.prefix_len,
            token_manager.loss_slice.stop - prefix_cache.prefix_len,
        )
        target_logits = outputs.logits[:, rel, :]
    else:
        # Forward the full sequence, splicing in the differentiable suffix.
        full_ids = token_manager.build_input_ids(suffix_ids)  # [1, seq_len]
        full_embeds = embedding_layer(full_ids)  # [1, seq_len, embed_dim]

        combined_embeds = torch.cat([
            full_embeds[:, :token_manager.suffix_slice.start, :],
            suffix_embeds,
            full_embeds[:, token_manager.suffix_slice.stop:, :],
        ], dim=1)

        outputs = model(inputs_embeds=combined_embeds)
        target_logits = outputs.logits[:, token_manager.loss_slice, :]

    target_ids = token_manager.target_ids  # [1, target_len]

    # Cross-entropy loss on the target response only
    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
    )

    loss.backward()

    grad = one_hot.grad.detach().clone()  # [suffix_len, vocab_size]
    loss_val = loss.item()

    return grad, loss_val


def sample_candidates(
    suffix_ids: Tensor,
    grad: Tensor,
    topk: int,
    search_width: int,
) -> Tensor:
    """
    Steps 2–3 of Algorithm 1 (vectorized): one candidate keeps the current
    suffix, the remaining `search_width - 1` each get a single random
    position substituted with a random token from that position's top-k.
    """
    top_indices = (-grad).topk(topk, dim=1).indices  # [suffix_len, topk]
    suffix_len = suffix_ids.shape[-1]
    device = suffix_ids.device

    # Expand original suffix sequence across the candidate dimension
    candidates = suffix_ids.reshape(1, -1).repeat(search_width, 1)

    # Vectorized indexing arrays for mutations
    batch_indices = torch.arange(1, search_width, device=device)
    random_positions = torch.randint(0, suffix_len, (search_width - 1,), device=device)
    random_topk_idx = torch.randint(0, topk, (search_width - 1,), device=device)

    # Extract target substitution tokens directly
    sampled_tokens = top_indices[random_positions, random_topk_idx]

    # Apply all mutations in one shot
    candidates[batch_indices, random_positions] = sampled_tokens

    return candidates


@torch.no_grad()
def evaluate_candidates(
    model,
    token_manager: TokenManager,
    candidate_suffix_ids: Tensor,
    mini_batch_size: int = 64,
    prefix_cache: Optional[PrefixCache] = None,
) -> Tensor:
    """
    Step 5 of Algorithm 1: mini-batched forward pass that scores every
    candidate suffix by its target-token cross-entropy loss.
    """
    vocab_size = model.get_input_embeddings().weight.shape[0]
    num_candidates = candidate_suffix_ids.shape[0]
    all_losses = []

    for start in range(0, num_candidates, mini_batch_size):
        end = min(start + mini_batch_size, num_candidates)
        batch_suffix = candidate_suffix_ids[start:end]
        bsz = batch_suffix.shape[0]

        if prefix_cache is not None:
            after = token_manager.after_suffix_ids.expand(bsz, -1)
            target = token_manager.target_ids.expand(bsz, -1)
            input_ids = torch.cat([batch_suffix, after, target], dim=1)
            outputs = model(
                input_ids=input_ids,
                past_key_values=prefix_cache.get(bsz),
                use_cache=False,
            )
            rel = slice(
                token_manager.loss_slice.start - prefix_cache.prefix_len,
                token_manager.loss_slice.stop - prefix_cache.prefix_len,
            )
            target_logits = outputs.logits[:, rel, :]
        else:
            input_ids = token_manager.build_input_ids(batch_suffix)
            outputs = model(input_ids=input_ids)
            target_logits = outputs.logits[:, token_manager.loss_slice, :]

        target_ids = token_manager.target_ids.expand(bsz, -1)

        losses = F.cross_entropy(
            target_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(bsz, -1).mean(dim=1)

        all_losses.append(losses)

    return torch.cat(all_losses)


def run_gcg(
    model,
    tokenizer,
    goal: str,
    target: str,
    suffix_ids: Tensor,
    num_steps: int = 250,
    topk: int = 256,
    search_width: int = 512,
    mini_batch_size: int = 64,
    use_kv_cache: bool = True,
    verbose: bool = True,
) -> Tuple[Tensor, list]:
    """
    Full Algorithm 1 execution loop.

    Returns (best_suffix_ids, loss_history).
    """
    device = suffix_ids.device

    token_manager = TokenManager(
        tokenizer=tokenizer,
        goal=goal,
        target=target,
        suffix_ids=suffix_ids,
        device=str(device),
    )

    prefix_cache = None
    if use_kv_cache:
        prefix_cache = PrefixCache(model, token_manager.before_suffix_ids)

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    iterator = trange(num_steps, desc="GCG Attack Metrics", disable=not verbose)

    for step in iterator:
        # Step 1: Compute gradients
        grad, current_loss = compute_token_gradient(
            model, token_manager, suffix_ids, prefix_cache,
        )

        # Steps 2-3: vectorized candidate generation
        candidates = sample_candidates(suffix_ids, grad, topk, search_width)

        # Step 4: template-aware re-tokenization filter
        candidates = token_manager.filter_candidates(candidates)

        # Step 5: batched forward-pass evaluation
        losses = evaluate_candidates(
            model, token_manager, candidates, mini_batch_size, prefix_cache,
        )

        # Step 6: greedy selection
        best_idx = losses.argmin().item()
        new_loss = losses[best_idx].item()

        suffix_ids = candidates[best_idx].detach().clone()
        loss_history.append(new_loss)

        # Save checkpoint for best suffix discovered so far
        if new_loss < best_loss:
            best_loss = new_loss
            best_suffix = suffix_ids.clone()

        if verbose:
            iterator.set_postfix(step_loss=f"{new_loss:.4f}", best_loss=f"{best_loss:.4f}")

    return best_suffix, loss_history
