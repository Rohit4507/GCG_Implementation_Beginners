import time
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import trange

from .kv_cache import PrefixCache
from .token_manager import TokenManager
from .algorithm1 import (
    compute_token_gradient,
    sample_candidates,
    evaluate_candidates,
)


def run_universal_gcg(
    model,
    tokenizer,
    goals: List[str],
    targets: List[str],
    suffix_ids: Tensor,
    num_steps: int = 250,
    topk: int = 256,
    search_width: int = 512,
    mini_batch_size: int = 32,
    num_prompts_start: int = 1,
    prompt_add_interval: int = 50,
    use_kv_cache: bool = True,
    verbose: bool = True,
    stats_out: Optional[dict] = None,
) -> Tuple[Tensor, list]:
    """
    Algorithm 2: Universal/multi-prompt GCG Engine.

    A shared suffix is optimized across multiple goal/target pairs.
    Prompts are introduced progressively according to
    num_prompts_start and prompt_add_interval.

    If `stats_out` is a dict it is filled in with real compute counters. Note
    that a universal step is far more expensive than a single-prompt step:
    each step back-props once per active prompt and scores every candidate
    against every active prompt.
    """

    assert len(goals) == len(targets), (
        "Goals and targets dimension mismatch!"
    )

    device = suffix_ids.device
    total_prompts = len(goals)

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    candidate_forwards = 0
    gradient_forwards = 0
    t_start = time.perf_counter()

    # KV caches of the fixed prompt prefixes, built lazily and reused
    # across steps (the prefix of each prompt never changes).
    prefix_caches = {}

    iterator = trange(
        num_steps,
        desc="Universal GCG Loop",
        disable=not verbose,
    )

    for step in iterator:

        # ---------------------------------------------------------
        # Progressive prompt schedule
        # ---------------------------------------------------------
        num_active = min(
            num_prompts_start + step // prompt_add_interval,
            total_prompts,
        )

        # ---------------------------------------------------------
        # Build TokenManagers for currently active prompts
        # ---------------------------------------------------------
        active_tms = []

        for i in range(num_active):
            tm = TokenManager(
                tokenizer=tokenizer,
                goal=goals[i],
                target=targets[i],
                suffix_ids=suffix_ids,
                device=str(device),
            )

            active_tms.append(tm)

        # ---------------------------------------------------------
        # Step 1: Aggregate normalized gradients
        # ---------------------------------------------------------
        # Size from the embedding matrix, not the tokenizer. On real HF
        # models these differ (Qwen2.5-1.5B: tokenizer.vocab_size 151643 vs
        # 151936 embedding rows) and `compute_token_gradient` returns a
        # gradient shaped by the embedding matrix, so sizing the accumulator
        # from the tokenizer makes `aggregated_grad += grad` a hard crash.
        embedding_matrix = model.get_input_embeddings().weight
        grad_vocab = embedding_matrix.shape[0]

        aggregated_grad = torch.zeros(
            suffix_ids.shape[0],
            grad_vocab,
            device=device,
            dtype=embedding_matrix.dtype,
        )

        total_loss = 0.0

        for tm_idx, tm in enumerate(active_tms):

            if use_kv_cache and tm_idx not in prefix_caches:
                prefix_caches[tm_idx] = PrefixCache(
                    model, tm.before_suffix_ids
                )

            grad, loss_val = compute_token_gradient(
                model,
                tm,
                suffix_ids,
                prefix_caches.get(tm_idx),
            )

            # Normalize each prompt's gradient
            grad_norm = torch.linalg.vector_norm(grad)

            if grad_norm > 0:
                grad = grad / grad_norm

            aggregated_grad += grad
            total_loss += loss_val
            gradient_forwards += 1

        # Average normalized gradients
        aggregated_grad /= num_active

        mean_loss = total_loss / num_active

        # ---------------------------------------------------------
        # Steps 2–3: Sample candidate suffixes
        # ---------------------------------------------------------
        candidates = sample_candidates(
            suffix_ids,
            aggregated_grad,
            topk,
            search_width,
            valid_vocab=len(tokenizer),
        )

        # ---------------------------------------------------------
        # Step 4: Universal filtering
        # Candidate must remain valid for every active prompt.
        # ---------------------------------------------------------
        for tm in active_tms:

            candidates = tm.filter_candidates(
                candidates
            )

            # Stop if filtering leaves <= 1 candidate
            if candidates.shape[0] <= 1:
                break

        # ---------------------------------------------------------
        # Step 5: Evaluate candidates across ALL active prompts
        # ---------------------------------------------------------
        num_candidates = candidates.shape[0]

        total_candidate_losses = torch.zeros(
            num_candidates,
            device=device,
        )

        for tm_idx, tm in enumerate(active_tms):

            losses = evaluate_candidates(
                model,
                tm,
                candidates,
                mini_batch_size,
                prefix_caches.get(tm_idx),
            )

            total_candidate_losses += losses
            candidate_forwards += int(candidates.shape[0])

        # Average candidate loss across active prompts
        mean_candidate_losses = (
            total_candidate_losses / num_active
        )

        # ---------------------------------------------------------
        # Step 6: Greedy candidate selection
        # ---------------------------------------------------------
        best_idx = mean_candidate_losses.argmin().item()

        new_loss = mean_candidate_losses[
            best_idx
        ].item()

        suffix_ids = candidates[
            best_idx
        ].detach().clone()

        # ---------------------------------------------------------
        # Update global best suffix
        # ---------------------------------------------------------
        if new_loss < best_loss:

            best_loss = new_loss
            best_suffix = suffix_ids.clone()

        # Store loss history
        loss_history.append(
            (step, new_loss)
        )

        # ---------------------------------------------------------
        # Progress display
        # ---------------------------------------------------------
        suffix_str = tokenizer.decode(
            suffix_ids,
            skip_special_tokens=True,
        )

        if verbose:
            iterator.set_postfix(
                mean_loss=f"{new_loss:.4f}",
                prompts=f"{num_active}/{total_prompts}",
                suffix=(
                    suffix_str[:15] + "..."
                    if len(suffix_str) > 15
                    else suffix_str
                ),
            )

    if stats_out is not None:
        stats_out.update(
            steps_run=len(loss_history),
            gradient_forwards=gradient_forwards,
            candidate_forwards=candidate_forwards,
            wall_clock_s=time.perf_counter() - t_start,
            best_loss=best_loss,
        )

    return best_suffix, loss_history