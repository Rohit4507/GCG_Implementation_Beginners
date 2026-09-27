import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from tqdm import trange

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
    verbose: bool = True,
) -> Tuple[Tensor, list]:
    """
    Algorithm 2: Universal/multi-prompt GCG Engine (Optimized & Vectorized 🚀)
    """
    assert len(goals) == len(targets), "Goals and targets dimension mismatch!"
    device = suffix_ids.device
    total_prompts = len(goals)

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    iterator = trange(num_steps, desc="Universal GCG Loop", disable=not verbose)

    for step in iterator:
        # ---- Progressive prompt schedule (Algorithm 2, line 3) ----
        num_active = min(
            num_prompts_start + step // prompt_add_interval,
            total_prompts,
        )
        
        # Build active managers dynamically based on the current step's suffix configuration
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

        # ---- Step 1: Aggregate gradients across active prompts ----
        aggregated_grad = torch.zeros(suffix_ids.shape[0], tokenizer.vocab_size, device=device)
        total_loss = 0.0

        for tm in active_tms:
            grad, loss_val = compute_token_gradient(model, tm, suffix_ids)
            aggregated_grad += grad
            total_loss += loss_val

        # Average aggregation mapping
        aggregated_grad /= num_active
        mean_loss = total_loss / num_active

        # ---- Steps 2–3: Sample candidates using aggregated gradient ----
        candidates = sample_candidates(suffix_ids, aggregated_grad, topk, search_width)

        # ---- Step 4: Universal Filter via Cross-Prompt Validation ----
        # Suffix must not break boundaries on ANY active prompt structure
        for tm in active_tms:
            candidates = tm.filter_candidates(candidates)
            if candidates.shape[0] <= 1: # Break early if filter drops everything
                break

        # ---- Step 5: Evaluate candidates across ALL active prompts ----
        num_candidates = candidates.shape[0]
        total_candidate_losses = torch.zeros(num_candidates, device=device)

        for tm in active_tms:
            losses = evaluate_candidates(model, tm, candidates, mini_batch_size)
            total_candidate_losses += losses

        mean_candidate_losses = total_candidate_losses / num_active

        # ---- Step 6: Greedy selection mechanics ----
        best_idx = mean_candidate_losses.argmin().item()
        new_loss = mean_candidate_losses[best_idx].item()

        suffix_ids = candidates[best_idx].detach().clone()

        if new_loss < best_loss:
            best_loss = new_loss
            best_suffix = suffix_ids.clone()

        loss_history.append((step, new_loss))

        # Dynamic string metrics output console display
        suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=True)
        if verbose:
            iterator.set_postfix(
                mean_loss=f"{new_loss:.4f}",
                prompts=f"{num_active}/{total_prompts}",
                suffix=suffix_str[:15] + "..." if len(suffix_str) > 15 else suffix_str
            )

    return best_suffix, loss_history
