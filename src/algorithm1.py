import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from tqdm import trange

from .token_manager import TokenManager


def compute_token_gradient(
    model,
    token_manager: TokenManager,
    suffix_ids: Tensor,
) -> Tuple[Tensor, float]:
    """
    Step 1 of Algorithm 1:
    Computes gradients using the one-hot mutation proxy matrix.
    """
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight  # [vocab_size, embed_dim]
    vocab_size = embedding_matrix.shape[0]

    # Explicit memory reset to prevent leak
    model.zero_grad(set_to_none=True)

    # One-hot encode the suffix tokens
    one_hot = F.one_hot(
        suffix_ids,
        num_classes=vocab_size
    ).to(dtype=embedding_matrix.dtype)
    
    one_hot = one_hot.clone().detach().requires_grad_(True)

    # Suffix embeddings via matrix multiplication
    suffix_embeds = one_hot @ embedding_matrix  # [suffix_len, embed_dim]
    suffix_embeds = suffix_embeds.unsqueeze(0)   # [1, suffix_len, embed_dim]

    # Embeddings for the fixed parts
    full_ids = token_manager.build_input_ids(suffix_ids)  # [1, seq_len]
    full_embeds = embedding_layer(full_ids)               # [1, seq_len, embed_dim]

    # Splice in the differentiable suffix embeddings
    combined_embeds = torch.cat([
        full_embeds[:, :token_manager.suffix_slice.start, :],
        suffix_embeds,
        full_embeds[:, token_manager.suffix_slice.stop:, :],
    ], dim=1)

    # Forward pass
    outputs = model(inputs_embeds=combined_embeds)
    logits = outputs.logits

    # Extract logits that predict the target tokens
    target_logits = logits[:, token_manager.loss_slice, :]
    target_ids = token_manager.target_ids  # [1, target_len]

    # Cross-entropy loss
    loss = F.cross_entropy(
        target_logits.reshape(-1, vocab_size),
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
    Steps 2–3 of Algorithm 1 (Vectorized Version — No Loops! 🚀)
    """
    top_indices = (-grad).topk(topk, dim=1).indices  # [suffix_len, topk]
    suffix_len = suffix_ids.shape[0]
    device = suffix_ids.device

    # Expand original suffix sequence across the candidate dimension
    candidates = suffix_ids.unsqueeze(0).repeat(search_width, 1)

    # Vectorized indexing arrays for mutations
    batch_indices = torch.arange(1, search_width, device=device)
    random_positions = torch.randint(0, suffix_len, (search_width - 1,), device=device)
    random_topk_idx = torch.randint(0, topk, (search_width - 1,), device=device)

    # Extract target substitution tokens directly using gather mechanics
    sampled_tokens = top_indices[random_positions, random_topk_idx]

    # Apply all mutations in one single parallel clock cycle
    candidates[batch_indices, random_positions] = sampled_tokens

    return candidates


@torch.no_grad()
def evaluate_candidates(
    model,
    token_manager: TokenManager,
    candidate_suffix_ids: Tensor,
    mini_batch_size: int = 64,
) -> Tensor:
    """
    Step 5 of Algorithm 1: Mini-batched forward evaluation loop.
    """
    vocab_size = model.get_input_embeddings().weight.shape[0]
    num_candidates = candidate_suffix_ids.shape[0]
    all_losses = []

    for start in range(0, num_candidates, mini_batch_size):
        end = min(start + mini_batch_size, num_candidates)
        batch_suffix = candidate_suffix_ids[start:end]

        input_ids = token_manager.build_input_ids(batch_suffix)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        target_logits = logits[:, token_manager.loss_slice, :]
        target_ids = token_manager.target_ids.expand(batch_suffix.shape[0], -1)

        losses = F.cross_entropy(
            target_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(batch_suffix.shape[0], -1).mean(dim=1)

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
    verbose: bool = True,
) -> Tuple[Tensor, list]:
    """
    Full Algorithm 1 Execution Engine.
    """
    device = suffix_ids.device

    token_manager = TokenManager(
        tokenizer=tokenizer,
        goal=goal,
        target=target,
        suffix_ids=suffix_ids,
        device=str(device),
    )

    loss_history = []
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    iterator = trange(num_steps, desc="GCG Attack Metrics", disable=not verbose)

    for step in iterator:
        # Step 1: Compute Gradients
        grad, current_loss = compute_token_gradient(model, token_manager, suffix_ids)

        # Steps 2-3: Parallel Vectorized Candidate Generation
        candidates = sample_candidates(suffix_ids, grad, topk, search_width)

        # Step 4: Template-aware Re-tokenization Filter
        candidates = token_manager.filter_candidates(candidates)

        # Step 5: Batched Forward Pass Evaluator
        losses = evaluate_candidates(model, token_manager, candidates, mini_batch_size)

        # Step 6: Greedy selection
        best_idx = losses.argmin().item()
        new_loss = losses[best_idx].item()

        # Update running step trackers
        suffix_ids = candidates[best_idx].detach().clone()
        loss_history.append(new_loss)

        # Save checkpoint for best structural candidate discovered so far
        if new_loss < best_loss:
            best_loss = new_loss
            best_suffix = suffix_ids.clone()

        # Update progress bar metrics dynamically
        if verbose:
            iterator.set_postfix(step_loss=f"{new_loss:.4f}", best_loss=f"{best_loss:.4f}")

    return best_suffix, loss_history
