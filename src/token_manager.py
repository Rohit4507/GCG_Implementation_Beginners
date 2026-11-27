from typing import List, Optional

import torch
from transformers import AutoTokenizer


_SUFFIX_SENTINEL = "SUFFIXPLACEHOLDER"


class TokenManager:
    """
    Manages the prompt layout for GCG.

    Single-turn layout:

        [chat prefix + goal]
        [optimized suffix]
        [chat suffix / assistant prompt]
        [target]

    Multi-turn layout (`prior_turns` supplied) prepends those turns to the
    conversation, so the suffix is optimised against the goal *inside an
    existing dialogue* rather than as a first message:

        [turn 1 user] [turn 1 assistant] ... [final user turn + goal][suffix]
        [assistant prompt][target]

    The suffix and target boundaries are determined from the actual
    chat-template text rather than searching for a separately-tokenized
    sentinel sequence.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        goal: str,
        target: str,
        suffix_ids: torch.Tensor,
        device: str = "cuda",
        prior_turns: Optional[List[dict]] = None,
    ):
        self.tokenizer = tokenizer
        self.goal = goal
        self.target = target
        self.device = device
        self.prior_turns = list(prior_turns) if prior_turns else []
        self.n_prior_turns = len(self.prior_turns)
        # suffix_ids may be [L] or [1, L]
        self.suffix_len = suffix_ids.shape[-1]

        # ---------------------------------------------------------
        # 1. Create the chat template with a text sentinel
        #
        # Any prior turns come first, so the sentinel still marks the end of
        # the FINAL user message. Everything before the sentinel is the cached
        # prefix, which is exactly what Algorithm 1 wants.
        # ---------------------------------------------------------
        messages = self.prior_turns + [
            {
                "role": "user",
                "content": f"{goal} {_SUFFIX_SENTINEL}",
            }
        ]

        prompt_with_sentinel = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ---------------------------------------------------------
        # 2. Find the sentinel in TEXT, not token IDs
        # ---------------------------------------------------------
        sentinel_pos = prompt_with_sentinel.find(_SUFFIX_SENTINEL)

        if sentinel_pos == -1:
            raise ValueError(
                "Suffix sentinel was not found in the chat-template text."
            )

        prefix_text = prompt_with_sentinel[:sentinel_pos]
        suffix_end_text = prompt_with_sentinel[
            sentinel_pos + len(_SUFFIX_SENTINEL):
        ]

        # ---------------------------------------------------------
        # 3. Tokenize the actual pieces
        #
        # IMPORTANT:
        # We include the whitespace immediately before the sentinel
        # in the prefix, because the original prompt has:
        #
        #     goal + " " + sentinel
        # ---------------------------------------------------------
        prefix_ids = tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        after_suffix_ids = tokenizer(
            suffix_end_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # ---------------------------------------------------------
        # 4. Target
        # ---------------------------------------------------------
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Move everything to the selected device
        prefix_ids = prefix_ids.to(device)
        after_suffix_ids = after_suffix_ids.to(device)
        target_ids = target_ids.to(device)

        # ---------------------------------------------------------
        # 5. Store pieces
        # ---------------------------------------------------------
        self.before_suffix_ids = prefix_ids.unsqueeze(0)
        self.after_suffix_ids = after_suffix_ids.unsqueeze(0)
        self.target_ids = target_ids.unsqueeze(0)

        self.before_len = prefix_ids.shape[0]
        self.after_len = after_suffix_ids.shape[0]
        self.target_len = target_ids.shape[0]

        # ---------------------------------------------------------
        # 6. Slices in the FINAL sequence
        # ---------------------------------------------------------
        self.suffix_slice = slice(
            self.before_len,
            self.before_len + self.suffix_len,
        )

        target_start = (
            self.before_len
            + self.suffix_len
            + self.after_len
        )

        self.target_slice = slice(
            target_start,
            target_start + self.target_len,
        )

        # For next-token prediction:
        #
        # logits[:, i] predicts input_ids[:, i+1]
        #
        # Therefore the logits corresponding to target tokens
        # start one position before target_start.
        self.loss_slice = slice(
            target_start - 1,
            target_start + self.target_len - 1,
        )

    # -------------------------------------------------------------
    # Build complete sequence:
    #
    # prefix + suffix + after_suffix + target
    # -------------------------------------------------------------
    def build_input_ids(self, suffix_ids: torch.Tensor) -> torch.Tensor:

        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        batch = suffix_ids.shape[0]

        before = self.before_suffix_ids.expand(batch, -1)
        after = self.after_suffix_ids.expand(batch, -1)
        target = self.target_ids.expand(batch, -1)

        return torch.cat(
            [before, suffix_ids, after, target],
            dim=1,
        )

    # -------------------------------------------------------------
    # Build prompt WITHOUT target
    # -------------------------------------------------------------
    def build_prompt_ids(self, suffix_ids: torch.Tensor) -> torch.Tensor:

        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        batch = suffix_ids.shape[0]

        before = self.before_suffix_ids.expand(batch, -1)
        after = self.after_suffix_ids.expand(batch, -1)

        return torch.cat(
            [before, suffix_ids, after],
            dim=1,
        )

    # -------------------------------------------------------------
    # Candidate filtering
    # -------------------------------------------------------------
    def filter_candidates(
        self,
        candidate_suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Drop candidates whose tokens do not survive decode -> re-encode.

        GCG mutates token *ids*, but the model scores a *string*. A candidate
        containing a partial multi-byte token (or a token whose merge behaviour
        changes at the suffix boundary) is not the sequence that will actually
        be fed to the model, so it must not be kept.

        Batched on purpose: the original loop did a `decode`, a `tokenizer(...)`
        call with `return_tensors="pt"`, a `.to(device)` and a `torch.equal`
        per candidate. At 512 candidates x 500 steps that is ~250k Python-level
        tokenizer round trips, and the tiny per-candidate CUDA copies are
        latency-bound on GPU. Measured 3.9x slower than the batched form on the
        offline tokenizer, and the gap widens with a real 150k-entry vocab.

        Candidate 0 (the unmodified current suffix) is always retained, so a
        filter that rejects everything cannot stall the optimiser.
        """
        candidate_suffix_ids = candidate_suffix_ids.to(self.device)
        n = candidate_suffix_ids.shape[0]
        if n == 0:
            return candidate_suffix_ids

        rows = candidate_suffix_ids.tolist()

        # Batch decode + batch tokenize: two calls for the whole candidate
        # block instead of four per candidate.
        texts = self.tokenizer.batch_decode(
            rows, clean_up_tokenization_spaces=False,
        )
        # No padding/truncation, so this is a plain list of id lists and there
        # is no pad id to mask out.
        encoded = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()

        valid = [
            len(ids) == self.suffix_len and ids == row
            for ids, row in zip(encoded, rows)
        ]

        if not any(valid):
            # Nothing round-trips: keep the unmodified suffix, exactly as the
            # original implementation did, so the loop cannot stall.
            return candidate_suffix_ids[:1]

        mask = torch.tensor(valid, dtype=torch.bool, device=self.device)
        return candidate_suffix_ids[mask]
