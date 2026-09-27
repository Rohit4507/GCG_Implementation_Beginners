import torch
from transformers import AutoTokenizer


_SUFFIX_SENTINEL = "SUFFIXPLACEHOLDER"


class TokenManager:
    """
    Manages the prompt layout for GCG.

    Layout:

        [chat prefix + goal]
        [optimized suffix]
        [chat suffix / assistant prompt]
        [target]

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
    ):
        self.tokenizer = tokenizer
        self.goal = goal
        self.target = target
        self.device = device
        self.suffix_len = suffix_ids.shape[0]

        # ---------------------------------------------------------
        # 1. Create the chat template with a text sentinel
        # ---------------------------------------------------------
        messages = [
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

        valid_mask = []

        for i in range(candidate_suffix_ids.shape[0]):

            suffix_str = self.tokenizer.decode(
                candidate_suffix_ids[i],
                clean_up_tokenization_spaces=False,
            )

            re_encoded = self.tokenizer(
                suffix_str,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(0).to(self.device)

            valid = (
                re_encoded.shape[0] == self.suffix_len
                and torch.equal(
                    re_encoded,
                    candidate_suffix_ids[i],
                )
            )

            valid_mask.append(valid)

        valid_mask = torch.tensor(
            valid_mask,
            device=self.device,
            dtype=torch.bool,
        )

        if valid_mask.sum() == 0:
            return candidate_suffix_ids[:1]

        return candidate_suffix_ids[valid_mask]