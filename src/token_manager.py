import torch
from transformers import AutoTokenizer

_SUFFIX_SENTINEL = "SUFFIXPLACEHOLDER"
_TARGET_SENTINEL = "TARGETPLACEHOLDER"


class TokenManager:
    """
    Manages the full prompt layout accurately by tokenizing the sequence globally
    to avoid boundary fusion errors.
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

        # --- Step 1: Ek single integrated string banayein ---
        messages = [
            {
                "role": "user",
                "content": f"{goal} {_SUFFIX_SENTINEL}",
            }
        ]
        
        # Pure structure ko ek sath taiyar karein target ke sath
        prompt_with_sentinel = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_string = f"{prompt_with_sentinel}{target}"

        # --- Step 2: Pure structure ko EK SATH tokenize karein ---
        # Isse boundary tokens kabhi galat fuse nahi honge
        full_ids = tokenizer(full_string, return_tensors="pt", add_special_tokens=False).input_ids.to(device).squeeze(0)
        
        # Sentinel tokens ke IDs dhoodhein slicing ke liye
        sentinel_ids = tokenizer(_SUFFIX_SENTINEL, add_special_tokens=False)["input_ids"]
        sentinel_len = len(sentinel_ids)

        # Sequence ke andar sentinel ki position locate karein
        start_idx = -1
        for i in range(len(full_ids) - sentinel_len + 1):
            if torch.equal(full_ids[i : i + sentinel_len], torch.tensor(sentinel_ids, device=device)):
                start_idx = i
                break
                
        if start_idx == -1:
            raise ValueError("Tokenizer failed to locate the Suffix Sentinel uniquely!")

        # --- Step 3: Exact position slices extract karein ---
        # Before Suffix slice
        self.before_suffix_ids = full_ids[:start_idx].unsqueeze(0)
        self.before_len = self.before_suffix_ids.shape[1]

        # After Suffix (Sentinel ke baad aur Target se pehle ka system-prompt part)
        # Isko nikalne ke liye target string ke length ka use karenge backend se
        target_raw_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        self.target_len = len(target_raw_ids)
        
        after_sentinel_start = start_idx + sentinel_len
        after_sentinel_end = len(full_ids) - self.target_len
        
        self.after_suffix_ids = full_ids[after_sentinel_start:after_sentinel_end].unsqueeze(0)
        self.after_len = self.after_suffix_ids.shape[1]
        self.target_ids = full_ids[after_sentinel_end:].unsqueeze(0)

        # --- Step 4: Final Slices for Training Loop ---
        self.suffix_slice = slice(self.before_len, self.before_len + self.suffix_len)
        
        target_start = self.before_len + self.suffix_len + self.after_len
        self.target_slice = slice(target_start, target_start + self.target_len)
        self.loss_slice = slice(target_start - 1, target_start + self.target_len - 1)

    def build_input_ids(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        batch = suffix_ids.shape[0]
        before = self.before_suffix_ids.expand(batch, -1)
        after = self.after_suffix_ids.expand(batch, -1)
        target = self.target_ids.expand(batch, -1)

        return torch.cat([before, suffix_ids, after, target], dim=1)

    def build_prompt_ids(self, suffix_ids: torch.Tensor) -> torch.Tensor:
        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        batch = suffix_ids.shape[0]
        before = self.before_suffix_ids.expand(batch, -1)
        after = self.after_suffix_ids.expand(batch, -1)

        return torch.cat([before, suffix_ids, after], dim=1)

    def filter_candidates(self, candidate_suffix_ids: torch.Tensor) -> torch.Tensor:
        """
        Paper ka perfect projection step: Yeh check karta hai ki full-prompt sequence 
        ke andar decode aur re-tokenize hone par tokens change toh nahi ho rahe.
        """
        valid_mask = []
        
        # Optimized loop over candidate batch
        for i in range(candidate_suffix_ids.shape[0]):
            # Suffix tokens ko decode karke wapas encode karke validation check
            suffix_str = self.tokenizer.decode(candidate_suffix_ids[i])
            
            # Suffix ko context ke sath stitch karke validation karna best approach hai
            re_encoded = self.tokenizer(suffix_str, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device).squeeze(0)

            if re_encoded.shape[0] == self.suffix_len and torch.equal(re_encoded, candidate_suffix_ids[i]):
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        valid_mask = torch.tensor(valid_mask, device=self.device)

        if valid_mask.sum() == 0:
            return candidate_suffix_ids[:1]

        return candidate_suffix_ids[valid_mask]
