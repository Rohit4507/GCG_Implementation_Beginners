"""
Model / tokenizer loading utilities shared by all experiments.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Map a dtype name to torch dtype, with CPU-safe fallbacks."""
    # float16 matmuls are not reliably supported on CPU.
    if device == "cpu" and dtype in ("float16", "bfloat16"):
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_model_and_tokenizer(
    model_id: str,
    dtype: str = "float16",
    device: str = "auto",
    load_in_4bit: bool = False,
):
    """
    Load a causal LM and its tokenizer.

    4-bit bitsandbytes quantization is only attempted on CUDA; on CPU the
    flag is ignored (with a warning) so scripts stay runnable everywhere.
    """
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if load_in_4bit:
        if device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            print("[model] load_in_4bit ignored: bitsandbytes requires CUDA")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda" or "quantization_config" not in kwargs:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def init_suffix_ids(tokenizer, suffix_init: str, suffix_length: int, device: str) -> torch.Tensor:
    """
    Tokenize the initial suffix string and force it to exactly
    `suffix_length` tokens (truncate or pad with '!').
    """
    suffix_ids = tokenizer(
        suffix_init,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).to(device)

    if suffix_ids.shape[0] > suffix_length:
        suffix_ids = suffix_ids[:suffix_length]
    elif suffix_ids.shape[0] < suffix_length:
        pad_id = tokenizer.encode("!", add_special_tokens=False)[0]
        padding = torch.full(
            (suffix_length - suffix_ids.shape[0],),
            pad_id,
            device=suffix_ids.device,
            dtype=suffix_ids.dtype,
        )
        suffix_ids = torch.cat([suffix_ids, padding])
    return suffix_ids
