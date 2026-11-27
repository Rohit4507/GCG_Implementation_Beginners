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


def _is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


class QuantizationUnavailable(RuntimeError):
    """Raised when 4-bit was requested but could not be applied."""


def _require_bitsandbytes():
    """
    Import bitsandbytes, or fail loudly.

    `transformers` only *warns* when `quantization_config` asks for bitsandbytes
    and the package is missing, then loads the model unquantized. On a 7-8B
    model on a 16 GB card that is either an OOM or — worse — a silent fp16 run
    that a run record would go on to describe as 4-bit.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise QuantizationUnavailable(
            "load_in_4bit was requested but bitsandbytes is not installed, so "
            "the model would load unquantized. Install it with "
            "`pip install 'bitsandbytes>=0.46'` (or requirements-gpu.txt), or "
            "drop --load-in-4bit to record the run as fp16/bf16 honestly."
        ) from exc


def model_quantization(model, requested: bool) -> dict:
    """
    What quantization the returned model is *actually* in.

    Run records use this instead of echoing the CLI flag, so a record can never
    claim a 4-bit run that loaded fp16.
    """
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    ptype = None
    if quant is not None:
        ptype = (
            quant.get("bnb_4bit_quant_type")
            if isinstance(quant, dict)
            else getattr(quant, "bnb_4bit_quant_type", None)
        )
    is_4bit = bool(getattr(model, "is_quantized", False)) or quant is not None
    return {
        "requested_4bit": bool(requested),
        "applied_4bit": is_4bit,
        "quant_type": ptype if is_4bit else None,
        "weight_dtype": str(getattr(model, "dtype", "") or ""),
    }


def load_model_and_tokenizer(
    model_id: str,
    dtype: str = "float16",
    device: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
):
    """
    Load a causal LM and its tokenizer.

    4-bit bitsandbytes quantization is only attempted on CUDA; on CPU the
    flag is ignored (with a warning) so scripts stay runnable everywhere. If it
    *is* attempted, a missing bitsandbytes raises `QuantizationUnavailable`
    rather than silently loading an unquantized model.

    `trust_remote_code` is off by default: `--model` is arbitrary text, and
    this flag executes Python shipped inside the model repository. Every model
    named in the docs loads without it.
    """
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import transformers
    ver_parts = []
    for part in transformers.__version__.split(".")[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        ver_parts.append(int(digits) if digits else 0)
    dtype_key = "dtype" if tuple(ver_parts) >= (4, 56) else "torch_dtype"

    kwargs = {
        dtype_key: torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit:
        if _is_cuda_device(device):
            _require_bitsandbytes()
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            target = (
                f"cuda:{torch.cuda.current_device()}"
                if device == "cuda" and torch.cuda.is_available()
                else device
            )
            # Pin 4-bit weights to a single GPU so multi-GPU setups (e.g. Kaggle 2xT4)
            # do not split a small model across cuda:0 and cuda:1 and trigger device
            # mismatches or accelerate-hook errors.
            kwargs["device_map"] = {"": target}
        else:
            print("[model] load_in_4bit ignored: bitsandbytes requires CUDA")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    # Belt and braces: if quantization was asked for and did not happen, stop
    # here instead of writing a run record that misdescribes the model.
    if load_in_4bit and _is_cuda_device(device):
        actual = model_quantization(model, requested=True)
        if not actual["applied_4bit"]:
            raise QuantizationUnavailable(
                f"{model_id} loaded WITHOUT 4-bit quantization even though "
                "load_in_4bit was requested; refusing to record this as a "
                "4-bit run."
            )

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
