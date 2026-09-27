import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import GCGConfig
from src.token_manager import TokenManager
from src.algorithm1 import (
    compute_token_gradient,
    sample_candidates,
    evaluate_candidates,
)


def main():
    cfg = GCGConfig()
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # ---- Load model ----
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[cfg.dtype]

    print(f"Loading {cfg.model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch_dtype,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Initialize suffix ----
    suffix_ids = tokenizer(
        cfg.suffix_init,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).to(device)

    if suffix_ids.shape[0] > cfg.suffix_length:
        suffix_ids = suffix_ids[:cfg.suffix_length]
    elif suffix_ids.shape[0] < cfg.suffix_length:
        pad_id = tokenizer.encode(
            "!", add_special_tokens=False
        )[0]
        padding = torch.full(
            (cfg.suffix_length - suffix_ids.shape[0],),
            pad_id,
            device=device,
            dtype=suffix_ids.dtype,
        )
        suffix_ids = torch.cat([suffix_ids, padding])

    # ---- Check 1: Suffix shape ----
    print("\n[1] Suffix shape:", tuple(suffix_ids.shape))
    assert suffix_ids.shape == (8,)
    print("PASS")

    # ---- Token Manager ----
    tm = TokenManager(
        tokenizer=tokenizer,
        goal=cfg.goals[0],
        target=cfg.targets[0],
        suffix_ids=suffix_ids,
        device=str(device),
    )

    # ---- Check 2: Gradient ----
    grad, initial_loss = compute_token_gradient(
        model, tm, suffix_ids
    )

    print("\n[2] Gradient shape:", tuple(grad.shape))
    print("Expected:", (8, tokenizer.vocab_size))
    assert grad.shape == (8, tokenizer.vocab_size)
    print("PASS")

    # ---- Check 3: NaN / Inf ----
    has_nan = torch.isnan(grad).any().item()
    has_inf = torch.isinf(grad).any().item()

    print("\n[3] NaN:", has_nan)
    print("    Inf:", has_inf)

    assert not has_nan
    assert not has_inf
    print("PASS")

    # ---- Check 4: Candidate generation ----
    candidates = sample_candidates(
        suffix_ids,
        grad,
        topk=10,
        search_width=16,
    )

    print("\n[4] Candidates before filtering:",
          candidates.shape[0])

    candidates = tm.filter_candidates(candidates)

    print("    Candidates after filtering:",
          candidates.shape[0])

    assert candidates.shape[0] > 0
    print("PASS")

    # ---- Check 5: Candidate evaluation ----
    losses = evaluate_candidates(
        model,
        tm,
        candidates,
        mini_batch_size=8,
    )

    print("\n[5] Loss shape:", tuple(losses.shape))
    print("    Initial loss:", float(initial_loss))
    print("    Best candidate loss:", losses.min().item())

    assert losses.shape[0] == candidates.shape[0]
    assert not torch.isnan(losses).any()
    assert not torch.isinf(losses).any()
    print("PASS")

    # ---- Check 6: input_ids vs inputs_embeds ----
    prompt_ids = tm.build_prompt_ids(suffix_ids)

    with torch.no_grad():
        logits_ids = model(
            input_ids=prompt_ids
        ).logits

        inputs_embeds = model.get_input_embeddings()(
            prompt_ids
        )

        logits_embeds = model(
            inputs_embeds=inputs_embeds
        ).logits

    difference = torch.max(
        torch.abs(logits_ids - logits_embeds)
    ).item()

    print("\n[6] input_ids vs inputs_embeds")
    print("    Max difference:", difference)

    assert difference < 1e-4
    print("PASS")

    # ---- Final ----
    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()