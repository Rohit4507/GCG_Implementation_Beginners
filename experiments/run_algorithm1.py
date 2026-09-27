"""
Run individual GCG (Algorithm 1) on a single benign prompt.
Optimized and synchronized with the enhanced TokenManager engine.
"""
#for run python -m experiments.run_algorithm1
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import GCGConfig
from src.token_manager import TokenManager
from src.algorithm1 import run_gcg


def main():
    cfg = GCGConfig()
    torch.manual_seed(cfg.seed)

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
        dtype=torch_dtype
    ).to(cfg.device)
    model.eval()


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Initialize suffix ----
    suffix_ids = tokenizer(
        cfg.suffix_init,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).to(cfg.device)

    # Truncate or pad to exact suffix_length safely
    if suffix_ids.shape[0] > cfg.suffix_length:
        suffix_ids = suffix_ids[:cfg.suffix_length]
    elif suffix_ids.shape[0] < cfg.suffix_length:
        pad_id = tokenizer.encode("!", add_special_tokens=False)[0]
        padding = torch.full(
            (cfg.suffix_length - suffix_ids.shape[0],),
            pad_id,
            device=cfg.device,
            dtype=suffix_ids.dtype,
        )
        suffix_ids = torch.cat([suffix_ids, padding])

    print(f"Initial suffix ({suffix_ids.shape[0]} tokens):")
    print(repr(tokenizer.decode(suffix_ids)))
    print()

    # ---- Run GCG ----
    goal = cfg.goals[0]
    target = cfg.targets[0]

    print(f"Goal:   {goal}")
    print(f"Target: {target}")
    print()

    best_suffix, loss_history = run_gcg(
        model=model,
        tokenizer=tokenizer,
        goal=goal,
        target=target,
        suffix_ids=suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
    )

    # ---- Results ----
    print("\n" + "=" * 60)
    print("Optimized suffix:")
    print(repr(tokenizer.decode(best_suffix)))

    # ---- Generate ----
    #from token_manager import TokenManager

    # FIX: Casting device parameter explicitly as a string representation block
    tm = TokenManager(
        tokenizer=tokenizer, 
        goal=goal, 
        target=target, 
        suffix_ids=best_suffix, 
        device=str(cfg.device)
    )
    prompt_ids = tm.build_prompt_ids(best_suffix)

    print("\nFull prompt sent to model:")
    print(tokenizer.decode(prompt_ids[0]))
    print()

    # Setup hyperparameter configuration flags cleanly for greedy execution
    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cfg.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = cfg.temperature
    else:
        gen_kwargs["do_sample"] = False  # Pure greedy parsing protocol

    #with torch.no_grad():
    #    output_ids = model.generate(prompt_ids, **gen_kwargs)

    #attention_mask = torch.ones_like(prompt_ids)

    attention_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    response = tokenizer.decode(
        output_ids[0][prompt_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print("Model response:")
    print(response)

    # ---- Plot loss curve ----
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Cross-Entropy Target Loss", color="crimson")
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss Matrix Value")
    plt.title("GCG Optimization — Individual Attack Execution Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_individual.png", dpi=150)
    plt.close() # Safe memory close for background tracking environments
    print("\nLoss plot saved to loss_individual.png")


if __name__ == "__main__":
    main()
