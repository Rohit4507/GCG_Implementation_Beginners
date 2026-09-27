"""
Run universal GCG (Algorithm 2) on multiple benign prompts.

This finds ONE suffix that works across ALL prompts simultaneously.
"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import GCGConfig
from src.algorithm2 import run_universal_gcg
from src.token_manager import TokenManager

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
        cfg.model_id, torch_dtype=torch_dtype
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
    print(f"\nNumber of prompts: {len(cfg.goals)}")
    print(f"Progressive schedule: start={cfg.num_prompts_start}, "
          f"add 1 every {cfg.prompt_add_interval} steps")
    print()

    # ---- Run Universal GCG ----
    best_suffix, loss_history = run_universal_gcg(
        model=model,
        tokenizer=tokenizer,
        goals=cfg.goals,
        targets=cfg.targets,
        suffix_ids=suffix_ids,
        num_steps=cfg.num_steps,
        topk=cfg.topk,
        search_width=cfg.search_width,
        num_prompts_start=cfg.num_prompts_start,
        prompt_add_interval=cfg.prompt_add_interval,
    )

    # ---- Results ----
    suffix_str = tokenizer.decode(best_suffix, skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("Universal optimized suffix:")
    print(repr(suffix_str))

    # ---- Test generation on ALL prompts ----
    print("\n" + "=" * 60)
    print("Generation results:")
    print("=" * 60)

    for i, (goal, target) in enumerate(zip(cfg.goals, cfg.targets)):
        tm = TokenManager(tokenizer, goal, target, best_suffix, cfg.device)
        prompt_ids = tm.build_prompt_ids(best_suffix)

        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][prompt_ids.shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n--- Prompt {i+1}: {goal} ---")
        print(f"Target: {target.strip()}")
        print(f"Output: {response[:200]}")

    # ---- Plot ----
    steps, losses = zip(*loss_history)

    plt.figure(figsize=(10, 4))

    plt.plot(steps, losses)

    plt.xlabel("Step")
    plt.ylabel("Mean Loss (across active prompts)")
    plt.title("GCG Optimization — Universal Attack (Algorithm 2)")
    plt.tight_layout()

    plt.savefig(
        "results/plots/loss_universal.png",
        dpi=150,
    )

    plt.show()

    print("\nLoss plot saved to results/plots/loss_universal.png")


if __name__ == "__main__":
    main()