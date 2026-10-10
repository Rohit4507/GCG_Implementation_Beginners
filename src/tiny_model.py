"""
Tiny, randomly-initialized model + tokenizer for offline smoke tests.

Everything is built locally (no Hugging Face downloads):
  - a small BPE tokenizer trained on the experiment text itself,
  - a 2-layer GPT-2-style causal LM with random weights.

The point is NOT to jailbreak anything (random weights cannot be); it is
to exercise every code path — chat templating, tokenization slices,
gradients, KV-cache, candidate search, evaluation, defenses — in seconds
on CPU so regressions are caught immediately.
"""
import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _corpus(goals, targets):
    texts = list(goals) + list(targets)
    # Guarantee character coverage for any text we may tokenize later.
    texts.append(string.printable)
    texts.append(
        "Sure, here is please help me with my request write explain "
        "describe tell about the a an of to in and or not"
    )
    return texts


def build_tiny_model_and_tokenizer(
    goals=None,
    targets=None,
    vocab_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
):
    goals = goals or []
    targets = targets or []

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<bos>"],
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_corpus(goals, targets), trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    tokenizer.chat_template = _CHAT_TEMPLATE

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_ctx=256,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config).to(device).eval()
    return model, tokenizer
