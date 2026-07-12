"""Phase 1 — model loading.

Loads a small causal LM (GPT-2 124M to start) via HuggingFace transformers and
exposes a thin wrapper the generation loop calls. We lean on HF's own forward +
`past_key_values` (HF's built-in KV cache) for the baseline. In Phase 2 we
replace that cache with our own paged implementation (kv_cache.py) — keeping this
wrapper small is what makes that swap clean.

This file is mostly boilerplate on purpose; the learning is in generate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    name: str = "gpt2"  # HF hub id; later: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32  # fp16 only pays off on a tensor-core GPU


class LoadedModel:
    """Holds the HF model + tokenizer and the few facts the loop needs."""

    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.eos_token_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, past_key_values=None):
        """One forward pass.

        Returns (logits_for_last_position, past_key_values). On the first call
        pass the full prompt and past_key_values=None (this is *prefill*); on
        every later call pass just the single new token plus the returned cache
        (this is *decode*). Skipping recomputation of the prompt each step is the
        whole reason inference is tractable.
        """
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # logits: [batch, seq, vocab]; we only ever sample from the last position.
        return out.logits[:, -1, :], out.past_key_values


def load_model(config: ModelConfig | None = None) -> LoadedModel:
    """Download (cached) and load the model + tokenizer onto the target device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = config or ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.name)
    model = AutoModelForCausalLM.from_pretrained(config.name, torch_dtype=config.dtype)
    model.to(config.device).eval()

    # GPT-2 ships without a pad token; reuse EOS so batched padding has something.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return LoadedModel(model, tokenizer, config)
