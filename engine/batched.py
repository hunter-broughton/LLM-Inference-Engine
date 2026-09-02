"""Phase 2 — batched decode (the throughput engine).

`generate.py` decodes ONE sequence per forward pass. That leaves a GPU almost
idle: at batch size 1 the hardware spends its time streaming weights in from
memory and barely any doing math. Batching many sequences into a SINGLE forward
pass amortizes those weight loads across the batch — the same reason continuous
batching is the biggest throughput win in the project.

This module runs a set of sequences together, one batched step at a time:

    prefill : left-pad the prompts so every sequence's real last token sits at
              index -1, run one batched forward -> a shared KV cache + first logits
    decode  : each step, sample one next token PER ROW, feed the whole [B, 1]
              batch back through the model with the shared cache

Left-padding + an attention mask + explicit position_ids is what lets sequences
of different prompt lengths share one aligned batch (GPT-2 uses absolute position
embeddings, so the position_ids must reflect each row's real positions, not the
padded offsets). This is exactly how HF's own batched generation works, unrolled
so the batching is ours and measurable.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .model import LoadedModel
from .sampling import SamplingParams, sample


@torch.inference_mode()
def generate_batched(
    model: LoadedModel,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    params: SamplingParams | None = None,
) -> list[list[int]]:
    """Decode all `prompts` together. Returns per-prompt generated token ids."""
    params = params or SamplingParams()
    if params.seed is not None:
        torch.manual_seed(params.seed)

    tok = model.tokenizer
    hf = model.model               # the raw HF model (takes attention_mask/position_ids)
    device = model.config.device
    eos = model.eos_token_id

    # LEFT-pad: puts every sequence's real final token at index -1, so one slice
    # [:, -1, :] gives the next-token logits for the whole batch at once.
    old_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True)
    tok.padding_side = old_side

    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    B = input_ids.shape[0]

    # position_ids from the mask: real tokens get 0..n-1, pad slots get 0 (masked).
    pos = (attn.long().cumsum(-1) - 1).clamp_min(0)

    out = hf(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values

    outputs: list[list[int]] = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        next_ids = sample(logits, params)               # [B]
        eos_now = next_ids.eq(eos)
        for i in range(B):
            if finished[i]:
                continue
            if eos_now[i]:
                finished[i] = True                      # stop recording, keep in batch
            else:
                outputs[i].append(int(next_ids[i]))
        if bool(finished.all()):
            break

        # Extend the mask by one real position and feed the whole batch back. We
        # keep finished rows in the batch (harmless) rather than doing cache
        # surgery — evicting them is the scheduler's job (the control plane).
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], dim=1)
        pos_step = attn.long().sum(-1, keepdim=True) - 1   # next absolute position per row
        out = hf(
            input_ids=next_ids[:, None],
            attention_mask=attn,
            position_ids=pos_step,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past = out.past_key_values

    return outputs
