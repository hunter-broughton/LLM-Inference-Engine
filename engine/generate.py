"""Phase 1 — the single-stream generation loop.

This is the heart of Phase 1 and the thing to implement yourself. The shape:

    prefill  : run the whole prompt once, get logits + KV cache
    decode   : loop — sample a token, feed *just that token* back in with the
               cache, repeat until EOS or max_new_tokens

`generate` yields tokens as they're produced so a caller (the benchmark, later
the FastAPI server) can measure time-to-first-token and stream. The structure is
laid out; the four TODO(you) steps are the loop itself.
"""

from __future__ import annotations

from typing import Iterator

import torch

from .model import LoadedModel
from .sampling import SamplingParams, sample


@torch.inference_mode()
def generate(
    model: LoadedModel,
    prompt: str,
    max_new_tokens: int = 128,
    params: SamplingParams | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (excludes the prompt)."""
    params = params or SamplingParams()
    if params.seed is not None:
        torch.manual_seed(params.seed)

    device = model.config.device
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # TODO(you): PREFILL. Run the full prompt through model.forward with
    #   past_key_values=None to get (logits, past_key_values).

    # TODO(you): DECODE LOOP, up to max_new_tokens times:
    #   1. next_id = sample(logits, params)            # [batch]
    #   2. if next_id == model.eos_token_id: break
    #   3. yield int(next_id)
    #   4. logits, past_key_values = model.forward(
    #          next_id.view(1, 1), past_key_values)    # feed ONE token + cache
    #
    # The view(1, 1) matters: after prefill you pass a single new token, not the
    # growing sequence — the cache already holds everything before it.
    raise NotImplementedError("implement the prefill + decode loop")


def generate_text(
    model: LoadedModel,
    prompt: str,
    max_new_tokens: int = 128,
    params: SamplingParams | None = None,
) -> str:
    """Convenience wrapper: collect the generated ids and decode to a string."""
    ids = list(generate(model, prompt, max_new_tokens, params))
    return model.tokenizer.decode(ids)
