"""Phase 1 — the single-stream generation loop.

This is the heart of Phase 1. The shape:

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

    # PREFILL. Run the whole prompt once with no cache. This both (a) fills
    # past_key_values with the K/V vectors for every prompt token and (b) gives
    # us the logits for the first token we're about to generate.
    logits, past_key_values = model.forward(input_ids, past_key_values=None)

    # DECODE LOOP. One token per iteration, reusing the cache so the model only
    # ever does work for the single newest token (not the whole growing sequence).
    for _ in range(max_new_tokens):
        next_id = sample(logits, params)          # [batch] == [1]
        if int(next_id) == model.eos_token_id:    # model says "done" — stop
            break                                 # before yielding the EOS token
        yield int(next_id)
        # Feed just that one token back in, WITH the cache. view(1, 1) reshapes
        # [1] -> [batch=1, seq=1]; the cache already holds everything before it.
        logits, past_key_values = model.forward(next_id.view(1, 1), past_key_values)


def generate_text(
    model: LoadedModel,
    prompt: str,
    max_new_tokens: int = 128,
    params: SamplingParams | None = None,
) -> str:
    """Convenience wrapper: collect the generated ids and decode to a string."""
    ids = list(generate(model, prompt, max_new_tokens, params))
    return model.tokenizer.decode(ids)
