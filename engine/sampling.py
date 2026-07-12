"""Phase 1 — token sampling.

Given the logits for the next position, pick a token id. Greedy is provided as
the worked example; temperature / top-k / top-p are yours to implement — they're
small but they're where the "how does sampling actually work" understanding
lives, so they're left as TODO(you).

"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0.0 => greedy (argmax)
    top_k: int | None = None  # keep only the k highest-prob tokens
    top_p: float | None = None  # nucleus: keep the smallest set with cumprob >= p
    seed: int | None = None


def sample(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Return next-token ids, shape [batch].

    `logits` is [batch, vocab] (the last-position logits from model.forward).
    """
    # Greedy is just the argmax — no randomness, ignores temperature/top-k/top-p.
    if params.temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    # TODO(you): temperature scaling.
    #   probs = softmax(logits / temperature). Lower temp => sharper (more
    #   deterministic); higher temp => flatter (more random).
    logits = logits / params.temperature


    # TODO(you): top-k filtering (if params.top_k is not None).
    #   Keep the k largest logits, set the rest to -inf before softmax.
    if params.top_k is not None:
        kth = torch.topk(logits, params.top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    # TODO(you): top-p / nucleus filtering (if params.top_p is not None).
    #   Sort by prob desc, take the cumulative sum, drop the tail past p.
    if params.top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumprobs - sorted_probs > params.top_p
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))


    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
