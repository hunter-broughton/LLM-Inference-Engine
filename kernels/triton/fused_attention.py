"""Phase 2 — fused attention in Triton (FlashAttention-style).

The harder, higher-value fused kernel. Standard attention materializes the full
[seq, seq] scores matrix in HBM — O(seq^2) memory traffic and the dominant cost
at long context. FlashAttention avoids ever writing that matrix: it walks the K/V
in blocks, keeping a running max and running sum (the *online softmax*) plus the
running output accumulator in SRAM, so memory traffic drops to O(seq).

This is genuinely the hardest kernel in the project. Suggested path:
  1. Get a NON-fused, non-paged version correct first (it's allowed to be slow).
  2. Add the online-softmax running-stats trick (the FlashAttention core).
  3. Only then wire it to the paged KV cache (read K/V via the block table).

Reference to match (non-causal here for clarity; you'll want causal masking):
    scores = (q @ k.T) / sqrt(head_dim)
    out    = softmax(scores, dim=-1) @ v
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, head_dim,
    scale,                      # 1 / sqrt(head_dim)
    BLOCK_M: tl.constexpr,      # query block size
    BLOCK_N: tl.constexpr,      # key/value block size
):
    # TODO(you): the FlashAttention inner loop.
    #   - each program handles a block of BLOCK_M queries
    #   - init running max m = -inf, running sum l = 0, accumulator acc = 0
    #   - loop over key/value blocks of BLOCK_N:
    #       s = scale * (q @ k_block.T)            # [BLOCK_M, BLOCK_N]
    #       (apply causal mask here if causal)
    #       m_new = max(m, rowmax(s))
    #       p = exp(s - m_new)
    #       correction = exp(m - m_new)
    #       l = l * correction + rowsum(p)
    #       acc = acc * correction + p @ v_block
    #       m = m_new
    #   - out = acc / l ; store it
    pass


def fused_attention(
    q: torch.Tensor,  # [seq, head_dim]
    k: torch.Tensor,  # [seq, head_dim]
    v: torch.Tensor,  # [seq, head_dim]
    causal: bool = True,
) -> torch.Tensor:
    """Single-head attention via the Triton kernel. Batch/heads/paging come later."""
    assert q.is_cuda and q.shape == k.shape == v.shape
    seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    scale = head_dim ** -0.5
    # TODO(you): choose BLOCK_M/BLOCK_N, build the grid, launch _attention_kernel.
    raise NotImplementedError("launch _attention_kernel")
