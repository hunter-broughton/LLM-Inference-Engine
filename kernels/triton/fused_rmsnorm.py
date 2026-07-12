"""Phase 2 — fused RMSNorm in Triton.

This is the "replace at least one hot path with a custom kernel" deliverable, the
Triton flavor. In PyTorch, RMSNorm is several ops (square, mean, rsqrt, mul, mul)
— each launches a kernel and round-trips the activations through HBM. Fusing them
into ONE kernel means the row is read once, normalized in registers/SRAM, and
written once. That's the win, and Nsight should show the dropped memory traffic.

Triton model: you write the per-program (per-row here) body in Python; Triton
compiles it. One program instance handles one row of length N.

Reference to match: y = x * rsqrt(mean(x^2) + eps) * weight
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,          # [n_rows, N] input
    w_ptr,          # [N] weight
    y_ptr,          # [n_rows, N] output
    stride_row,     # elements between rows of x/y
    N,              # row length (hidden dim)
    eps,
    BLOCK_SIZE: tl.constexpr,  # >= N, power of two; one row fits in one program
):
    row = tl.program_id(0)
    # TODO(you):
    #   offs = tl.arange(0, BLOCK_SIZE); mask = offs < N
    #   x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0)
    #   ms = tl.sum(x * x, axis=0) / N
    #   x_norm = x * tl.rsqrt(ms + eps)
    #   w = tl.load(w_ptr + offs, mask=mask)
    #   tl.store(y_ptr + row * stride_row + offs, x_norm * w, mask=mask)
    pass


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in for the PyTorch RMSNorm reference, backed by the Triton kernel."""
    assert x.is_cuda and x.dim() == 2, "expects a 2D [rows, N] CUDA tensor"
    n_rows, N = x.shape
    y = torch.empty_like(x)
    # TODO(you): pick BLOCK_SIZE = triton.next_power_of_2(N), launch with a 1D
    # grid of n_rows programs, passing x.stride(0) as stride_row.
    raise NotImplementedError("launch _rmsnorm_kernel")
