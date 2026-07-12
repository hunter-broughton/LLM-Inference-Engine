"""Microbenchmark: rmsnorm CUDA kernel vs a PyTorch reference.

Like softmax, rmsnorm is memory-bound (one reduction pass + one scale pass over
the row), so report GB/s. This is the kernel you'll FUSE into the model hot path
in Phase 2 — so note the standalone number now; you'll want the before/after
when the fused version lands. Paste it into ../../BENCHMARKS.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch

from common import ARCH, CUDA, cuda_time_ms, load_kernel


def _torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def main() -> None:
    assert CUDA, "this benchmark needs a CUDA GPU"
    ext = load_kernel("rmsnorm")

    rows, D = 8192, 2048
    eps = 1e-6
    x = torch.randn(rows, D, device="cuda")
    w = torch.randn(D, device="cuda")

    mine = cuda_time_ms(lambda: ext.rmsnorm(x, w, eps))
    ref = cuda_time_ms(lambda: _torch_rmsnorm(x, w, eps))

    bytes_moved = 2 * rows * D * 4  # read x, write y (weight reuse is negligible)
    gbps = lambda ms: bytes_moved / (ms / 1e3) / 1e9

    print(f"rmsnorm  arch={ARCH}  ({rows}x{D})")
    print(f"  kernel : {mine:7.3f} ms   {gbps(mine):7.1f} GB/s")
    print(f"  torch  : {ref:7.3f} ms   {gbps(ref):7.1f} GB/s")


if __name__ == "__main__":
    main()
