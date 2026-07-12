"""Microbenchmark: tiled matmul CUDA kernel vs torch.matmul (cuBLAS).

matmul is the opposite of vector_add: it does O(M*N*K) FLOPs over O(M*K + K*N)
bytes, so for decent shapes it is *compute-bound*. The number that matters here
is GFLOP/s. Don't be discouraged that cuBLAS is far ahead — it uses register
tiling and (on newer cards) tensor cores. Your goal is to understand the gap,
not close it. Paste your number into ../../BENCHMARKS.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch

from common import ARCH, CUDA, cuda_time_ms, load_kernel


def main() -> None:
    assert CUDA, "this benchmark needs a CUDA GPU"
    ext = load_kernel("matmul")

    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    mine = cuda_time_ms(lambda: ext.matmul(a, b))
    ref = cuda_time_ms(lambda: a @ b)

    flops = 2.0 * M * N * K  # one multiply + one add per inner-product term
    gflops = lambda ms: flops / (ms / 1e3) / 1e9

    print(f"matmul  arch={ARCH}  ({M}x{K}) @ ({K}x{N})")
    print(f"  kernel : {mine:7.3f} ms   {gflops(mine):8.1f} GFLOP/s")
    print(f"  cuBLAS : {ref:7.3f} ms   {gflops(ref):8.1f} GFLOP/s")


if __name__ == "__main__":
    main()
