"""Microbenchmark: vector_add CUDA kernel vs torch (a + b).

WORKED EXAMPLE of the Phase 0 bench pattern — copy this shape for matmul /
softmax / rmsnorm. Each bench: time your kernel, time the PyTorch/cuBLAS
equivalent, and report the metric that actually characterizes the kernel.

vector_add does 3 memory ops per element (read a, read b, write c) and one cheap
add, so it is *memory-bound*: the number that matters is achieved GB/s, not
GFLOP/s. Compare it to your card's peak memory bandwidth — that ratio is the
real result. Paste it into ../../BENCHMARKS.md.

Run (on a CUDA machine):  python kernels/bench/bench_vector_add.py
"""

import os
import sys

# Benches live in bench/ but reuse the helpers next door in tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch

from common import ARCH, CUDA, cuda_time_ms, load_kernel


def main() -> None:
    assert CUDA, "this benchmark needs a CUDA GPU"
    ext = load_kernel("vector_add")

    n = 1 << 24  # ~16.7M floats
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    mine = cuda_time_ms(lambda: ext.vector_add(a, b))
    ref = cuda_time_ms(lambda: a + b)

    bytes_moved = 3 * n * 4  # read a, read b, write c; 4 bytes/float
    gbps = lambda ms: bytes_moved / (ms / 1e3) / 1e9

    print(f"vector_add  arch={ARCH}  n={n:,}")
    print(f"  kernel : {mine:7.3f} ms   {gbps(mine):7.1f} GB/s")
    print(f"  torch  : {ref:7.3f} ms   {gbps(ref):7.1f} GB/s")


if __name__ == "__main__":
    main()
