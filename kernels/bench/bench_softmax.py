"""Microbenchmark: row-wise softmax CUDA kernel vs torch.softmax.

softmax reads the row, writes the row, and does a couple of cheap passes (max,
exp+sum, divide). It's memory-bound, so report GB/s. The interesting failure
mode to watch in profiling: if you used one block per row with too few threads,
you'll be latency-bound on the reductions, not bandwidth-bound — Nsight will say
so. Paste your number into ../../BENCHMARKS.md.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch

from common import ARCH, CUDA, cuda_time_ms, load_kernel


def main() -> None:
    assert CUDA, "this benchmark needs a CUDA GPU"
    ext = load_kernel("softmax")

    rows, cols = 8192, 2048
    x = torch.randn(rows, cols, device="cuda")

    mine = cuda_time_ms(lambda: ext.softmax(x))
    ref = cuda_time_ms(lambda: torch.softmax(x, dim=-1))

    bytes_moved = 2 * rows * cols * 4  # read x, write y
    gbps = lambda ms: bytes_moved / (ms / 1e3) / 1e9

    print(f"softmax  arch={ARCH}  ({rows}x{cols})")
    print(f"  kernel : {mine:7.3f} ms   {gbps(mine):7.1f} GB/s")
    print(f"  torch  : {ref:7.3f} ms   {gbps(ref):7.1f} GB/s")


if __name__ == "__main__":
    main()
