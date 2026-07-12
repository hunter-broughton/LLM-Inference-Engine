"""Shared helpers for kernel tests and benchmarks.

Keeps the two fiddly things — the cpp_extension build config (the *arch flag*
matters, see PROJECT.md) and correct CUDA-event timing — in one place so each
test/bench stays short and obvious.

`test_vector_add.py` is deliberately the standalone, no-helper version of the
same load pattern: read that first to see what `load_kernel` is doing under the
hood, then use this helper everywhere else.
"""

import os

import torch

CUDA = torch.cuda.is_available()
CUDA_DIR = os.path.join(os.path.dirname(__file__), "..", "cuda")

# Compile target. Local Pascal dev card is sm_61; override per machine WITHOUT
# editing files, e.g.  KERNEL_ARCH=sm_89 python kernels/bench/bench_matmul.py
ARCH = os.environ.get("KERNEL_ARCH", "sm_61")


def load_kernel(name: str):
    """JIT-compile and load kernels/cuda/<name>.cu as a Python module.

    The module exposes whatever you bound in that file's PYBIND11_MODULE block
    (e.g. ext.matmul). Compilation happens once per process and is cached on disk
    by torch, so repeated runs are fast.
    """
    from torch.utils.cpp_extension import load

    return load(
        name=f"{name}_ext",
        sources=[os.path.join(CUDA_DIR, f"{name}.cu")],
        extra_cuda_cflags=[f"-arch={ARCH}"],
        verbose=True,
    )


def cuda_time_ms(fn, *, warmup: int = 10, iters: int = 100) -> float:
    """Median run time of a no-arg callable, in milliseconds.

    Why CUDA events and not time.perf_counter(): kernel launches are async, so
    wall-clock around a launch measures almost nothing. Events record on the GPU
    stream and we synchronize before reading them.

    Why warmup: first launches pay JIT/cache/clock-spin-up costs that aren't
    representative. Why median: shrugs off the occasional scheduler hiccup.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    times.sort()
    return times[len(times) // 2]
