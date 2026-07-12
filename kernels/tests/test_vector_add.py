"""Correctness test for the vector_add CUDA kernel vs a PyTorch reference.

This is the worked example for the Phase 0 test pattern: compile the .cu with
torch.utils.cpp_extension, run it, and assert torch.allclose against PyTorch.
Copy this shape for matmul / softmax / rmsnorm.

Run (on a CUDA machine):  pytest kernels/tests/test_vector_add.py
"""

import os
import pytest
import torch

CUDA = torch.cuda.is_available()
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "cuda")


def _load():
    from torch.utils.cpp_extension import load
    return load(
        name="vector_add_ext",
        sources=[os.path.join(KERNEL_DIR, "vector_add.cu")],
        verbose=True,
        # Local Pascal card is sm_61; override per machine (e.g. sm_89 on an L4).
        extra_cuda_cflags=["-arch=sm_61"],
    )


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_vector_add_matches_pytorch():
    ext = _load()
    a = torch.randn(1_000_000, device="cuda")
    b = torch.randn(1_000_000, device="cuda")

    out = ext.vector_add(a, b)
    ref = a + b

    assert torch.allclose(out, ref, atol=1e-5), "kernel output diverges from PyTorch"
