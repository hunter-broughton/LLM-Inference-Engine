"""Correctness test: tiled matmul CUDA kernel vs torch.matmul (cuBLAS).

Same shape as test_vector_add.py, but using the shared `load_kernel` helper.
Tolerances are looser than vector_add because matmul accumulates K products in
fp32 — rounding adds up, so we allow a small rtol/atol.

Run (on a CUDA machine):  pytest kernels/tests/test_matmul.py
"""

import pytest
import torch

from common import CUDA, load_kernel


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_matmul_matches_pytorch():
    ext = load_kernel("matmul")

    # Non-square and not a multiple of TILE_SIZE on purpose: this catches the
    # boundary bugs (the last partial tile) that square power-of-two shapes hide.
    M, K, N = 200, 320, 168
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    out = ext.matmul(a, b)
    ref = a @ b

    assert out.shape == (M, N), f"expected ({M}, {N}), got {tuple(out.shape)}"
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), "kernel diverges from cuBLAS"
