"""Correctness test: rmsnorm CUDA kernel vs a PyTorch reference.

RMSNorm:  y = x / sqrt(mean(x^2) + eps) * weight   (no mean-subtraction).
We spell out the reference by hand rather than calling F.rms_norm so the math
you're implementing in CUDA is staring back at you here.

Run (on a CUDA machine):  pytest kernels/tests/test_rmsnorm.py
"""

import pytest
import torch

from common import CUDA, load_kernel


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_rmsnorm_matches_pytorch():
    ext = load_kernel("rmsnorm")

    D = 768  # GPT-2 hidden size
    eps = 1e-6
    x = torch.randn(64, D, device="cuda")
    weight = torch.randn(D, device="cuda")

    out = ext.rmsnorm(x, weight, eps)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    assert torch.allclose(out, ref, atol=1e-5), "kernel diverges from reference"
