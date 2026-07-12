"""Correctness test: row-wise softmax CUDA kernel vs torch.softmax(x, dim=-1).

The thing to get right is numerical stability: subtract the row max before exp.
The large-magnitude row below would overflow exp() if you skipped that, so it
doubles as a stability check, not just a correctness check.

Run (on a CUDA machine):  pytest kernels/tests/test_softmax.py
"""

import pytest
import torch

from common import CUDA, load_kernel


@pytest.mark.skipif(not CUDA, reason="requires a CUDA GPU")
def test_softmax_matches_pytorch():
    ext = load_kernel("softmax")

    x = torch.randn(128, 1024, device="cuda")
    x[0] += 50.0  # push one row's magnitude up to exercise the max-subtraction

    out = ext.softmax(x)
    ref = torch.softmax(x, dim=-1)

    assert torch.allclose(out, ref, atol=1e-5), "kernel diverges from torch.softmax"
    # Each row must sum to 1 — a cheap, kernel-independent sanity check.
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "rows don't sum to 1"
