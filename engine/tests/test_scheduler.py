"""Tests for the continuous-batching scheduler (Phase 2).

The behaviors worth pinning down (and the ones that break subtly):
  - admission respects max_batch and KV-cache capacity (backpressure, no drops)
  - a finished request is evicted and its KV blocks are freed
  - requests of different lengths coexist (the whole point of continuous batching)

These need a model + KV cache, so they're heavier. Use the smallest model and a
tiny KV pool. Marked to skip without CUDA since the forward pass needs the GPU.

Run:  pytest engine/tests/test_scheduler.py
"""

import pytest
import torch

from engine.sampling import SamplingParams
from engine.scheduler import Request, Scheduler, Status

CUDA = torch.cuda.is_available()


def _make_request(req_id: int, n_prompt: int = 4, max_new: int = 8) -> Request:
    return Request(
        req_id=req_id,
        prompt_ids=list(range(n_prompt)),
        params=SamplingParams(temperature=0.0),
        max_new_tokens=max_new,
    )


def test_request_starts_waiting():
    req = _make_request(0)
    assert req.status is Status.WAITING
    assert not req.done


@pytest.mark.skipif(not CUDA, reason="scheduler.step needs the model forward pass")
def test_admission_respects_max_batch():
    pytest.skip("TODO(you): build a Scheduler with max_batch=2, add 3 requests, "
                "call step() once, assert exactly 2 are RUNNING and 1 WAITING")


@pytest.mark.skipif(not CUDA, reason="scheduler.step needs the model forward pass")
def test_finished_request_frees_its_blocks():
    pytest.skip("TODO(you): run a short request to completion; assert it lands in "
                "step()'s returned list and its KV blocks return to the pool")
