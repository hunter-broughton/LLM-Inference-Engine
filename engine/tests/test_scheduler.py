"""Tests for the continuous-batching scheduler (Phase 2).

The behaviors worth pinning down (and the ones that break subtly):
  - admission respects max_batch and KV-cache capacity (backpressure, no drops)
  - a finished request is evicted and its KV blocks are freed
  - requests of different lengths coexist (the whole point of continuous batching)

The forward pass runs fine on CPU with gpt2, so these run without a GPU — just
slower. We load the model once per module and use a tiny KV pool.

Run:  pytest engine/tests/test_scheduler.py
"""

import pytest
import torch

from engine.kv_cache import KVCacheConfig, PagedKVCache
from engine.model import ModelConfig, load_model
from engine.sampling import SamplingParams
from engine.scheduler import Request, Scheduler, Status


@pytest.fixture(scope="module")
def model():
    return load_model(ModelConfig(name="gpt2", device="cpu"))


def _kv_cache(num_blocks: int = 64) -> PagedKVCache:
    # gpt2 geometry (12 layers / 12 heads / head_dim 64). In this baseline the
    # pool is accounting-only, so we keep num_blocks small to stay light.
    cfg = KVCacheConfig(
        num_layers=12, num_kv_heads=12, head_dim=64,
        block_size=16, num_blocks=num_blocks, device="cpu", dtype=torch.float32,
    )
    return PagedKVCache(cfg)


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


def test_admission_respects_max_batch(model):
    sched = Scheduler(model, _kv_cache(), max_batch=2)
    for i in range(3):
        sched.add_request(_make_request(i))

    sched.step()  # admits up to max_batch, then decodes one token each
    assert len(sched.running) == 2   # exactly max_batch admitted
    assert len(sched.waiting) == 1   # the third is held (backpressure, not dropped)


def test_finished_request_frees_its_blocks(model):
    kv = _kv_cache()
    free_before = len(kv.allocator.free)
    sched = Scheduler(model, kv, max_batch=4)
    sched.add_request(_make_request(0, max_new=5))

    finished = []
    while sched.has_work():
        finished += sched.step()

    assert len(finished) == 1
    assert 1 <= len(finished[0].output_ids) <= 5          # capped by max_new
    assert len(kv.allocator.free) == free_before          # all KV blocks returned
    assert not sched.running and not sched.waiting


def test_mixed_length_requests_coexist(model):
    # The defining property of continuous batching: a short and a long request
    # share the running batch, and the short one finishes and leaves without
    # blocking the long one.
    sched = Scheduler(model, _kv_cache(), max_batch=4)
    sched.add_request(_make_request(0, max_new=2))
    sched.add_request(_make_request(1, max_new=10))

    sched.step()                       # both admitted + one decode step
    assert len(sched.running) == 2

    done_ids = []
    while sched.has_work():
        done_ids += [r.req_id for r in sched.step()]
    assert done_ids == [0, 1]          # short finishes first, long follows
