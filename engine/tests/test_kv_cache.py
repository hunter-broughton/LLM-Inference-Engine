"""Tests for the paged KV cache (Phase 2).

These don't need a GPU for the block-table *bookkeeping* — the allocator and
block-table math are pure Python and that's exactly where the bugs hide. The
tensor pools allocate on whatever device KVCacheConfig picks, so run on CPU if
no CUDA: override config.device="cpu" in a fixture if needed.

Run:  pytest engine/tests/test_kv_cache.py
"""

import pytest
import torch

from engine.kv_cache import BlockAllocator, KVCacheConfig, PagedKVCache


def test_allocator_hands_out_and_reclaims_blocks():
    alloc = BlockAllocator(num_blocks=4)
    assert alloc.can_allocate(4)

    got = alloc.allocate(3)
    assert len(got) == 3
    assert not alloc.can_allocate(2)  # only 1 left

    alloc.free_blocks(got)
    assert alloc.can_allocate(4)      # all back


def test_allocator_refuses_oversubscription():
    alloc = BlockAllocator(num_blocks=2)
    assert not alloc.can_allocate(3)
    # Contract: allocate() fails loud rather than half-filling. The scheduler
    # must gate on can_allocate() first; reaching allocate() short is a bug.
    with pytest.raises(RuntimeError):
        alloc.allocate(3)


def _tiny_cache(block_size=16, num_blocks=8) -> PagedKVCache:
    # 1 layer / 1 head / small head_dim on CPU: we only exercise the block-table
    # bookkeeping here, not the attention math, so the pool can be tiny.
    cfg = KVCacheConfig(
        num_layers=1, num_kv_heads=1, head_dim=8,
        block_size=block_size, num_blocks=num_blocks,
        device="cpu", dtype=torch.float32,
    )
    return PagedKVCache(cfg)


def test_append_grows_block_table_on_boundary():
    cache = _tiny_cache(block_size=16)
    cache.add_sequence(0)

    cache.append_tokens(0, 16)                     # exactly fills one block
    assert len(cache.sequences[0].block_table) == 1

    cache.append_tokens(0, 1)                       # spills into a second block
    assert len(cache.sequences[0].block_table) == 2
    assert cache.sequences[0].length == 17


def test_free_sequence_returns_blocks_to_pool():
    cache = _tiny_cache(block_size=16, num_blocks=8)
    cache.add_sequence(0)
    cache.append_tokens(0, 40)                       # ceil(40/16) = 3 blocks
    assert cache.allocator.can_allocate(5)           # 8 - 3 left

    cache.free_sequence(0)
    assert cache.allocator.can_allocate(8)           # all back
    assert 0 not in cache.sequences
