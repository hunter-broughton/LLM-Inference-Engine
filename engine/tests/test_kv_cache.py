"""Tests for the paged KV cache (Phase 2).

These don't need a GPU for the block-table *bookkeeping* — the allocator and
block-table math are pure Python and that's exactly where the bugs hide. The
tensor pools allocate on whatever device KVCacheConfig picks, so run on CPU if
no CUDA: override config.device="cpu" in a fixture if needed.

Run:  pytest engine/tests/test_kv_cache.py
"""

import pytest

from engine.kv_cache import BlockAllocator


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
    # TODO(you): decide the contract — does allocate(3) raise, or return []?
    # Assert whatever you chose here so the scheduler can rely on it.


# TODO(you): once PagedKVCache.append_tokens works, add a test that:
#   - adds a sequence, appends block_size+1 tokens, and asserts the block table
#     grew to exactly 2 blocks (the boundary case)
#   - frees the sequence and asserts the blocks returned to the pool
