"""Phase 2 — paged KV cache.

THE IDEA (from vLLM's PagedAttention). A naive KV cache reserves one big
contiguous tensor per request, sized to max_seq_len. That wastes memory (most
requests are shorter) and fragments the GPU. Instead we cut KV memory into
fixed-size *blocks* and hand them out on demand, like OS virtual memory:

  - a pool of physical blocks, each holding `block_size` tokens of K and V
  - per sequence, a *block table*: the list of physical blocks it owns, in order
  - logical position t lives in block_table[t // block_size] at slot
    t % block_size

This lets sequences grow a block at a time and lets the engine pack many
sequences into one pool — which is what makes high concurrency possible.

The allocator and block-table mechanics are the learning core: implement the
TODO(you) methods. Shapes/structure are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16        # tokens per block; powers of two are convenient
    num_blocks: int = 4096      # total physical blocks in the pool (memory budget)
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class BlockAllocator:
    """Hands out and reclaims physical block ids from a free list."""

    def __init__(self, num_blocks: int):
        # Simple stack of free block ids. Pop to allocate, push to free.
        self.free: list[int] = list(range(num_blocks))

    def can_allocate(self, n: int) -> bool:
        return len(self.free) >= n

    def allocate(self, n: int) -> list[int]:
        # TODO(you): pop n ids off the free list and return them. Raise (or
        # signal the scheduler) if not enough are free — that's backpressure.
        raise NotImplementedError

    def free_blocks(self, block_ids: list[int]) -> None:
        # TODO(you): return these ids to the free list.
        raise NotImplementedError


@dataclass
class SequenceCache:
    """Per-sequence view: which physical blocks it owns and how full it is."""

    block_table: list[int] = field(default_factory=list)
    length: int = 0  # number of tokens currently cached


class PagedKVCache:
    """Owns the physical K/V pools and the per-sequence block tables.

    Memory layout (one tensor each for K and V, shared by all sequences):
        [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
    A (layer, head, token) lookup goes through the sequence's block table to a
    physical block, then indexes the slot within it.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.allocator = BlockAllocator(config.num_blocks)
        self.sequences: dict[int, SequenceCache] = {}

        shape = (
            config.num_blocks,
            config.num_layers,
            config.num_kv_heads,
            config.block_size,
            config.head_dim,
        )
        self.k_pool = torch.empty(shape, dtype=config.dtype, device=config.device)
        self.v_pool = torch.empty(shape, dtype=config.dtype, device=config.device)

    def add_sequence(self, seq_id: int) -> None:
        self.sequences[seq_id] = SequenceCache()

    def blocks_needed(self, current_len: int, extra_tokens: int) -> int:
        """How many *new* blocks to hold `extra_tokens` more, given current_len."""
        # TODO(you): use ceil division on block boundaries. Hint: compare
        # ceil((current_len + extra_tokens) / block_size) to the blocks already
        # implied by current_len.
        raise NotImplementedError

    def append_tokens(self, seq_id: int, num_tokens: int) -> None:
        """Reserve room for `num_tokens` more tokens of this sequence.

        Grows the block table by allocating blocks when the last one fills up.
        (Writing the actual K/V values happens in the attention kernel, which is
        handed the block table; this method only manages the *mapping*.)
        """
        # TODO(you): compute blocks_needed, allocate them, extend block_table,
        # and bump SequenceCache.length by num_tokens.
        raise NotImplementedError

    def free_sequence(self, seq_id: int) -> None:
        """Return a finished sequence's blocks to the pool."""
        # TODO(you): free the block_table via the allocator and drop the entry.
        raise NotImplementedError
