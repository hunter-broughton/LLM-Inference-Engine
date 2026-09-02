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
        # Contract: raise if we can't satisfy the request in full. The scheduler
        # is expected to call can_allocate() first, so reaching here without room
        # is a bug, not backpressure — fail loud rather than half-allocate.
        if not self.can_allocate(n):
            raise RuntimeError(f"out of KV blocks: need {n}, have {len(self.free)}")
        taken, self.free = self.free[:n], self.free[n:]
        return taken

    def free_blocks(self, block_ids: list[int]) -> None:
        # Return ids to the pool. Order doesn't matter — any free block is
        # interchangeable, since a sequence's *logical* order lives in its
        # block_table, not in the physical layout of the pool.
        self.free.extend(block_ids)


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
        # Blocks are allocated in whole units, so this is pure ceil-division on
        # block boundaries: how many blocks does the sequence occupy AFTER the
        # append, minus how many it already occupies now. The subtraction is what
        # makes growth cheap — a partly-filled last block absorbs new tokens for
        # free until it spills over into a fresh block.
        bs = self.config.block_size
        have = (current_len + bs - 1) // bs                       # ceil(current/bs)
        need = (current_len + extra_tokens + bs - 1) // bs        # ceil(total/bs)
        return need - have

    def append_tokens(self, seq_id: int, num_tokens: int) -> None:
        """Reserve room for `num_tokens` more tokens of this sequence.

        Grows the block table by allocating blocks when the last one fills up.
        (Writing the actual K/V values happens in the attention kernel, which is
        handed the block table; this method only manages the *mapping*.)
        """
        seq = self.sequences[seq_id]
        new_blocks = self.blocks_needed(seq.length, num_tokens)
        if new_blocks:
            seq.block_table.extend(self.allocator.allocate(new_blocks))
        seq.length += num_tokens

    def free_sequence(self, seq_id: int) -> None:
        """Return a finished sequence's blocks to the pool."""
        seq = self.sequences.pop(seq_id)
        self.allocator.free_blocks(seq.block_table)
