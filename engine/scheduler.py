"""Phase 2 — continuous-batching scheduler.

THE IDEA. Static batching waits for a full batch, pads everyone to the same
length, runs them in lockstep, and can't admit a new request until the whole
batch finishes — so a short request stuck behind a long one waits, and the GPU
runs padding. Continuous batching instead rebuilds the batch *every decode step*:
finished sequences leave, waiting ones join (subject to KV-cache space), and no
two sequences need the same length. This is the single biggest throughput win in
the project and the reason concurrency scales.

The control loop (`step`) is the learning core. Request lifecycle, queues, and
admission structure are laid out; the TODO(you) parts are the policy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from .kv_cache import PagedKVCache
from .model import LoadedModel
from .sampling import SamplingParams, sample


class Status(Enum):
    WAITING = auto()   # admitted to the engine, not yet running
    RUNNING = auto()   # in the active batch, decoding
    FINISHED = auto()  # hit EOS or max_new_tokens


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    params: SamplingParams
    max_new_tokens: int
    status: Status = Status.WAITING
    output_ids: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status is Status.FINISHED


class Scheduler:
    """Owns the request queues and drives the per-step batch.

    Typical wiring (the serving layer calls add_request from request handlers and
    runs step() in a background loop):

        sched = Scheduler(model, kv_cache, max_batch=32)
        sched.add_request(Request(...))
        while sched.has_work():
            finished = sched.step()   # one decode step across the live batch
    """

    def __init__(
        self,
        model: LoadedModel,
        kv_cache: PagedKVCache,
        max_batch: int = 32,
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.max_batch = max_batch
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self._ids = itertools.count()

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _admit(self) -> None:
        """Promote WAITING -> RUNNING while there's batch slack and KV space.

        This is the admission policy. Keep it simple first (FCFS up to
        max_batch, only if the KV cache can fit the prompt), then make it
        smarter if profiling says to.
        """
        # TODO(you): while len(self.running) < self.max_batch and self.waiting:
        #   - peek the next waiting request
        #   - check kv_cache has room for its prompt (allocator.can_allocate)
        #   - if yes: add_sequence + append_tokens(prompt), mark RUNNING, move it
        #   - if no: stop (leave it waiting — that's backpressure, not a drop)
        raise NotImplementedError

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run ONE decode step over the running batch. Return finished requests.

        Returns the requests that completed *this* step so the caller can stream
        their final tokens out and release resources.
        """
        self._admit()
        if not self.running:
            return []

        # TODO(you): the core continuous-batching step.
        #   1. Build this step's input: the last token of each running request
        #      (prompts were already prefilled at admission).
        #   2. Forward the batch through the model + paged KV cache, getting
        #      per-sequence next-token logits.
        #   3. For each running request: sample (via `sample`), append the token,
        #      grow its KV (kv_cache.append_tokens), and check the stop condition
        #      (EOS or len(output_ids) >= max_new_tokens) -> Status.FINISHED.
        #   4. Evict finished requests from self.running, free their KV blocks,
        #      and collect them to return.
        raise NotImplementedError
