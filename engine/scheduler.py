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

    # Runtime state, filled at admission (prefill) and advanced each step. Kept
    # per-request because each sequence carries its own HF KV cache and its own
    # "logits to sample next" — that independence is what lets sequences of
    # different lengths share one running batch (the point of continuous batching).
    past_key_values: object = None
    last_logits: object = None  # [1, vocab] — what step() samples from next

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
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting[0]                       # FCFS: peek, don't pop yet
            n = len(req.prompt_ids)

            # Backpressure gate: only admit if the KV pool can hold the prompt.
            # If not, STOP (don't skip ahead) — the queue stays ordered and the
            # waiting request is pressure on upstream, never a dropped request.
            need = self.kv_cache.blocks_needed(0, n)
            if not self.kv_cache.allocator.can_allocate(need):
                break

            self.waiting.pop(0)
            self.kv_cache.add_sequence(req.req_id)
            self.kv_cache.append_tokens(req.req_id, n)   # reserve prompt blocks

            # PREFILL this request now, so step() only ever does single-token
            # decodes over the running batch (same prefill/decode split as
            # generate.py, just once per admitted request).
            ids = torch.tensor([req.prompt_ids], device=self.model.config.device)
            req.last_logits, req.past_key_values = self.model.forward(ids, None)
            req.status = Status.RUNNING
            self.running.append(req)

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run ONE decode step over the running batch. Return finished requests.

        Returns the requests that completed *this* step so the caller can stream
        their final tokens out and release resources.
        """
        self._admit()
        if not self.running:
            return []

        # One decode step across the whole live batch. Each running request
        # samples its next token from the logits it carries (from prefill on its
        # first step, or from the previous step after that), same ordering as
        # generate.py: sample -> stop-check -> advance.
        for req in self.running:
            next_id = sample(req.last_logits, req.params)
            tok = int(next_id)

            # Stop conditions come BEFORE advancing: EOS means the model is done
            # (don't emit the sentinel), and hitting the token budget caps it.
            if tok == self.model.eos_token_id:
                req.status = Status.FINISHED
                continue
            req.output_ids.append(tok)
            if len(req.output_ids) >= req.max_new_tokens:
                req.status = Status.FINISHED
                continue

            # Advance: feed the one new token back with this request's own cache,
            # and grow its KV accounting by one token (allocates a fresh block
            # only when the current one spills over — see blocks_needed).
            req.last_logits, req.past_key_values = self.model.forward(
                next_id.view(1, 1), req.past_key_values
            )
            self.kv_cache.append_tokens(req.req_id, 1)

        # Evict finished requests: free their KV blocks back to the pool (so a
        # waiting request can be admitted next step) and hand them to the caller.
        finished = [r for r in self.running if r.done]
        for req in finished:
            self.kv_cache.free_sequence(req.req_id)
        self.running = [r for r in self.running if not r.done]
        return finished
