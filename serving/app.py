"""Phase 3 — FastAPI server with token streaming, backed by the scheduler.

Requests are submitted to the continuous-batching Scheduler, not run inline. A
background engine thread drives `scheduler.step()` and pushes each new token to
the request's queue; the /generate handler streams those tokens out as SSE. Because
the scheduler admits only `max_batch` requests at once, everything else genuinely
WAITS — so `queue_depth` (len(scheduler.waiting)) is a real backlog signal, which is
exactly what KEDA scales on (see deploy/keda-scaledobject.yaml).

  - /generate   streams tokens (Server-Sent Events) as the engine produces them.
  - /metrics(.json) expose queue_depth + running-batch size for KEDA / Prometheus.
"""

from __future__ import annotations

import itertools
import os
import queue
import threading
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

STATE: dict = {}
_DONE = object()  # sentinel pushed to a request's queue when it finishes


def _engine_loop(app_state: dict) -> None:
    """Background thread: drive the scheduler and fan tokens out to per-req queues."""
    sched = app_state["scheduler"]
    sinks: dict[int, queue.Queue] = app_state["sinks"]
    emitted: dict[int, int] = app_state["emitted"]
    stop: threading.Event = app_state["stop"]

    while not stop.is_set():
        if not sched.has_work():
            stop.wait(0.005)
            continue
        finished = sched.step()
        # Stream any newly-produced tokens for every live request.
        for req in sched.running + finished:
            sink = sinks.get(req.req_id)
            if sink is None:
                continue
            while emitted.get(req.req_id, 0) < len(req.output_ids):
                idx = emitted.get(req.req_id, 0)
                sink.put(req.output_ids[idx])
                emitted[req.req_id] = idx + 1
        for req in finished:
            sink = sinks.get(req.req_id)
            if sink is not None:
                sink.put(_DONE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from engine import ModelConfig, load_model
    from engine.kv_cache import KVCacheConfig, PagedKVCache
    from engine.scheduler import Scheduler

    model = load_model(ModelConfig(name=os.environ.get("MODEL_NAME", "gpt2")))
    kv = PagedKVCache(KVCacheConfig(
        num_layers=12, num_kv_heads=12, head_dim=64,
        block_size=16, num_blocks=512, device=model.config.device,
        dtype=model.config.dtype,
    ))
    # Small max_batch so concurrent load overflows into the WAITING queue -> the
    # queue_depth signal KEDA scales on actually moves under load.
    max_batch = int(os.environ.get("MAX_BATCH", "4"))
    STATE["model"] = model
    STATE["scheduler"] = Scheduler(model, kv, max_batch=max_batch)
    STATE["sinks"] = {}
    STATE["emitted"] = {}
    STATE["ids"] = itertools.count()
    STATE["stop"] = threading.Event()
    STATE["thread"] = threading.Thread(target=_engine_loop, args=(STATE,), daemon=True)
    STATE["thread"].start()
    yield
    STATE["stop"].set()
    STATE.clear()


app = FastAPI(title="mini-llm-inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok" if STATE.get("model") is not None else "loading"


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest) -> StreamingResponse:
    """Submit to the scheduler; stream tokens as the engine produces them."""
    from engine.sampling import SamplingParams
    from engine.scheduler import Request

    model = STATE["model"]
    sched = STATE["scheduler"]
    req_id = next(STATE["ids"])
    sink: queue.Queue = queue.Queue()
    STATE["sinks"][req_id] = sink
    STATE["emitted"][req_id] = 0

    prompt_ids = model.tokenizer(req.prompt).input_ids
    sched.add_request(Request(
        req_id=req_id,
        prompt_ids=prompt_ids,
        params=SamplingParams(temperature=req.temperature),
        max_new_tokens=req.max_new_tokens,
    ))

    async def token_stream():
        try:
            while True:
                # Block for the next token in a worker thread so the event loop
                # stays free to admit and stream other requests.
                item = await anyio.to_thread.run_sync(sink.get)
                if item is _DONE:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {model.tokenizer.decode([item])}\n\n"
        finally:
            STATE["sinks"].pop(req_id, None)
            STATE["emitted"].pop(req_id, None)

    return StreamingResponse(token_stream(), media_type="text/event-stream")


def _snapshot() -> tuple[int, int]:
    sched = STATE.get("scheduler")
    return (len(sched.waiting), len(sched.running)) if sched else (0, 0)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-format metrics. KEDA scales on `inference_queue_depth`."""
    queue_depth, running = _snapshot()
    return (
        "# HELP inference_queue_depth Requests waiting for admission\n"
        "# TYPE inference_queue_depth gauge\n"
        f"inference_queue_depth {queue_depth}\n"
        "# HELP inference_running_batch Requests currently decoding\n"
        "# TYPE inference_running_batch gauge\n"
        f"inference_running_batch {running}\n"
    )


@app.get("/metrics.json")
async def metrics_json() -> JSONResponse:
    """Same signals as JSON, for KEDA's metrics-api scaler (scales on queue_depth)."""
    queue_depth, running = _snapshot()
    return JSONResponse({"queue_depth": queue_depth, "running_batch": running})
