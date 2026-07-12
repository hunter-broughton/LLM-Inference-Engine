"""Phase 3 — FastAPI server with token streaming.

Wraps the engine in an HTTP API. Two design notes that matter for the K8s story:

  - /generate STREAMS tokens (Server-Sent Events) instead of blocking until the
    full response — that's what makes TTFT a real, user-visible metric and what
    a load test will hammer.
  - /metrics exposes queue depth and running-batch size in Prometheus format.
    This is the signal KEDA scales on (see deploy/keda-scaledobject.yaml): scale
    out when requests are backing up, not just on raw CPU. That custom-metric
    autoscaling is the differentiator the project is built around.

The endpoint *shapes* are here; wiring them to the Scheduler is the TODO(you).
Phase 1 can run a simpler version that just calls engine.generate directly —
start there, then move to the scheduler for concurrency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

# Process-wide handles, populated on startup. The model is big and loads once.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO(you): on startup, load the model and build the scheduler, e.g.
    #   from engine import load_model, ModelConfig
    #   STATE["model"] = load_model(ModelConfig(name="gpt2"))
    #   STATE["scheduler"] = Scheduler(STATE["model"], PagedKVCache(...))
    #   ...and kick off a background task that drains scheduler.step().
    yield
    STATE.clear()


app = FastAPI(title="mini-llm-inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    """Liveness/readiness probe target for k8s."""
    return "ok" if STATE.get("model") is not None else "loading"


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens as Server-Sent Events (`data: <token>\\n\\n`)."""

    async def token_stream():
        # TODO(you): submit the request to the scheduler and yield tokens as the
        # background loop produces them. Minimal Phase-1 version, no scheduler:
        #   from engine.generate import generate
        #   from engine.sampling import SamplingParams
        #   for tok_id in generate(STATE["model"], req.prompt, req.max_new_tokens,
        #                          SamplingParams(temperature=req.temperature)):
        #       text = STATE["model"].tokenizer.decode([tok_id])
        #       yield f"data: {text}\n\n"
        raise NotImplementedError("wire /generate to the engine")
        yield  # unreachable; keeps this an async generator until implemented

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-format metrics. KEDA scales on `inference_queue_depth`."""
    sched = STATE.get("scheduler")
    queue_depth = len(sched.waiting) if sched else 0
    running = len(sched.running) if sched else 0
    return (
        "# HELP inference_queue_depth Requests waiting for admission\n"
        "# TYPE inference_queue_depth gauge\n"
        f"inference_queue_depth {queue_depth}\n"
        "# HELP inference_running_batch Requests currently decoding\n"
        "# TYPE inference_running_batch gauge\n"
        f"inference_running_batch {running}\n"
    )
