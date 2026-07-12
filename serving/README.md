# Serving

FastAPI layer (Phase 3) that wraps the engine in a streaming HTTP API.

## Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/generate` | POST | Stream tokens (SSE) for a prompt |
| `/healthz` | GET | k8s liveness/readiness probe |
| `/metrics` | GET | Prometheus metrics; `inference_queue_depth` drives KEDA |

## Run locally

```bash
uvicorn serving.app:app --reload --port 8000
# then:
curl -N -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "Hello", "max_new_tokens": 32}'
```

`-N` disables curl buffering so you see tokens stream in.

## Build order

Start simple: have `/generate` call `engine.generate` directly (Phase 1). Once
the Phase 2 scheduler exists, move to submitting requests to it and draining
`step()` in a background task — that's what gives you real concurrency, and it's
what makes the `/metrics` queue depth meaningful for autoscaling (`../deploy/`).
