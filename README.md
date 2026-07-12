# Mini LLM Inference Engine

A from-the-ground-up LLM **inference** engine ("mini vLLM") — hand-written CUDA/Triton
kernels, a paged KV cache, a continuous-batching scheduler, and a FastAPI serving layer
that autoscales on Kubernetes.

> **Status:** Phase 0 (CUDA fundamentals) — in progress. See [PROJECT.md](PROJECT.md)
> for the full plan and [BENCHMARKS.md](BENCHMARKS.md) for measured results.

## Results

> Headline throughput/latency numbers go here as phases complete. Recruiters and
> engineers read the top first — this table is the lead. Placeholder until Phase 1.

| Metric | Baseline (PyTorch) | This engine | Hardware |
|---|---|---|---|
| Throughput (tok/s, 32 concurrent) | _TBD_ | _TBD_ | _TBD_ |
| TTFT (p50) | _TBD_ | _TBD_ | _TBD_ |
| Latency (p99) | _TBD_ | _TBD_ | _TBD_ |

## Architecture

```
client ──HTTP/stream──► FastAPI (serving/)
                            │
                            ▼
                     scheduler (engine/)  ── continuous batching
                            │
                            ▼
                  model + paged KV cache (engine/)
                            │
                            ▼
                custom kernels (kernels/: CUDA + Triton)
```

_Architecture diagram and detail to be expanded in Phase 3._

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

CUDA work targets an NVIDIA GPU (local dev card is Pascal / `sm_61`; headline
benchmarks are run on a rented modern GPU — see [PROJECT.md](PROJECT.md#gpu--environment-notes)).

<!-- TODO(phase0): on the CUDA dev machine, paste `nvidia-smi` and `nvcc --version`
output here so the environment is documented. -->

```
$ nvidia-smi
(TODO: capture on GPU machine)

$ nvcc --version
(TODO: capture on GPU machine)
```

## Repo layout

| Path | Purpose |
|---|---|
| `kernels/cuda/` | Raw CUDA C++ kernels (`.cu`) |
| `kernels/triton/` | Triton fused kernels |
| `kernels/tests/` | Correctness tests vs PyTorch reference |
| `kernels/bench/` | Per-kernel microbenchmarks |
| `engine/` | Model loading, paged KV cache, scheduler |
| `serving/` | FastAPI server, streaming |
| `deploy/` | Dockerfile, k8s manifests, KEDA/HPA |
| `bench/` | End-to-end throughput/latency harness |

## License

MIT
