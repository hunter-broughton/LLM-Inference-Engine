# Mini LLM Inference Engine

A from-the-ground-up LLM **inference** engine ("mini vLLM") — hand-written CUDA/Triton
kernels, a paged KV cache, a continuous-batching scheduler, and a FastAPI serving layer
that autoscales on Kubernetes.

> **Status:** Phases 1 & 2 core complete; single-stream generation, paged KV cache,
> continuous-batching scheduler, streaming FastAPI server, and a local Kubernetes +
> KEDA autoscaling demo all working. GPU/CUDA work runs on a free Google Colab T4.
> See [PROJECT.md](PROJECT.md) for the plan and [BENCHMARKS.md](BENCHMARKS.md) for numbers.

## Results

**Phase 2 — batched decode vs PyTorch baseline** (`gpt2`, CPU, greedy, `bench/concurrency.py`):

| Concurrency | PyTorch baseline (HF generate) | This engine (batched) | Speedup |
|---|---|---|---|
| 8 | 102.7 tok/s | 451.8 tok/s | **4.4×** |
| 32 | 106.7 tok/s | 836.5 tok/s | **7.8×** |

Batched greedy is verified token-identical to single-stream greedy (`test_batched.py`),
so the throughput gain changes nothing about the output.

Phase 1 single-stream baseline (`bench/run.py`): **128.5 tok/s**, **13.3 ms** TTFT (CPU).

**Phase 2 — custom Triton attention kernel**: FlashAttention-style fused kernel, verified
against PyTorch, **1.48× faster than naive attention at seq=4096** (Colab T4). Wired into
GPT-2, it delivers **+17% median end-to-end throughput** vs eager attention at long context
(RTX 4090), profiled with Nsight Systems (`notebooks/phase3_end2end_attention_colab.ipynb`, `notebooks/NSIGHT.md`).

**Phase 3 — Kubernetes autoscaling** (local kind + KEDA): scaled **1 → 7 pods** under load
on a custom in-flight-requests metric (`deploy/`).

> CPU for Phases 1–2 throughput, T4 for the kernel; a GPU widens the batching gap further.

## Quick start

```bash
# 1. Generate text (Phase 1)
python bench/run.py --n 5 --max-new 32

# 2. Run the tests (sampling, paged KV cache, continuous-batching scheduler)
pytest engine/tests/ -q

# 3. Serve it with token streaming
uvicorn serving.app:app --port 8000
curl -N -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt":"The future of GPU computing is","max_new_tokens":20}'

# 4. Local Kubernetes + KEDA autoscaling demo (needs Docker + kind + kubectl + helm)
./deploy/demo.sh
```

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

Under load, KEDA reads the server's `/metrics.json` (`inflight` gauge) and scales
the Deployment's pod count — custom-metric autoscaling on real inference load, not
CPU%. See [deploy/README.md](deploy/README.md).

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
