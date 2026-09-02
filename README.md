# Mini LLM Inference Engine

A from-the-ground-up LLM **inference** engine — a "mini [vLLM](https://github.com/vllm-project/vllm)" —
built to understand (and hand-write) the parts of a modern serving stack that make language-model
inference fast: token sampling, a KV cache, batched autoregressive decoding, a **paged KV cache**,
a **continuous-batching scheduler**, custom **CUDA/Triton kernels**, a streaming **FastAPI** server,
and **Kubernetes autoscaling** with KEDA.

It runs [GPT-2 (124M)](https://huggingface.co/gpt2) end to end. Everything except the CUDA/Triton
kernels runs on a laptop CPU; the kernels run on any NVIDIA GPU (a free Google Colab T4 is enough).

> **Why this project?** Serving throughput and latency are exactly what AI-infrastructure teams
> hire for, and they come from a handful of specific techniques. This repo implements those
> techniques from scratch, with a correctness test and a measured before/after for each one — so
> every performance claim below is reproducible, not asserted.

---

## Table of contents

- [Results](#results)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running each piece](#running-each-piece)
- [Benchmarks](#benchmarks)
- [Where things run (hardware)](#where-things-run-hardware)
- [Design notes & honest limitations](#design-notes--honest-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Results

All numbers are reproducible with the commands in [Running each piece](#running-each-piece); the
full log lives in [BENCHMARKS.md](BENCHMARKS.md).

**Throughput under concurrency — the engine vs. a stock PyTorch baseline** (`gpt2`, greedy, CPU):

| Concurrent requests | PyTorch baseline (HF `generate()`) | This engine (batched decode) | Speedup |
|---|---|---|---|
| 8  | 102.7 tok/s | 451.8 tok/s | **4.4×** |
| 32 | 106.7 tok/s | 836.5 tok/s | **7.8×** |

Batched decode is verified **token-identical** to single-stream greedy, so the speedup changes
nothing about the output. Single-stream baseline: **128.5 tok/s**, **13.3 ms** time-to-first-token.

**Custom Triton attention kernel** (FlashAttention-style, fp16):

- **1.48× faster** than a naive materialized-scores attention at `seq=4096` (Colab T4), verified
  against PyTorch.
- Wired into GPT-2 in place of the eager attention path, it delivers **+17% median end-to-end
  throughput** at long context (batch 16, 960-token prompt) on an RTX 4090 — correctness-gated and
  profiled with **Nsight Systems**.

**Kubernetes autoscaling** (local `kind` cluster + KEDA): under load the server's request queue
builds, and KEDA scales the deployment **1 → 6 pods** on a custom `queue_depth` metric, then back
down when it drains.

---

## Architecture

```
                       ┌──────────────────────────────────────────────┐
   HTTP / SSE stream   │  serving/  — FastAPI                          │
  ───────────────────► │   POST /generate   (streams tokens as SSE)    │
                       │   GET  /metrics.json (queue_depth, running)   │◄─── KEDA polls this
                       └───────────────┬──────────────────────────────┘        │
                                       │ submit request                        │ scales pods
                                       ▼                                        ▼
                       ┌──────────────────────────────────────────────┐   Kubernetes + KEDA
                       │  engine/scheduler.py — continuous batching    │   (deploy/)
                       │   admit (FCFS, KV-gated) → decode step → evict│
                       └───────────────┬──────────────────────────────┘
                                       │ one batched step
                        ┌──────────────┴───────────────┐
                        ▼                               ▼
        ┌───────────────────────────┐   ┌──────────────────────────────┐
        │ engine/model.py (GPT-2)   │   │ engine/kv_cache.py            │
        │ engine/generate.py        │   │  paged KV cache (block table) │
        │ engine/batched.py         │   │  block allocator (free list)  │
        │ engine/sampling.py        │   └──────────────────────────────┘
        └───────────────┬───────────┘
                        ▼
        ┌───────────────────────────────────────────────┐
        │ kernels/ — hand-written CUDA + Triton          │
        │  cuda: vector_add, tiled matmul, rmsnorm       │
        │  triton: fused attention, fused rmsnorm        │
        └───────────────────────────────────────────────┘
```

The request path: a client POSTs to `/generate`; the server submits the request to the
**scheduler**, which admits it into a running batch (bounded by `max_batch`, so excess requests
**queue**), runs the model + KV cache one decode step at a time, and streams each new token back as
a Server-Sent Event. `/metrics.json` exposes the live `queue_depth`, which **KEDA** reads to scale
pods — autoscaling on real inference backlog rather than CPU%.

---

## How it works

Each subsystem is small on purpose, with a test and a benchmark. In build order:

1. **Sampling** (`engine/sampling.py`) — turns a row of logits into a token id: greedy (argmax),
   temperature scaling, top-k, and top-p / nucleus filtering (the `-inf` mask → softmax idiom).

2. **Generation loop** (`engine/generate.py`) — the autoregressive core: a **prefill** pass over
   the whole prompt to build the KV cache and get the first logits, then a **decode** loop that
   feeds one token back at a time *with the cache* (so the model never re-reads the prompt — the
   whole reason inference is tractable). Yields tokens so callers can measure time-to-first-token.

3. **Paged KV cache** (`engine/kv_cache.py`) — vLLM's PagedAttention idea: instead of one big
   contiguous KV tensor per request, cut KV memory into fixed-size **blocks** handed out from a free
   list, with a per-sequence **block table** mapping logical positions to physical blocks. Sequences
   grow a block at a time and finished ones return their blocks instantly — which is what lets many
   requests share one pool at high concurrency.

4. **Continuous-batching scheduler** (`engine/scheduler.py`) — rebuilds the batch every step:
   finished sequences leave, waiting ones join (subject to KV space), and no two sequences need the
   same length. Admission is FCFS with backpressure (a request that doesn't fit **waits**, it's
   never dropped). This is the biggest throughput lever in the project.

5. **Batched decode** (`engine/batched.py`) — runs many sequences through **one** forward pass via
   left-padding + an attention mask + explicit `position_ids`, amortizing weight loads across the
   batch. This is what delivers the 4.4× / 7.8× throughput numbers, and it's verified
   token-identical to single-stream.

6. **Custom kernels** (`kernels/`) — hand-written CUDA (`vector_add`, tiled shared-memory `matmul`,
   block-per-row `rmsnorm`) and Triton (a FlashAttention-style fused attention that never
   materializes the `[seq, seq]` score matrix, and a fused RMSNorm). Each has a correctness test
   against a PyTorch reference. The fused attention is wired into GPT-2 for the end-to-end throughput
   lift.

7. **Serving** (`serving/app.py`) — a FastAPI server that submits requests to the scheduler from a
   background loop and streams tokens as SSE, exposing `queue_depth` for autoscaling.

8. **Deploy** (`deploy/`) — a CPU Docker image, Kubernetes manifests, and a KEDA `ScaledObject`
   that autoscales on `queue_depth`, plus one-command scripts to run the whole thing on a local
   `kind` cluster and watch it scale.

---

## Repository layout

| Path | What's in it |
|---|---|
| `engine/sampling.py` | Greedy / temperature / top-k / top-p sampling |
| `engine/model.py` | Loads GPT-2 (HuggingFace), thin forward wrapper |
| `engine/generate.py` | Single-stream prefill + decode loop |
| `engine/batched.py` | Batched decode (the throughput path) |
| `engine/kv_cache.py` | Paged KV cache: block allocator + block tables |
| `engine/scheduler.py` | Continuous-batching scheduler (admit / step / evict) |
| `engine/tests/` | CPU tests for sampling, KV cache, scheduler, batched decode |
| `kernels/cuda/` | CUDA kernels: `vector_add`, `matmul`, `rmsnorm` |
| `kernels/triton/` | Triton kernels: `fused_attention`, `fused_rmsnorm` |
| `kernels/tests/`, `kernels/bench/` | Kernel correctness tests + microbenchmarks |
| `serving/app.py` | FastAPI streaming server + metrics |
| `deploy/` | Dockerfile, k8s manifests, KEDA config, `demo.sh` / `loadtest.sh` / `teardown.sh` |
| `bench/` | `run.py` (single-stream), `concurrency.py` + `baseline_hf.py` (throughput) |
| `notebooks/` | Colab notebooks for the GPU kernels + `NSIGHT.md` profiling guide |
| `PROJECT.md` | The full project plan / spec |
| `BENCHMARKS.md` | Running log of every measured result |

---

## Getting started

**Requirements:** Python 3.11+ (works with 3.12/3.13). A GPU is **not** needed for the engine,
tests, server, or K8s demo — only for the CUDA/Triton kernels.

```bash
git clone <this-repo>
cd "Inference Engine Project"

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the GPT-2 weights (~500 MB) from HuggingFace and caches them.

---

## Running each piece

### 1. Generate text / single-stream benchmark

```bash
python bench/run.py --n 5 --max-new 32
# prints tokens/sec, time-to-first-token, p50/p99 latency
```

### 2. Run the tests (no GPU needed)

```bash
pytest engine/tests/ -q          # sampling, paged KV cache, scheduler, batched decode
```

### 3. Throughput: the engine vs. the PyTorch baseline

```bash
python bench/baseline_hf.py                              # the HF generate() reference
python bench/concurrency.py --concurrency 1,8,32 --max-new 64   # engine vs baseline, per concurrency
```

### 4. Serve it with token streaming

```bash
uvicorn serving.app:app --port 8000
# in another terminal:
curl -N -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"The future of GPU computing is","max_new_tokens":20}'
curl -s localhost:8000/metrics.json     # {"queue_depth": N, "running_batch": M}
```

`MAX_BATCH` (env var, default 4) sets how many requests decode at once before the rest queue.

### 5. Kubernetes + KEDA autoscaling demo (local, free)

Needs Docker running, plus `kind`, `kubectl`, and `helm` (`brew install kind kubectl helm`).

```bash
./deploy/demo.sh                 # builds the image, starts a kind cluster, installs KEDA, applies manifests
# then, in two more terminals:
kubectl get pods -w
kubectl port-forward svc/inference-server 8000:80
./deploy/loadtest.sh             # drives concurrent load; watch pods scale 1 -> 6
./deploy/teardown.sh             # delete the cluster when done
```

### 6. CUDA / Triton kernels (GPU — free Google Colab T4)

The kernels need an NVIDIA GPU. Open a notebook in [Google Colab](https://colab.research.google.com/)
(Runtime → Change runtime type → **T4 GPU**) and Run all:

- `notebooks/phase0_vector_add_colab.ipynb` — your first CUDA kernel
- `notebooks/phase0_kernels_colab.ipynb` — tiled matmul + RMSNorm (CUDA & Triton), correctness + timing
- `notebooks/phase2_fused_attention_colab.ipynb` — the FlashAttention-style Triton kernel vs naive & SDPA
- `notebooks/phase3_end2end_attention_colab.ipynb` — the kernel wired into GPT-2, end-to-end tok/s lift

Each notebook **gates on correctness** (asserts against PyTorch) before reporting any speedup.

### 7. Nsight profiling (rented GPU)

`ncu` needs GPU performance-counter access that free Colab blocks. To profile with Nsight, use a
rented GPU with root (Lambda / RunPod, ~\$0.30–0.75/hr) and follow
[`notebooks/NSIGHT.md`](notebooks/NSIGHT.md) — it drives `notebooks/profile_attention.py` under
`nsys`/`ncu`. This is how the +17% end-to-end number and the kernel trace were captured.

---

## Benchmarks

Full, dated log with hardware and reproduce-commands: **[BENCHMARKS.md](BENCHMARKS.md)**. Ground rule
of the project: *no kernel or engine change is recorded without a before/after measurement.*

---

## Where things run (hardware)

| Part | Where it runs | Notes |
|---|---|---|
| Engine, tests, benchmarks, server, K8s demo | **Any CPU** (laptop) | No GPU required |
| CUDA / Triton kernels | **NVIDIA GPU** | Free Google Colab T4 is enough |
| Kernel op benchmark (1.48×) | Colab **T4** (`sm_75`) | fp16 |
| End-to-end +17% lift + Nsight trace | Rented **RTX 4090** | via RunPod, ~\$0.30 for the run |

The engine uses HuggingFace's built-in KV cache for the actual K/V tensors; the paged KV cache runs
the **admission / capacity control plane** (block accounting, backpressure). Fusing the paged blocks
directly into the attention math is the next kernel step (see [Roadmap](#roadmap)).

---

## Design notes & honest limitations

- **The throughput win is batching.** At batch size 1 a GPU (or CPU) spends most of its time
  streaming weights; batching many sequences through one forward pass amortizes that. The 4.4× is
  measured against the *sequential* PyTorch baseline — the standard vLLM-style comparison.
- **The custom attention kernel is educational, not production-tuned.** It beats a naive
  materialized-scores attention (1.48× at long context), but it is still ~12× off PyTorch's own
  fused SDPA — expected for a hand-written kernel without warp specialization / register blocking.
- **The end-to-end +17% holds at long context** (long prompt, batched — the prefill-dominated
  regime where attention is a meaningful slice of the work). Short prompts see little or no lift;
  the honest claim is "up to ~17% at long context," with the config stated.
- **KEDA scales on `queue_depth` read from one pod** via the Service (the demo's one moving part).
  The production-correct version scrapes every pod with Prometheus and scales on the fleet sum —
  the `/metrics` endpoint already emits Prometheus format for exactly that upgrade.
- **Model scope is intentionally small:** GPT-2 (124M) for fast iteration. TinyLlama (RoPE, RMSNorm,
  GQA) is the planned next model.

---

## Roadmap

- Fuse the paged KV blocks into a custom paged-attention kernel (so the cache drives real attention).
- Add TinyLlama-1.1B (RoPE / RMSNorm / GQA) as a second, modern architecture.
- Finish the Phase 0 kernel set (softmax) and add Nsight bottleneck notes per kernel.
- Prometheus-based fleet-aggregate autoscaling; optional scale-to-zero.

---

## License

MIT
