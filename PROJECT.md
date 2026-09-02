# Mini LLM Inference Engine — Project Brief

A from-the-ground-up LLM **inference** engine ("mini vLLM"), built to learn CUDA
and to produce a portfolio-grade AI-infrastructure project. The work is staged so
that every phase ends in a shippable artifact with real benchmark numbers.

> This document is the source of truth for the project. Claude Code should read it
> first, keep it updated as phases complete, and treat the **Acceptance Criteria**
> in each phase as the definition of done. The `## Current Status` section near the
> top says where we are and what to do next.

---

## Current Status

- **Phase:** 1 & 2 core complete; 3 (serving + local K8s/KEDA) runnable. GPU work
  moved to **Google Colab (free T4)** — the local Pascal PC is no longer available.
- **Done:**
  - **Phase 1** — `sampling.py` (greedy/temp/top-k/top-p), `generate.py`
    (prefill + decode loop), `bench/run.py`. GPT-2 generates coherent text; CPU
    baseline recorded in `BENCHMARKS.md` (128.5 tok/s, 13.3 ms TTFT).
  - **Phase 2 (control plane + batched throughput)** — paged KV cache
    (`kv_cache.py`), continuous-batching scheduler (`scheduler.py`), and batched
    decode (`batched.py`); 10 CPU tests passing incl. batched==single-stream
    correctness. **Measured 4.40× (C=8) / 7.84× (C=32) aggregate throughput** over
    the HF `generate()` baseline (`bench/concurrency.py`, `bench/baseline_hf.py`).
  - **Phase 2 (fused kernel)** — FlashAttention-style Triton kernel
    (`kernels/triton/fused_attention.py`), correctness-gated vs PyTorch (~2e-3),
    **1.48× faster than naive attention at seq=4096** (Colab T4). Wired into GPT-2
    (eager→Triton): **+17.4% median end-to-end throughput** at long context
    (batch 16, 960-tok prompt) on an RTX 4090, correctness-gated, profiled with
    **Nsight Systems** (`nsys`). Fixed a decode-loop autotune bug (key on N_KV
    re-tuned every token, ~14× slowdown) → single fixed config.
  - **Phase 0 (first kernel)** — `vector_add.cu` implemented; run it on a free T4
    via `notebooks/phase0_vector_add_colab.ipynb`.
  - **Phase 3** — FastAPI server with SSE streaming + `/metrics(.json)`; CPU
    Dockerfile; local **kind + KEDA** autoscaling on the `inflight` custom metric;
    **demonstrated 1 → 7 pods under load** (`deploy/demo.sh`, `loadtest.sh`, `teardown.sh`).
- **Next task (optional depth):** softmax/matmul CUDA kernels; wire the Triton
  attention into GPT-2 for an end-to-end tok/s number; re-run Phase 1/2 benches on
  the T4 for GPU headline figures.
- Update this section at the end of every working session.

---

## Goal & Non-Goals

**Goal:** Serve a small language model efficiently on a single GPU, with as much of
the hot path as possible understood and (where it earns its place) hand-written.
Demonstrate measurable throughput/latency wins backed by profiling, then run it as
a real service on Kubernetes.

**This is an inference project, not a training project.** Inference is the right
target: it runs on one GPU, it's cheap, and serving throughput/latency is exactly
what infrastructure teams hire for.

**Non-goals:**

- Training or fine-tuning models (out of scope).
- Beating production vLLM on absolute throughput (not the point; the point is
  understanding, measurable improvement over our own baseline, and a clean systems story).
- Supporting many model architectures. One or two small models, done well.

---

## Builder Context (informs technical decisions)

- Strong **Kubernetes / GitOps / operator** experience (Azure Arc-enabled
  Kubernetes). The Phase 3 serving layer should lean into this — it's the
  differentiator. Prefer GitOps-friendly tooling (e.g. KEDA) and consider a small
  operator/CRD as a stretch goal.
- Distributed-systems background (Paxos-based KV store in Go). Go is a reasonable
  choice for the serving control plane if we want it; the engine itself is Python.
- ML coursework (CNNs/ViTs); comfortable with PyTorch.
- **Hardware:** local dev GPU is a Pascal-generation card (~2017–2018, likely
  GTX 1060/1070/1080), compute capability **sm_61**. See `## GPU & Environment Notes`
  — this has real consequences for kernel work.

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Engine / orchestration | **Python + PyTorch** | Model loading, KV cache, scheduler |
| Raw kernels | **CUDA C++** | For fundamentals + credibility; `sm_61` locally |
| Fused kernels | **Triton** | For the production-style fused path; modern, productive |
| Python ↔ CUDA bridge | `torch.utils.cpp_extension` (`load_inline` / `load`) | Simplest; pybind11 if we outgrow it |
| Serving | **FastAPI** | HTTP + token streaming |
| Container | Docker (CUDA runtime base image) | |
| Orchestration | **Kubernetes** (kind/k3s locally) | GPU scheduling, autoscaling |
| Autoscaling | **KEDA** (or HPA on a custom metric) | Scale on queue depth / GPU util |
| Profiling | **Nsight Compute (`ncu`)**, **Nsight Systems (`nsys`)**, `torch.profiler` | |

**Models:** start with **GPT-2 (124M)** for fast iteration, then **TinyLlama-1.1B**
for a modern architecture (RoPE, RMSNorm, SwiGLU, GQA). Weights via Hugging Face.

---

## Repo Structure

```
.
├── PROJECT.md            # this file (source of truth)
├── README.md             # leads with the benchmark numbers, then architecture
├── BENCHMARKS.md         # running log of measured results per phase
├── kernels/              # raw CUDA + Triton kernels
│   ├── cuda/             # .cu source
│   ├── triton/           # .py Triton kernels
│   ├── tests/            # correctness vs PyTorch reference (allclose)
│   └── bench/            # per-kernel microbenchmarks
├── engine/               # model loading, KV cache, scheduler/batching
├── serving/              # FastAPI server, request lifecycle, streaming
├── deploy/               # Dockerfile, k8s manifests, KEDA/HPA config
├── bench/                # end-to-end throughput/latency harness
└── notebooks/            # profiling explorations (optional)
```

---

## Phases

Each phase is a stopping point: shippable on its own, with a number to show for it.

### Phase 0 — CUDA Fundamentals

*Learn the GPU memory model by writing kernels from scratch.*

Implement, in this order: `vector_add` → tiled `matmul` (shared memory) →
row-wise numerically-stable `softmax` → `rmsnorm`.

For **each** kernel: a correctness test against a PyTorch reference (`torch.allclose`
within tolerance), a microbenchmark vs the PyTorch/cuBLAS equivalent, and a one-paragraph
profiling note from `ncu` (is it memory-bound or compute-bound? occupancy? what's the limiter?).

**Acceptance criteria:**

- All four kernels pass correctness tests.
- A benchmark table exists for each in `BENCHMARKS.md`.
- For each kernel, we can state its bottleneck and why, citing Nsight output.

### Phase 1 — Baseline Inference

*Get a working, measured baseline to improve on.*

Load GPT-2 (124M) and implement a generation loop in plain PyTorch (greedy + temperature
sampling). Build a benchmark harness measuring **tokens/sec**, **TTFT** (time to first
token), and **p50/p99 latency** for single-stream generation.

**Acceptance criteria:**

- Generates coherent text from a prompt.
- Baseline numbers checked into `BENCHMARKS.md` (this is the bar Phase 2 beats).

### Phase 2 — Real Engine

*The part that stops it being a toy.*

- **Paged KV cache** (block-based allocation, not one contiguous tensor per request).
- **Continuous batching** scheduler: requests arrive and finish at different times;
  the batch is rebuilt each step rather than padded to a fixed shape.
- Replace **at least one hot path** with a custom kernel — fused RMSNorm or fused
  attention — in raw CUDA and/or Triton, wired into the engine via `cpp_extension`.
- Benchmark **throughput under concurrency** (1 / 8 / 32 / 64 concurrent requests)
  against the Phase 1 baseline.

**Acceptance criteria:**

- Measurable throughput improvement over Phase 1, documented in `BENCHMARKS.md`.
- Profiling evidence (Nsight) showing where the win came from.
- KV cache + continuous batching covered by tests.

### Phase 3 — Serving on Kubernetes (the differentiator)

*Where the K8s/operator experience turns this into something rare.*

- **FastAPI** server wrapping the engine, with streaming token responses.
- **Dockerize** (CUDA runtime base image); run on local **kind/k3s** with GPU access.
- **Autoscaling** via KEDA (or HPA) on a custom metric — queue depth or GPU utilization.
- **Stretch goals:** prefill/decode disaggregation across pods; a small operator/CRD
  to declaratively manage model deployments (GitOps-style).

**Acceptance criteria:**

- One-command deploy to a local cluster.
- Demonstrably scales pods under load.
- README shows the architecture diagram **and** the numbers.

---

## GPU & Environment Notes

- The local card is **Pascal (sm_61): no tensor cores.** It is great for learning
  CUDA fundamentals and for FP32 kernel work, but it **cannot** do FP16/BF16
  tensor-core kernels or hit modern throughput figures.
- **Strategy:** develop and learn locally; for FP16/tensor-core kernels and the
  *final headline benchmarks*, rent a modern GPU by the hour (an **L4** `sm_89` or
  **A10/A100** `sm_86/sm_80`) on Lambda / RunPod / Vast — a few dollars per hour.
  Don't let the old card cap the numbers that go on the resume.
- Compile kernels for the target arch explicitly (`-arch=sm_61` locally;
  re-build for `sm_89`/`sm_80` when benchmarking on cloud GPUs).
- First session: capture `nvidia-smi` and `nvcc --version` output into the README so
  the environment is documented.

---

## How Claude Code Should Work on This

- **Benchmark-driven.** No kernel or engine change merges without a before/after
  measurement. Numbers live in `BENCHMARKS.md`.
- **Test kernels against a reference.** Every custom kernel gets a correctness test
  comparing it to the PyTorch equivalent within tolerance.
- **Profile, don't guess.** Use `ncu`/`nsys` to identify bottlenecks before optimizing.
- Keep commits small and scoped to one change.
- Update `## Current Status` at the end of each session.
- The **README leads with results** (the throughput/latency table), then the
  architecture, then setup. Recruiters and engineers read the top first.

---

## Resources

- *Programming Massively Parallel Processors* (PMPP) — CUDA fundamentals.
- Karpathy's `llm.c` and `nanoGPT` — clean reference implementations to study.
- vLLM internals — read for the paged-attention and continuous-batching design.
- Triton tutorials (official) — fused-kernel patterns.
- GPU MODE community / kernel leaderboards — for momentum and benchmarking culture.
