# Benchmarks

Running log of measured results. **No kernel or engine change merges without a
before/after measurement here.** Record hardware, arch flag, and date with every entry.

---

## Phase 0 — Kernel microbenchmarks

Each kernel: correctness (`torch.allclose`) + microbench vs PyTorch/cuBLAS + a
one-paragraph Nsight note on the bottleneck.

### vector_add

| Impl | Size | Time | Bandwidth (GB/s) | Hardware | Arch |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Nsight note:** _TBD (memory-bound vs compute-bound? occupancy? limiter?)_

### matmul (tiled, shared memory)

Correctness-verified vs cuBLAS (max abs diff 5e-5). Colab T4, 2026-09-01.

| Impl | Shape | Time | vs cuBLAS | Hardware | Arch |
|---|---|---|---|---|---|
| tiled (shared mem, TILE=16) | 200×320×168 | 0.083 ms | — | T4 | sm_75 |
| cuBLAS (torch.matmul) | 200×320×168 | 0.038 ms | 1.0× (ceiling) | T4 | sm_75 |

**Note:** hand-written tiled kernel lands ~2.2× off cuBLAS on this small shape —
the expected gap for an educational kernel without register-blocking / vectorized
loads / double-buffering.

### softmax (row-wise, numerically stable)

| Impl | Shape | Time | Bandwidth (GB/s) | Hardware | Arch |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Nsight note:** _TBD_

### rmsnorm

Correctness-verified vs PyTorch reference. Colab T4, 2026-09-01.

| Impl | Shape | Time | vs unfused | Hardware | Arch |
|---|---|---|---|---|---|
| CUDA (block/row + tree reduce) | 64×768 | correct (1.9e-6) | — | T4 | sm_75 |
| fused Triton | 64×768 | 9.2 µs | **4.2× faster** | T4 | sm_75 |
| PyTorch (unfused, several ops) | 64×768 | 38.7 µs | 1.0× | T4 | sm_75 |

**Note:** fusing square→mean→rsqrt→mul→mul into one kernel (read the row once,
write once) is a **4.2×** speedup over PyTorch's multi-op sequence — the fusion win.

---

## Phase 1 — Baseline inference (GPT-2 124M)

Single-stream generation, greedy + temperature sampling. This is the bar Phase 2 beats.

CPU baseline (`gpt2`, fp32, greedy, `--n 5 --max-new 32`, `bench/run.py`):

| Metric | Value | Hardware | Date |
|---|---|---|---|
| Tokens/sec | 128.5 | Apple Silicon CPU (fp32) | 2026-07-18 |
| TTFT | 13.3 ms | Apple Silicon CPU (fp32) | 2026-07-18 |
| Latency p50 | 262.8 ms | Apple Silicon CPU (fp32) | 2026-07-18 |
| Latency p99 | 263.0 ms | Apple Silicon CPU (fp32) | 2026-07-18 |

> Re-run on a Colab T4 (fp16) for headline numbers — expect a large jump. Command:
> `python bench/run.py --n 20 --max-new 128`.

---

## Phase 2 — Real engine (paged KV cache + continuous batching + custom kernel)

Aggregate throughput under concurrency: engine batched decode (`engine/batched.py`,
driven by the continuous-batching scheduler + paged KV cache) vs the stock PyTorch
baseline (HF `model.generate()` run sequentially). Reproduce:
`python bench/concurrency.py --concurrency 1,8,32 --max-new 64`.
Correctness gate: batched greedy == single-stream greedy (`test_batched.py`).

**gpt2, CPU (Apple Silicon), fp32, greedy, max_new=64. 2026-08-31:**

| Concurrency | Baseline tok/s | Engine tok/s | Speedup |
|---|---|---|---|
| 1 | 98.6 | 99.1 | 1.00× |
| 8 | 102.7 | 451.8 | **4.40×** |
| 32 | 106.7 | 836.5 | **7.84×** |

> The win is batching: the baseline serves requests one at a time (batch-1, memory-
> bound), the engine runs them in one batched forward, amortizing weight loads. On a
> GPU the gap is typically larger (a GPU at batch-1 is even more underutilized).
> **Re-run on a Colab T4 for the headline numbers.**

### Fused attention kernel (Triton, FlashAttention-style)

Custom Triton kernel (online-softmax, autotuned) vs a naive fp16 materialized-scores
attention. Correctness-gated vs PyTorch fp32 ref + SDPA (max diff ~2e-3). Causal,
B=1 H=8 D=64, fp16, **Colab T4, 2026-09-01** (`notebooks/phase2_fused_attention_colab.ipynb`).

| seq | naive fp16 | Triton fused | speedup vs naive | (SDPA, ceiling) |
|---|---|---|---|---|
| 512 | 0.379 ms | 0.430 ms | 0.88× (loss) | 0.053 ms |
| 1024 | 1.439 ms | 1.172 ms | 1.23× | 0.106 ms |
| 2048 | 5.809 ms | 3.985 ms | 1.46× | 0.345 ms |
| 4096 | 22.258 ms | 15.073 ms | **1.48×** | 1.293 ms |

> The fused kernel avoids materializing the [seq, seq] score matrix, so its edge
> grows with context (loss at 512, 1.48× at 4096). It stays ~12× off PyTorch SDPA
> (a heavily hand-tuned production flash kernel) — expected for a from-scratch
> educational kernel; beating the naive memory-bound path is the honest claim.

**Where the win came from (profiler):** naive shows separate matmul + softmax +
elementwise CUDA ops (materialized-scores HBM traffic); the fused path collapses
them into one `_attention_kernel` launch (torch.profiler, seq=2048).

### End-to-end throughput: fused Triton attention in GPT-2 vs eager

Fused Triton attention monkeypatched into GPT-2's `scaled_dot_product_attention` vs
the eager (memory-bound, materialized-scores) baseline. Correctness-gated: greedy
generation identical to eager. **RTX 4090, fp16, 2026-09-02** (`notebooks/phase3_end2end_attention_colab.ipynb`).

| Config (batch × prompt × new) | Eager tok/s | Triton tok/s | Lift vs eager |
|---|---|---|---|
| 16 × 960 × 16 (5 runs, median) | 822.8 | 1031.1 | **+17.4%** (runs 14.8–26.8%) |
| 32 × 928 × 8 (5 runs, median) | 959.0 | 1269.6 | **+34.0%** (runs 20.5–56.7%) |
| 8 × 896 × 64 | 519.4 | 565.4 | +8.9% |
| 16 × 512 × 64 | 1177.8 | 1130.4 | −4.0% (short context) |

> The lift comes from **prefill** (the long-sequence attention the fused kernel
> accelerates), so it grows with context length and batch and goes slightly negative
> for short prompts. **Reproducible ~16–17% at long context** (960-token prompt,
> batch 16). SDPA remains the ceiling.
>
> **Gotcha this measurement found:** `@triton.autotune(key=["N_KV"])` is catastrophic
> in a decode loop — it re-tunes every token as the KV grows (~14× SLOWER end-to-end).
> Fixed to a single launch config; that fix is what unlocks the lift.

**Where the win came from (Nsight):** `nsys` CUDA-kernel summary of the decode path
(Nsight Systems 2024.6.2, RTX 4090) shows the fused `_attention_kernel` (3072 launches,
8.65 ms, ~4.9% of GPU time) in place of eager's separate score-matmul + softmax +
elementwise ops; decode time is dominated by the projection GEMMs (`gemvx`, ~55%),
which is why a fast attention kernel moves end-to-end throughput by teens-of-percent,
not more.

---

## Phase 3 — Serving on Kubernetes

Local `kind` cluster (CPU). Server routes requests through the continuous-batching
Scheduler (`max_batch=4`), so overload genuinely queues — KEDA `metrics-api` scaler
on **`queue_depth`** (target 2/pod, max 6). Load = 60 concurrent streaming requests.
2026-09-01.

| Metric | Value | Notes |
|---|---|---|
| Pods at idle | 1 | `minReplicaCount` |
| Pods under load | 6 | scaled 1 → 4 → 6 (hit `maxReplicaCount`) |
| Scale-up latency | ~20 s | idle → 4 pods, then → 6 |
| Trigger | `queue_depth` > 2/pod | HPA event: "external metric queue_depth above target → New size: 6" |

> Real queue depth: with `max_batch=4`, a pod admits 4 requests and the rest WAIT
> (verified locally: 12 concurrent → running_batch 4, queue_depth 10). The
> ClusterIP service load-balances, so a single KEDA poll reads one pod's depth;
> the production-correct version aggregates all pods via Prometheus.

> Caveat: the ClusterIP Service load-balances, so KEDA's single poll reads one
> pod's `inflight`, not the fleet sum — production-correct scaling scrapes all pods
> with Prometheus and scales on `sum(inference_inflight_requests)`.
