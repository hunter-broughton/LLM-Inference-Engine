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

| Impl | Shape | Time | GFLOP/s | Hardware | Arch |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Nsight note:** _TBD_

### softmax (row-wise, numerically stable)

| Impl | Shape | Time | Bandwidth (GB/s) | Hardware | Arch |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Nsight note:** _TBD_

### rmsnorm

| Impl | Shape | Time | Bandwidth (GB/s) | Hardware | Arch |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Nsight note:** _TBD_

---

## Phase 1 — Baseline inference (GPT-2 124M)

Single-stream generation, greedy + temperature sampling. This is the bar Phase 2 beats.

| Metric | Value | Hardware | Date |
|---|---|---|---|
| Tokens/sec | _TBD_ | | |
| TTFT | _TBD_ | | |
| Latency p50 | _TBD_ | | |
| Latency p99 | _TBD_ | | |

---

## Phase 2 — Real engine (paged KV cache + continuous batching + custom kernel)

Throughput under concurrency, vs Phase 1 baseline.

| Concurrency | Baseline tok/s | Engine tok/s | Speedup | Hardware | Date |
|---|---|---|---|---|---|
| 1 | _TBD_ | _TBD_ | | | |
| 8 | _TBD_ | _TBD_ | | | |
| 32 | _TBD_ | _TBD_ | | | |
| 64 | _TBD_ | _TBD_ | | | |

**Where the win came from (Nsight):** _TBD_

---

## Phase 3 — Serving on Kubernetes

| Metric | Value | Notes |
|---|---|---|
| Pods scaled under load | _TBD_ | KEDA/HPA on queue depth or GPU util |
| Sustained throughput (cluster) | _TBD_ | |
