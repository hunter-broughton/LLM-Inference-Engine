# Kernels

Raw CUDA (`cuda/`) and Triton (`triton/`) kernels, with correctness tests
(`tests/`) and microbenchmarks (`bench/`).

## The per-kernel loop (Phase 0)

For **every** kernel, in this order:

1. **Write it** — `cuda/<name>.cu`. Start from the stub; fill in the kernel body
   and the launch config.
2. **Test it** — `tests/test_<name>.py`. Load via `torch.utils.cpp_extension.load`,
   compare against the PyTorch reference with `torch.allclose` (see
   `test_vector_add.py` for the worked pattern).
3. **Benchmark it** — `bench/bench_<name>.py`. Time it vs the PyTorch/cuBLAS
   equivalent; record the number in `../BENCHMARKS.md`.
4. **Profile it** — run `ncu` and write a one-paragraph note: is it memory-bound or
   compute-bound? What's the occupancy? What's the limiter? Put the conclusion in
   `../BENCHMARKS.md`.

Kernel order: `vector_add` → `matmul` (tiled) → `softmax` → `rmsnorm`.

## Building

Kernels compile on first load via `cpp_extension` (no separate build step). The
arch flag matters:

- Local Pascal dev card: `-arch=sm_61`
- Rented L4: `-arch=sm_89`; A10: `sm_86`; A100: `sm_80`

Set it in the `extra_cuda_cflags` of each test/bench `load()` call.

## Profiling cheatsheet

```bash
# Per-kernel deep dive (compute/memory bound, occupancy, limiter)
ncu --set full -o profile python kernels/bench/bench_vector_add.py

# Timeline / whole-program (useful from Phase 1 on)
nsys profile -o timeline python bench/run.py
```
