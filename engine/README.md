# Engine

Model loading, the generation loop, and (Phase 2) the paged KV cache + scheduler.
Python + PyTorch; the hot paths get replaced with custom kernels from `../kernels/`.

## Files

| File | Phase | What it is |
|---|---|---|
| `model.py` | 1 | Load GPT-2 / TinyLlama via HF; thin forward wrapper |
| `sampling.py` | 1 | Greedy / temperature / top-k / top-p token sampling |
| `generate.py` | 1 | Single-stream prefill + decode loop |
| `kv_cache.py` | 2 | Block-based (paged) KV cache + allocator |
| `scheduler.py` | 2 | Continuous-batching scheduler |
| `tests/` | 1–2 | Unit tests (`pytest engine/tests`) |

## Build order

1. `model.py` is mostly done — read it to see the prefill/decode split.
2. Implement `sampling.sample`, then `generate.generate` (Phase 1). At that point
   `python bench/run.py` produces the baseline numbers for `BENCHMARKS.md`.
3. Phase 2: implement `kv_cache.py` (allocator first, it's unit-testable without
   a GPU), then `scheduler.py`, then swap in a fused kernel on the hot path.

The `TODO(you):` markers are the parts to implement; everything else is structure.
