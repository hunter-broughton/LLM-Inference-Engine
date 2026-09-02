# Profiling the attention kernel with Nsight

Nsight Compute (`ncu`) and Nsight Systems (`nsys`) are free, but reading GPU
performance counters needs privileges that **free Colab blocks**
(`ERR_NVGPUCTRPERM`). Run these on a **rented GPU VM with root** — a T4/L4/A10 on
Lambda, RunPod, or Vast is ~$0.4–1.5/hr. The NVIDIA "PyTorch" images ship `ncu`
and `nsys` preinstalled.

## 0. One-time setup on the box

```bash
git clone <your repo>            # or scp the folder up
cd "Inference Engine Project"
pip install torch triton transformers          # if not already present
python -c "import triton; print('triton', triton.__version__)"
ncu --version                                   # confirm Nsight Compute is there
```

If `ncu` reports `ERR_NVGPUCTRPERM`, enable counters (needs root):

```bash
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiler.conf'
# then reboot the instance (or load the module with the option)
```

## 1. Profile the attention op (ncu) — where the memory traffic goes

`profile_attention.py` (below) runs the naive path then our Triton kernel. Profile
each kernel's DRAM traffic and throughput:

```bash
# Profile just the naive path's kernels (softmax + the two GEMMs + elementwise):
ncu --set full --launch-count 8 -k "regex:softmax|gemm|elementwise" \
    -o naive_attention python notebooks/profile_attention.py --impl naive

# Profile our fused kernel (single launch that does it all):
ncu --set full --launch-count 4 -k "regex:_attention_kernel" \
    -o triton_attention python notebooks/profile_attention.py --impl triton
```

Open the `.ncu-rep` files in the Nsight Compute UI (or `ncu -i naive_attention.ncu-rep --page details`).
The number that tells the story: **DRAM Bytes / Memory Throughput**. The naive path
moves the full `[N, N]` score matrix through DRAM; the fused kernel does not — that's
the "memory-bound → fused" evidence for the bullet.

## 2. Trace the decode path (nsys) — timeline of a generation

```bash
nsys profile -o decode_trace --trace=cuda,nvtx \
    python notebooks/profile_attention.py --impl triton --generate
nsys stats decode_trace.nsys-rep        # CUDA kernel summary, sorted by time
```

`nsys stats` prints the per-kernel time breakdown of the whole generation — use it
to state what fraction of the decode path attention was, before and after.

## 3. What to record

- `ncu`: DRAM bytes and achieved memory throughput, naive vs fused (the fused
  kernel should move far fewer bytes — the point of FlashAttention).
- `nsys`: attention's share of decode time, and the end-to-end tokens/sec from
  `phase3_end2end_attention_colab.ipynb` (eager vs Triton).
- Put the numbers in `BENCHMARKS.md`.
