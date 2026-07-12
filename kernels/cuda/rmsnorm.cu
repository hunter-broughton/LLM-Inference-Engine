// rmsnorm — Phase 0, kernel 4: RMSNorm (the normalization TinyLlama uses).
//
// For each row x: y = x / sqrt(mean(x^2) + eps) * weight.
// No mean-subtraction (unlike LayerNorm) — just the RMS. Same parallel-reduction
// pattern as softmax (one reduction for the sum of squares per row). This is the
// kernel you'll later FUSE into the model hot path in Phase 2.
//
// Reference: y = torch.nn.functional.rms_norm(x, (D,), weight, eps) or compute
// manually: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.
//
// TODO(you): implement. One block per row; reduce sum-of-squares, then scale.

#include <torch/extension.h>
#include <cuda_runtime.h>

// TODO(you): __global__ void rmsnorm_kernel(...) — sum of squares, rsqrt, scale.
// TODO(you): torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) { ... }
// TODO(you): PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm", ...); }
