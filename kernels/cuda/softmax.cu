// softmax — Phase 0, kernel 3: row-wise, numerically stable.
//
// For each row: subtract the row max before exp() (numerical stability), sum the
// exps, divide. The interesting part is the PARALLEL REDUCTION to find the row
// max and the row sum within a block. Compare against torch.softmax(x, dim=-1).
//
// TODO(you): implement. One block per row is a clean starting layout; use a
// shared-memory reduction (or warp shuffles) for the max and the sum.

#include <torch/extension.h>
#include <cuda_runtime.h>

// TODO(you): __global__ void softmax_kernel(...) — row max, exp, row sum, divide.
// TODO(you): torch::Tensor softmax(torch::Tensor x) { ... }  // last-dim softmax
// TODO(you): PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("softmax", ...); }
