// matmul — Phase 0, kernel 2: tiled matrix multiply using shared memory.
//
// C = A @ B for A[M,K], B[K,N]. The point is the SHARED-MEMORY TILING: load a
// tile of A and a tile of B into __shared__ memory, __syncthreads(), accumulate
// the partial products, advance to the next tile. This is where you feel the
// memory hierarchy. Compare against torch.matmul (cuBLAS) in the bench.
//
// TODO(you): implement the tiled kernel + a vector_add-style C++ entry point and
// PYBIND11_MODULE binding. Start with a TILE_SIZE of 16 or 32.

#include <torch/extension.h>
#include <cuda_runtime.h>

// TODO(you): __global__ void matmul_kernel(...) with __shared__ tiles.
// TODO(you): torch::Tensor matmul(torch::Tensor a, torch::Tensor b) { ... }
// TODO(you): PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul", ...); }
