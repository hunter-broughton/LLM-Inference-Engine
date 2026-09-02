// matmul — Phase 0, kernel 2: tiled matrix multiply using shared memory.
//
// C = A @ B for A[M,K], B[K,N]. The point is SHARED-MEMORY TILING: each block
// loads a TILE x TILE tile of A and of B into __shared__ memory, __syncthreads(),
// accumulates the partial products, then advances to the next tile along K. Every
// A/B element is loaded from global memory once per tile and reused TILE times
// from fast shared memory — that reuse is the whole point. Compare vs cuBLAS
// (torch.matmul) in the bench.

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // output row this thread computes
    int col = blockIdx.x * TILE + threadIdx.x;   // output col this thread computes
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        // Cooperative load; zero-pad the edges so out-of-range tiles contribute 0.
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();   // don't overwrite the tile until everyone's done with it
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dims must match: A[M,K] @ B[K,N]");
    a = a.contiguous();
    b = b.contiguous();
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "tiled matmul A @ B (CUDA)");
}
