// rmsnorm — Phase 0, kernel 4: RMSNorm (the normalization TinyLlama uses).
//
// For each row x: y = x / sqrt(mean(x^2) + eps) * weight. No mean-subtraction
// (unlike LayerNorm). One block per row: the threads cooperatively reduce the
// sum of squares in shared memory, then each thread scales its slice of the row.

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* x, const float* w, float* y,
                               int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* xrow = x + (long)row * N;
    float* yrow = y + (long)row * N;

    // Each thread sums the squares of its strided slice of the row.
    extern __shared__ float sdata[];
    float local = 0.0f;
    for (int i = tid; i < N; i += nthreads) {
        float v = xrow[i];
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    // Tree reduction to sdata[0] = sum of squares over the whole row.
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / N + eps);   // 1 / sqrt(mean(x^2) + eps)
    for (int i = tid; i < N; i += nthreads)
        yrow[i] = xrow[i] * inv * w[i];
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, N]");
    TORCH_CHECK(weight.numel() == x.size(1), "weight must match row length");
    x = x.contiguous();
    weight = weight.contiguous();
    int rows = x.size(0), N = x.size(1);
    auto y = torch::empty_like(x);

    const int threads = 256;   // power of two for the tree reduction
    rmsnorm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
        N, (float)eps);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
}
