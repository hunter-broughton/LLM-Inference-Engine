// vector_add — Phase 0, kernel 1.
//
// Element-wise c[i] = a[i] + b[i]. The "hello world" of CUDA: gets you the
// launch-config + memory-model muscle memory before the harder kernels.
//
// This file is loaded from Python via torch.utils.cpp_extension (see
// kernels/tests/test_vector_add.py). Keep the binding name in sync with the test.

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Each thread computes ONE output element. Its global index is its position
    // within its block (threadIdx.x) plus all the threads in the blocks before it
    // (blockIdx.x * blockDim.x). This flat-index pattern is the CUDA fundamental.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Guard: we launch a whole number of fixed-size blocks, so the last block
    // usually has threads past the end of the array — they must NOT write, or
    // they scribble out of bounds. Every element-wise kernel needs this line.
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C++ entry point called from Python. Allocates output, computes the launch
// config, launches the kernel, returns the result tensor.
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "shapes must match");
    auto c = torch::empty_like(a);
    int n = a.numel();

    // Launch config: a "block" is a group of threads (256 is a common sweet
    // spot); we need enough blocks to cover all n elements. ceil(n/threads) via
    // integer math is (n + threads - 1) / threads — that's why the guard above
    // exists, since this rounds UP and over-provisions the last block.
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "A + B (CUDA)");
}
