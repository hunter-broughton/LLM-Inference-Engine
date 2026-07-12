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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO(you): guard against idx >= n, then write c[idx] = a[idx] + b[idx].
}

// C++ entry point called from Python. Allocates output, computes the launch
// config, launches the kernel, returns the result tensor.
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "shapes must match");
    auto c = torch::empty_like(a);
    int n = a.numel();

    // TODO(you): pick threads-per-block (e.g. 256) and compute blocks =
    // ceil(n / threads). Launch vector_add_kernel<<<blocks, threads>>>(...)
    // with the raw .data_ptr<float>() pointers and n.

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "A + B (CUDA)");
}
