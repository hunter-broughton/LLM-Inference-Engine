# Phase 3 — container image for the inference server.
#
# Base off an NVIDIA CUDA *runtime* image (not devel) so the container is smaller
# and ships the driver-userspace libs the GPU needs. Match the CUDA version to
# what your torch wheel was built against (check `torch.version.cuda`).
#
# NOTE: requirements.txt installs the default torch wheel. For a CUDA image you
# usually want the matching CUDA wheel from the PyTorch index — adjust the pip
# line below to your CUDA version when you build for real.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the engine + serving code (kernels too — they JIT-compile on first use).
COPY engine/ engine/
COPY kernels/ kernels/
COPY serving/ serving/

EXPOSE 8000

# One worker: the model lives in GPU memory and isn't fork-safe to duplicate.
# Scale by adding PODS (k8s + KEDA), not workers.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
