# Phase 3 — container image for the inference server (CPU build).
#
# This is the free/local-demo image: it runs gpt2 on CPU so it works on any
# machine (including a Mac kind cluster) with no GPU, no CUDA toolkit, no driver.
# For a real GPU deploy, swap the base for `nvidia/cuda:12.4.1-runtime-ubuntu22.04`,
# install the CUDA torch wheel, and add `nvidia.com/gpu` to the Deployment — the
# app code is identical; only the base image and the torch wheel change.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    MODEL_NAME=gpt2

WORKDIR /app

# Install deps first so the layer caches across code changes. Use the CPU-only
# torch wheel — it's a fraction of the size of the CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the engine + serving code (kernels too — they JIT-compile on first use).
COPY engine/ engine/
COPY kernels/ kernels/
COPY serving/ serving/

# Bake the model into the image so pods start fast and don't hit the HF Hub on
# every cold start (matters when KEDA is spinning pods up under load).
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')"

EXPOSE 8000

# One worker: the model loads once per process. Scale by adding PODS (k8s + KEDA),
# not workers.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
