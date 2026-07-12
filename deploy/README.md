# Deploy

Containerize the server and run it on a local Kubernetes cluster with GPU access
and KEDA autoscaling. This is the Phase 3 differentiator — the part that turns a
fast kernel into a *service that scales*.

## Files

| File | What it is |
|---|---|
| `Dockerfile` | CUDA-runtime image running the FastAPI server |
| `k8s/deployment.yaml` | The server Deployment (requests 1 GPU) |
| `k8s/service.yaml` | ClusterIP service in front of the pods |
| `k8s/keda-scaledobject.yaml` | Autoscale on `inference_queue_depth` |

## Local cluster (kind) — rough path

```bash
# 1. Build and load the image into kind.
docker build -t mini-llm-inference:latest -f deploy/Dockerfile .
kind load docker-image mini-llm-inference:latest

# 2. GPU access in kind requires the NVIDIA device plugin (and the NVIDIA
#    container toolkit on the host). Install the device plugin DaemonSet.

# 3. Install KEDA + a Prometheus that scrapes /metrics.

# 4. Apply the manifests.
kubectl apply -f deploy/k8s/

# 5. Generate load and watch it scale.
kubectl get pods -w
```

## Acceptance (from PROJECT.md)

- One-command deploy to a local cluster.
- Pods demonstrably scale under load.
- README shows the architecture diagram **and** the numbers.

> GPU-in-kind is the fiddly part; if it fights you, k3s with the NVIDIA runtime
> or a single rented GPU node is a fine fallback. The scaling *story* is what
> matters, not the specific local distro.
