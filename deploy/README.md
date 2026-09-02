# Deploy

Containerize the server and run it on a **local Kubernetes cluster (kind) with
KEDA autoscaling** — the Phase 3 differentiator: the part that turns a fast
engine into a *service that scales*. This variant is **CPU-only and free**, so it
runs on any machine (a Mac included) with no GPU.

## Files

| File | What it is |
|---|---|
| `Dockerfile` | CPU image running the FastAPI server (gpt2 baked in) |
| `k8s/deployment.yaml` | The server Deployment (KEDA owns replicas) |
| `k8s/service.yaml` | ClusterIP service in front of the pods |
| `k8s/keda-scaledobject.yaml` | Autoscale on the `queue_depth` custom metric |
| `demo.sh` | One command: cluster + build + KEDA + apply |
| `loadtest.sh` | Fire concurrent requests to trigger scale-out |
| `teardown.sh` | Delete the cluster (nothing left running) |

## Prerequisites

```bash
# Docker Desktop must be RUNNING. Then:
brew install kind kubectl helm
```

## Run it (and watch it scale)

```bash
./deploy/demo.sh                 # stand everything up (first run pulls torch, ~minutes)

# then, in two more terminals:
kubectl get pods -w              # terminal A: watch replicas change
kubectl port-forward svc/inference-server 8000:80   # terminal B
./deploy/loadtest.sh             # terminal B (new tab): generate load
```

Under load, `queue_depth` climbs past the `targetValue` and KEDA scales the
Deployment from 1 toward `maxReplicaCount: 5`; after the load stops and the
`cooldownPeriod` passes, it scales back down. Inspect the autoscaler:

```bash
kubectl get scaledobject,hpa
kubectl describe scaledobject inference-server
```

Tear down when done (the whole point of the local demo — nothing keeps running):

```bash
./deploy/teardown.sh
```

## How the autoscaling works

KEDA's `metrics-api` trigger polls `GET /metrics.json` every `pollingInterval`
seconds and reads the `queue_depth` value. It creates an HPA under the hood targeting
`targetValue` queued requests per pod. The server admits only `max_batch` requests
at a time (through the continuous-batching scheduler), so excess requests WAIT —
a deeper queue → more pods. This is **custom-metric autoscaling** — scaling on a
signal that actually reflects inference backlog, not on CPU%.

## Scaling to a real GPU cluster

The app code is identical on GPU. To deploy for real:

1. Base the image on `nvidia/cuda:12.4.1-runtime-ubuntu22.04` and install the CUDA
   torch wheel (see the comment in `Dockerfile`).
2. Add `resources.limits: {nvidia.com/gpu: 1}` to `deployment.yaml` (needs the
   NVIDIA device plugin on the nodes).
3. For fleet-accurate scaling, scrape every pod's `/metrics` (Prometheus format)
   with Prometheus and switch the ScaledObject to a `prometheus` trigger on
   `sum(inference_queue_depth)`.

The manifests here drop onto GKE/EKS unchanged apart from those three edits.
```
