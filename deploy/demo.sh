#!/usr/bin/env bash
# Phase 3 — one-command local KEDA autoscaling demo (CPU, free, on a Mac).
#
# Stands up a kind cluster, builds+loads the CPU server image, installs KEDA,
# applies the manifests, and waits for the server to be ready. After it prints
# READY, run ./deploy/loadtest.sh in another terminal and watch pods scale with
#   kubectl get pods -w
#
# Tear it all down with ./deploy/teardown.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=mini-llm
IMAGE=mini-llm-inference:latest

echo "==> 1/5  kind cluster '$CLUSTER'"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER"
fi
kubectl config use-context "kind-$CLUSTER"

echo "==> 2/5  build CPU image (first build downloads torch + bakes gpt2, ~several min)"
docker build -t "$IMAGE" -f deploy/Dockerfile .

echo "==> 3/5  load image into kind"
kind load docker-image "$IMAGE" --name "$CLUSTER"

echo "==> 4/5  install KEDA (helm)"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install keda kedacore/keda --namespace keda --create-namespace --wait

echo "==> 5/5  apply manifests"
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl rollout status deploy/inference-server --timeout=180s
kubectl apply -f deploy/k8s/keda-scaledobject.yaml

cat <<'EOF'

==================================================================
READY. Now, in TWO other terminals:

  # terminal A — watch the pods scale
  kubectl get pods -w

  # terminal B — port-forward, then generate load
  kubectl port-forward svc/inference-server 8000:80
  ./deploy/loadtest.sh          # fires concurrent requests

Watch replica count climb toward maxReplicaCount under load, then
fall back after the cooldown. Inspect the autoscaler with:

  kubectl get scaledobject,hpa
  kubectl describe scaledobject inference-server

Tear down with: ./deploy/teardown.sh
==================================================================
EOF
