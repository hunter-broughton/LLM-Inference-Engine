#!/usr/bin/env bash
# Tear down the local demo cluster so nothing keeps running (or costing).
set -euo pipefail
CLUSTER=mini-llm
kind delete cluster --name "$CLUSTER"
echo "deleted kind cluster '$CLUSTER'. Docker image mini-llm-inference:latest is"
echo "still cached locally; remove it with: docker rmi mini-llm-inference:latest"
