#!/usr/bin/env bash
# Fire N concurrent streaming requests at the server to drive `inflight` up so
# KEDA scales the Deployment out. Assumes `kubectl port-forward svc/inference-server
# 8000:80` is running in another terminal (or point URL at any reachable server).
set -euo pipefail

URL="${URL:-http://127.0.0.1:8000}"
CONCURRENCY="${CONCURRENCY:-20}"
ROUNDS="${ROUNDS:-30}"

echo "load: $CONCURRENCY concurrent x $ROUNDS rounds -> $URL/generate"
for r in $(seq 1 "$ROUNDS"); do
  for c in $(seq 1 "$CONCURRENCY"); do
    curl -sN -X POST "$URL/generate" \
      -H 'content-type: application/json' \
      -d '{"prompt":"Explain GPU inference in detail:","max_new_tokens":128,"temperature":0.8}' \
      >/dev/null &
  done
  wait
  printf "round %d/%d  inflight=%s\n" "$r" "$ROUNDS" \
    "$(curl -s "$URL/metrics.json" | grep -o '"inflight":[0-9]*' || echo n/a)"
done
echo "done — load stopped; KEDA will scale back down after the cooldown."
