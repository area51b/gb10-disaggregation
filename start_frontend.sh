#!/bin/bash
# start_frontend.sh — Run on gx10-9c8c
# Starts the Dynamo HTTP frontend on port 8000.
# Requires etcd and at least the decode worker to be running first.

set -e

# Stop existing frontend container if running
sudo docker stop dynamo-frontend 2>/dev/null || true

echo "Starting Dynamo frontend on port 8000..."

sudo docker run --rm -d \
  --name dynamo-frontend \
  --network host \
  -e ETCD_ENDPOINTS="192.168.1.24:2379" \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  python3 -m dynamo.sdk.ext.vllm.frontend \
    --host 0.0.0.0 \
    --port 8000

echo "Frontend starting. Watch logs with:"
echo "sudo docker logs -f dynamo-frontend"
echo "Wait for: Uvicorn running on http://0.0.0.0:8000"
