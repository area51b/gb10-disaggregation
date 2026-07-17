#!/bin/bash
# start_frontend.sh
# Starts the Dynamo HTTP frontend on port 8000.
# Requires etcd and at least the decode worker to be running first.

set -e

# Stop existing frontend container if running
sudo docker stop dynamo-frontend 2>/dev/null || true

echo "Starting Dynamo frontend on port 8000..."

sudo docker run -d \
  --name dynamo-frontend \
  --network host \
  -v /mnt/models:/mnt/models \
  -e HF_HOME=/mnt/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e ETCD_ENDPOINTS="192.168.1.26:2379" \
  -e NATS_SERVER="nats://192.168.1.26:4222" \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  python3 -m dynamo.frontend

echo "Frontend starting. Watch logs with:"
echo "sudo docker logs -f dynamo-frontend"
echo "Wait for: Uvicorn running on http://0.0.0.0:8000"
