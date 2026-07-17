#!/bin/bash
# start_aggregated.sh — Run on a single GB10 node (no disaggregation, no Dynamo runtime)
set -euo pipefail

MODEL="google/gemma-3-12b-it"
MAX_MODEL_LEN="131072"
GPU_MEM_UTIL="0.85"

sudo docker run -d \
  --name vllm-aggregated \
  --network host \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  -v /mnt/models:/mnt/models \
  -e HF_HOME=/mnt/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL"
