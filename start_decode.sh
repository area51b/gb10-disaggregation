#!/bin/bash
# start_decode.sh — Run on gx10-9c8c
# Starts the Dynamo vLLM decode worker.
#
# Usage:
#   MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_decode.sh
#
# Required env vars:
#   MODEL          — HuggingFace model ID
#   MAX_MODEL_LEN  — maximum context length (see models.env)
#
# Optional env vars:
#   HF_TOKEN       — required for gated models (Llama, Qwen)
#   GPU_MEM_UTIL   — GPU memory utilization fraction (default: 0.90)

set -e

: "${MODEL:?MODEL is required. Example: MODEL=google/gemma-3-12b-it}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required. Example: MAX_MODEL_LEN=131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# Stop existing decode container if running
sudo docker stop dynamo-decode 2>/dev/null || true

echo "Starting decode worker: $MODEL (MAX_MODEL_LEN=$MAX_MODEL_LEN)"

# Gemma 3 models require --language-model-only (multimodal architecture)
EXTRA_ARGS=""
if [[ "$MODEL" == *"gemma"* ]]; then
  EXTRA_ARGS="--language-model-only"
fi

sudo docker run -d \
  --name dynamo-decode \
  --network host \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size=16g \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v /sys/devices:/sys/devices \
  -v /mnt/models:/mnt/models \
  -e HF_HOME=/mnt/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e NIXL_LOG_LEVEL=DEBUG \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e ETCD_ENDPOINTS="192.168.1.26:2379" \
  -e NATS_SERVER="nats://192.168.1.26:4222" \
  -e UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc,gdr_copy,tcp \
  -e UCX_NET_DEVICES=rocep1s0f1:1 \
  -e UCX_IB_ADDR_TYPE=eth \
  -e UCX_RNDV_SCHEME=get_zcopy \
  -e UCX_RNDV_THRESH=0 \
  -e UCX_RC_TIMEOUT=600s \
  -e UCX_KEEPALIVE_INTERVAL=300s \
  -e UCX_IB_GPU_DIRECT_RDMA=yes \
  -e UCX_IB_REG_METHODS=odp,rcache,direct \
  -e UCX_RCACHE_PURGE_ON_FORK=y \
  -e UCX_IB_PREFER_NEAREST_DEVICE=y \
  -e UCX_LOG_LEVEL=info \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":0,"kv_parallel_size":1}' \
    --kv-events-config '{"enable_kv_cache_events":false}' \
    --no-enable-prefix-caching \
    $EXTRA_ARGS

echo "Decode worker starting. Watch logs with:"
echo "sudo docker logs -f dynamo-decode"
echo "Wait for: Registered endpoint 'dynamo.backend.generate'"
