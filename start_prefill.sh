#!/bin/bash
# start_prefill.sh — Run on gx10-4a61
# Starts the Dynamo vLLM prefill worker.
#
# Usage:
#   MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_prefill.sh
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

# Stop existing prefill container if running
sudo docker stop dynamo-prefill 2>/dev/null || true

echo "Starting prefill worker: $MODEL (MAX_MODEL_LEN=$MAX_MODEL_LEN)"

# Gemma 3 models require --language-model-only (multimodal architecture)
EXTRA_ARGS=""
if [[ "$MODEL" == *"gemma"* ]]; then
  EXTRA_ARGS="--language-model-only"
fi

sudo docker run -d \
  --name dynamo-prefill \
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
  -e UCX_LOG_LEVEL=info \
  -e NIXL_LOG_LEVEL=DEBUG \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e ETCD_ENDPOINTS="192.168.1.26:2379" \
  -e NATS_SERVER="nats://192.168.1.26:4222" \
  -e UCX_TLS=rc,rc_x,cuda_copy,cuda_ipc \
  -e UCX_NET_DEVICES=rocep1s0f1:1,rocep1s0f0:1 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0 \
  -e UCX_LOG_LEVEL=info \
  -e NIXL_LOG_LEVEL=INFO \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1}' \
    --kv-events-config '{"enable_kv_cache_events":false}' \
    $EXTRA_ARGS

echo "Prefill worker starting. Watch logs with:"
echo "sudo docker logs -f dynamo-prefill"
echo "Wait for: Registered endpoint 'dynamo.prefill.generate'"
