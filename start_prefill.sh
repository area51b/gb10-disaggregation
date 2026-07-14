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
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Stop existing prefill container if running
docker stop dynamo-prefill 2>/dev/null || true

echo "Starting prefill worker: $MODEL (MAX_MODEL_LEN=$MAX_MODEL_LEN)"

# Gemma 3 models require --language-model-only (multimodal architecture)
EXTRA_ARGS=""
if [[ "$MODEL" == *"gemma"* ]]; then
  EXTRA_ARGS="--language-model-only"
fi

docker run --rm -d \
  --name dynamo-prefill \
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
  -e ETCD_ENDPOINTS="192.168.1.24:2379" \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  python3 -m dynamo.sdk.ext.vllm.worker \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --worker-type prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1}' \
    --kv-events-config '{"enable_kv_cache_events":false}' \
    $EXTRA_ARGS

echo "Prefill worker starting. Watch logs with:"
echo "  docker logs -f dynamo-prefill"
echo "Wait for: Registered endpoint 'dynamo.prefill.generate'"
