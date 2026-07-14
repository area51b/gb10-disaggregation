#!/bin/bash
# benchmark_singlenode.sh — Run on gx10-9c8c
# Runs single-node vLLM benchmarks for all models.
# Starts and stops vLLM automatically between models.
#
# Usage:
#   HF_TOKEN=your_token ./benchmark_singlenode.sh

set -e
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/models.env"

start_vllm() {
    local model=$1
    local max_len=$2
    echo ""
    echo "=== Starting vLLM: $model (MAX_MODEL_LEN=$max_len) ==="

    docker stop vllm-singlenode 2>/dev/null || true
    sleep 3

    EXTRA_ARGS=""
    if [[ "$model" == *"gemma"* ]]; then
        EXTRA_ARGS="--language-model-only"
    fi

    docker run --rm -d \
        --name vllm-singlenode \
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
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --max-model-len "$max_len" \
            --gpu-memory-utilization 0.90 \
            --port 8000 \
            $EXTRA_ARGS

    echo "  Waiting for vLLM to be ready..."
    local waited=0
    until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge 600 ]; then
            echo "  ERROR: vLLM did not start after 10 minutes"
            exit 1
        fi
        echo "  Still waiting... (${waited}s)"
    done
    echo "  vLLM ready after ${waited}s"
}

stop_vllm() {
    docker stop vllm-singlenode 2>/dev/null || true
    sleep 5
}

run_bench() {
    local model=$1 prompt_len=$2 concurrency=$3
    local requests=10
    [ "$prompt_len" = "xlarge" ] && requests=20
    python3 "$SCRIPT_DIR/benchmark.py" \
        --model "$model" \
        --requests "$requests" \
        --prompt-len "$prompt_len" \
        --concurrency "$concurrency" \
        --tag single-node
}

# ── Gemma 3 1B ───────────────────────────────────────────────────────────────
start_vllm "$GEMMA_1B" "$GEMMA_1B_MAX_LEN"
run_bench "$GEMMA_1B" medium 1
run_bench "$GEMMA_1B" medium 10
run_bench "$GEMMA_1B" long 1
run_bench "$GEMMA_1B" long 10
stop_vllm

# ── Gemma 3 4B ───────────────────────────────────────────────────────────────
start_vllm "$GEMMA_4B" "$GEMMA_4B_MAX_LEN"
run_bench "$GEMMA_4B" medium 1
run_bench "$GEMMA_4B" medium 10
run_bench "$GEMMA_4B" long 1
run_bench "$GEMMA_4B" long 10
stop_vllm

# ── Gemma 3 12B ──────────────────────────────────────────────────────────────
start_vllm "$GEMMA_12B" "$GEMMA_12B_MAX_LEN"
run_bench "$GEMMA_12B" medium 1
run_bench "$GEMMA_12B" medium 10
run_bench "$GEMMA_12B" long 1
run_bench "$GEMMA_12B" long 10
stop_vllm

# ── Gemma 3 27B ──────────────────────────────────────────────────────────────
start_vllm "$GEMMA_27B" "$GEMMA_27B_MAX_LEN"
run_bench "$GEMMA_27B" medium 1
run_bench "$GEMMA_27B" medium 10
run_bench "$GEMMA_27B" long 1
run_bench "$GEMMA_27B" long 10
stop_vllm

# ── Llama 3.1 70B FP8 (gated) ────────────────────────────────────────────────
start_vllm "$LLAMA_70B" "$LLAMA_70B_MAX_LEN"
run_bench "$LLAMA_70B" medium 1
run_bench "$LLAMA_70B" medium 10
run_bench "$LLAMA_70B" long 1
run_bench "$LLAMA_70B" long 10
stop_vllm

# ── Qwen 2.5 72B FP8 (gated) ─────────────────────────────────────────────────
start_vllm "$QWEN_72B" "$QWEN_72B_MAX_LEN"
run_bench "$QWEN_72B" medium 1
run_bench "$QWEN_72B" medium 10
run_bench "$QWEN_72B" long 1
run_bench "$QWEN_72B" long 10
stop_vllm

echo ""
echo "=== ALL SINGLE-NODE BENCHMARKS COMPLETE ==="
echo "Results saved to: $SCRIPT_DIR/results/"
