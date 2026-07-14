#!/bin/bash
# benchmark.sh — Run from either GB10 node (needs network access to frontend)
set -euo pipefail

MODEL="google/gemma-3-12b-it"
URL="192.168.1.26:8000"
BASE_DIR="/mnt/models/results/gemma-3-12b-it"
mkdir -p "$BASE_DIR"

sudo docker run -d \
  --name dynamo-benchmark \
  --network host \
  -v /mnt/models:/mnt/models \
  -e HF_HOME=/mnt/models/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13 \
  /bin/bash -c '
set -euo pipefail
MODEL="google/gemma-3-12b-it"
URL="192.168.1.26:8000"
BASE_DIR="/mnt/models/results/gemma-3-12b-it"

# Shape: ISL OSL LABEL
SHAPES=(
  "1000 128 isl1k-osl128"
)

for shape in "${SHAPES[@]}"; do
  read -r ISL OSL LABEL <<< "$shape"
  echo ""
  echo "========================================"
  echo "Shape: ISL=$ISL OSL=$OSL ($LABEL)"
  echo "========================================"

  for c in 1 5 10; do
    if [ "$c" -le 10 ]; then
      COUNT=$(( c * 40 ))
    elif [ "$c" -le 100 ]; then
      COUNT=$(( c * 25 ))
    else
      COUNT=$(( c * 15 ))
    fi
    COUNT=$(( COUNT < 30 ? 30 : COUNT > 15000 ? 15000 : COUNT ))

    OUTPUT_DIR="${BASE_DIR}/${LABEL}/c${c}"
    echo "--- Concurrency $c | Requests $COUNT | ISL=$ISL OSL=$OSL ---"

    aiperf profile \
      --model "$MODEL" \
      --url "$URL" \
      --endpoint-type chat \
      --streaming \
      --concurrency "$c" \
      --request-count "$COUNT" \
      --synthetic-input-tokens-mean "$ISL" \
      --output-tokens-mean "$OSL" \
      --extra-inputs ignore_eos:true \
      --extra-inputs min_tokens:"$OSL" \
      --artifact-dir "$OUTPUT_DIR" \
      --tokenizer "$MODEL" \
      --ui none
  done
done

echo ""
echo "=== Baseline benchmark complete ==="
'
