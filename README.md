# Dell Pro Max with GB10 — NVIDIA Dynamo Disaggregated Inference

Scripts and benchmark tools for running NVIDIA Dynamo disaggregated inference on two Dell Pro Max with GB10 nodes.

Companion to the article series:
- **Part 1:** How to Run NVIDIA Dynamo Disaggregated Inference on Two Dell Pro Max with GB10 *(setup guide)*
- **Part 2:** What Actually Happens When You Benchmark Disaggregated Inference on Two Dell Pro Max with GB10 *(results)*

---

## Hardware

| Node | Role | Management IP |
|---|---|---|
| promaxgb10-f51e | decode + etcd + frontend | 192.168.1.26 |
| promaxgb10-f525 | prefill | 192.168.1.24 |

Components:
* GB10 Grace Blackwell Superchip per node (sm_121)
* 128GB unified LPDDR5X (shared CPU + GPU memory)
* 1× 200GbE ConnectX-7 NICs per node

## Software

| Component | Version |
|---|---|
| Container | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13` |
| vLLM | 0.17.1 |
| NIXL | 0.10.1 |
| Service discovery | etcd |

---

## Critical fixes for sm_121

Three environment variables / flags are required. Without them, workers crash silently.

```bash
# 1. FlashInfer kernels compiled for sm_120, crash on sm_121
export VLLM_USE_FLASHINFER_SAMPLER=0

# 2. Gemma 3 is multimodal — text-only flag required
--language-model-only

# 3. etcd endpoint must not have http:// prefix
ETCD_ENDPOINTS="192.168.1.26:2379"   # correct
ETCD_ENDPOINTS="http://192.168.1.26:2379"  # wrong — silent failure
```

---

## Startup sequence

```bash
# 1. promaxgb10-f51e — start etcd
bash start_infra.sh
# wait: ==> etcd is ready

# 2. promaxgb10-f51e — start decode worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_decode.sh
# wait: Registered endpoint 'dynamo.backend.generate'

# 3. promaxgb10-f525 — start prefill worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_prefill.sh
# wait: Registered endpoint 'dynamo.prefill.generate'

# 4. promaxgb10-f51e — start HTTP frontend
./start_frontend.sh
# wait: Uvicorn running on http://0.0.0.0:8000

# 5. promaxgb10-f51e — start web UI proxy (optional)
python3 serve.py
# open: http://192.168.1.26:9999/dynamo_chat.html
```

---

## Models and MAX_MODEL_LEN

See `models.env` for all values. Key constraints:

| Model | MAX_MODEL_LEN | Notes |
|---|---|---|
| Gemma 3 1B | 32768 | Hard limit from architecture |
| Gemma 3 4B | 131072 | Fits comfortably |
| Gemma 3 12B | 131072 | Fits comfortably |
| Gemma 3 27B | 65536 | 131072 causes OOM (`NV_ERR_NO_MEMORY`) |
| Llama 3.1 70B FP8 | 32768 | Gated — requires `HF_TOKEN` |
| Qwen 2.5 72B FP8 | 32768 | Gated — requires `HF_TOKEN` |

---

## Running benchmarks

```bash
# AIPerf for benchmarking (automated, unattended)
./benchmark.sh
```

Prompt sizes: `short` (~5 tokens), `medium` (~100 tokens), `long` (~300 tokens), `xlarge` (~16K tokens), `xxlarge` (~27.5K tokens)

Results saved as JSON to `results/`.

---

## Files

| File | Node | Purpose |
|---|---|---|
| `docker-compose.yml` | promaxgb10-f51e | etcd service |
| `start_infra.sh` | promaxgb10-f51e | Start etcd |
| `start_decode.sh` | promaxgb10-f51e | Start decode worker |
| `start_prefill.sh` | promaxgb10-f525 | Start prefill worker |
| `start_frontend.sh` | promaxgb10-f51e | Start HTTP frontend |
| `serve.py` | promaxgb10-f51e | CORS proxy for web UI |
| `benchmark.sh` | promaxgb10-f51e | Run benchmarks |
| `models.env` | both | Model IDs and MAX_MODEL_LEN |

