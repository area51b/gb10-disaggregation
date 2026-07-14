# DGX Spark — NVIDIA Dynamo Disaggregated Inference

Scripts and benchmark tools for running NVIDIA Dynamo disaggregated inference on two ASUS Ascent GX10 (DGX Spark) nodes.

Companion to the article series:
- **Part 1:** How to Run NVIDIA Dynamo Disaggregated Inference on Two DGX Sparks *(setup guide)*
- **Part 2:** What Actually Happens When You Benchmark Disaggregated Inference on Two DGX Sparks *(results)*

---

## Hardware

| Node | Role | Management IP | KV Cache IP | NFS IP |
|---|---|---|---|---|
| gx10-9c8c | decode + etcd + frontend | 192.168.1.24 | 10.88.0.2 | 10.77.0.2 |
| gx10-4a61 | prefill | 192.168.68.119 | 10.88.0.1 | 10.77.0.1 |

- GB10 Grace Blackwell Superchip per node (sm_121)
- 128GB unified LPDDR5X (shared CPU + GPU memory)
- 2× 200GbE ConnectX-7 NICs per node

## Software

| Component | Version |
|---|---|
| Container | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0-dev.1` |
| vLLM | 0.17.1 |
| NIXL | 0.10.1 |
| Service discovery | etcd |

---

## Critical fixes for sm_121 (DGX Spark)

Three environment variables / flags are required. Without them, workers crash silently.

```bash
# 1. FlashInfer kernels compiled for sm_120, crash on sm_121
export VLLM_USE_FLASHINFER_SAMPLER=0

# 2. Gemma 3 is multimodal — text-only flag required
--language-model-only

# 3. etcd endpoint must not have http:// prefix
ETCD_ENDPOINTS="192.168.1.24:2379"   # correct
ETCD_ENDPOINTS="http://192.168.1.24:2379"  # wrong — silent failure
```

---

## Startup sequence

```bash
# 1. gx10-9c8c — start etcd
bash start_infra.sh
# wait: ==> etcd is ready

# 2. gx10-9c8c — start decode worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_decode.sh
# wait: Registered endpoint 'dynamo.backend.generate'

# 3. gx10-4a61 — start prefill worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_prefill.sh
# wait: Registered endpoint 'dynamo.prefill.generate'

# 4. gx10-9c8c — start HTTP frontend
./start_frontend.sh
# wait: Uvicorn running on http://0.0.0.0:8000

# 5. gx10-9c8c — start web UI proxy (optional)
python3 serve.py
# open: http://192.168.1.24:9999/dynamo_chat.html
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
# Single disaggregated run
python3 benchmark.py \
  --model google/gemma-3-12b-it \
  --requests 10 \
  --prompt-len medium \
  --concurrency 1 \
  --tag disaggregated

# All single-node baselines (automated, unattended)
HF_TOKEN=your_token ./benchmark_singlenode.sh
```

Prompt sizes: `short` (~5 tokens), `medium` (~100 tokens), `long` (~300 tokens), `xlarge` (~16K tokens), `xxlarge` (~27.5K tokens)

Results saved as JSON to `results/`.

---

## Files

| File | Node | Purpose |
|---|---|---|
| `docker-compose.yml` | gx10-9c8c | etcd service |
| `start_infra.sh` | gx10-9c8c | Start etcd |
| `start_decode.sh` | gx10-9c8c | Start decode worker |
| `start_prefill.sh` | gx10-4a61 | Start prefill worker |
| `start_frontend.sh` | gx10-9c8c | Start HTTP frontend |
| `serve.py` | gx10-9c8c | CORS proxy for web UI |
| `benchmark.py` | gx10-9c8c | Run benchmarks |
| `benchmark_singlenode.sh` | gx10-9c8c | Automated single-node runs |
| `models.env` | both | Model IDs and MAX_MODEL_LEN |

---

## NFS setup (summary)

Model weights live on a USB SSD on gx10-9c8c, shared via NFS to gx10-4a61.
NVMe-oF was tried first and caused repeated ext4 filesystem corruption on interrupted writes. NFS replaced it — zero errors since.

```bash
# gx10-9c8c (server) — /etc/exports
/mnt/models 10.77.0.1(rw,sync,no_subtree_check,no_root_squash)

# gx10-4a61 (client) — /etc/fstab
10.77.0.2:/mnt/models /mnt/models nfs rsize=1048576,wsize=1048576,hard,intr,_netdev,nofail 0 0
```

> Do not add `noresvport` — the server requires privileged source ports.
