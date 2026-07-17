# Dell Pro Max with GB10 — NVIDIA Dynamo Disaggregated Inference

Scripts and benchmark tools for running NVIDIA Dynamo disaggregated inference on two Dell Pro Max with GB10 nodes.

Companion to the article series:
- **Part 1:** How to Run NVIDIA Dynamo Disaggregated Inference on Two Dell Pro Max with GB10 *(setup guide)*
- **Part 2:** What Actually Happens When You Benchmark Disaggregated Inference on Two Dell Pro Max with GB10 *(results)*

---

## Hardware

| Node | Role | Management IP |
|---|---|---|
| promaxgb10-1 | decode + etcd + frontend | 192.168.1.26 |
| promaxgb10-2 | prefill | 192.168.1.24 |

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
# 1. promaxgb10-1 — start etcd
bash start_infra.sh
# wait: ==> etcd is ready

# 2. promaxgb10-1 — start decode worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_decode.sh
# wait: Registered endpoint 'dynamo.backend.generate'

# 3. promaxgb10-2 — start prefill worker
MODEL=google/gemma-3-12b-it MAX_MODEL_LEN=131072 ./start_prefill.sh
# wait: Registered endpoint 'dynamo.prefill.generate'

# 4. promaxgb10-1 — start HTTP frontend
./start_frontend.sh
# wait: Uvicorn running on http://0.0.0.0:8000

# 5. promaxgb10-1 — start web UI proxy (optional)
python3 serve.py
# open: http://192.168.1.26:9999/dynamo_chat.html
```

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
| `docker-compose.yml` | promaxgb10-1 | etcd service |
| `start_infra.sh` | promaxgb10-1 | Start etcd |
| `start_decode.sh` | promaxgb10-1 | Start decode worker |
| `start_prefill.sh` | promaxgb10-2 | Start prefill worker |
| `start_frontend.sh` | promaxgb10-1 | Start HTTP frontend |
| `serve.py` | promaxgb10-1 | CORS proxy for web UI |
| `benchmark.sh` | promaxgb10-1 | Run benchmarks |

---

## Results

Concurrency 1 | Requests 40 | ISL=1000 OSL=128

```bash
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃          Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│   Time to First │  1,195.43 │    902.80 │  4,368.32 │  3,838.07 │  1,248.54 │  1,020.67 │ 624.95 │
│      Token (ms) │           │           │           │           │           │           │        │
│  Time to Second │    129.43 │    126.83 │    132.19 │    131.74 │    130.50 │    129.38 │   1.06 │
│      Token (ms) │           │           │           │           │           │           │        │
│   Time to First │  1,195.43 │    902.80 │  4,368.32 │  3,838.07 │  1,248.54 │  1,020.67 │ 624.95 │
│    Output Token │           │           │           │           │           │           │        │
│            (ms) │           │           │           │           │           │           │        │
│ Request Latency │ 17,611.04 │ 17,319.28 │ 20,764.69 │ 20,247.27 │ 17,666.04 │ 17,432.99 │ 623.14 │
│            (ms) │           │           │           │           │           │           │        │
│     Inter Token │    129.26 │    129.06 │    129.71 │    129.62 │    129.35 │    129.24 │   0.11 │
│    Latency (ms) │           │           │           │           │           │           │        │
│    Output Token │      7.74 │      7.71 │      7.75 │      7.75 │      7.74 │      7.74 │   0.01 │
│  Throughput Per │           │           │           │           │           │           │        │
│            User │           │           │           │           │           │           │        │
│ (tokens/sec/us… │           │           │           │           │           │           │        │
│      E2E Output │      7.28 │      6.16 │      7.39 │      7.39 │      7.38 │      7.34 │   0.23 │
│           Token │           │           │           │           │           │           │        │
│      Throughput │           │           │           │           │           │           │        │
│ (tokens/sec/us… │           │           │           │           │           │           │        │
│ Output Sequence │    128.00 │    128.00 │    128.00 │    128.00 │    128.00 │    128.00 │   0.00 │
│ Length (tokens) │           │           │           │           │           │           │        │
│  Input Sequence │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │   0.00 │
│ Length (tokens) │           │           │           │           │           │           │        │
│    Output Token │      7.27 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      Throughput │           │           │           │           │           │           │        │
│    (tokens/sec) │           │           │           │           │           │           │        │
│         Request │      0.06 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      Throughput │           │           │           │           │           │           │        │
│  (requests/sec) │           │           │           │           │           │           │        │
│   Request Count │     40.00 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      (requests) │           │           │           │           │           │           │        │
└─────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────┘
```

Concurrency 5 | Requests 200 | ISL=1000 OSL=128

```bash
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃          Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│   Time to First │  1,511.11 │    656.21 │  6,558.64 │  5,292.64 │  2,042.25 │  1,490.49 │ 852.76 │
│      Token (ms) │           │           │           │           │           │           │        │
│  Time to Second │    111.54 │      9.04 │    236.70 │    223.58 │    116.10 │    112.01 │  24.40 │
│      Token (ms) │           │           │           │           │           │           │        │
│   Time to First │  1,511.11 │    656.21 │  6,558.64 │  5,292.64 │  2,042.25 │  1,490.49 │ 852.76 │
│    Output Token │           │           │           │           │           │           │        │
│            (ms) │           │           │           │           │           │           │        │
│ Request Latency │ 15,819.46 │ 14,919.55 │ 21,016.85 │ 19,491.85 │ 16,359.70 │ 15,837.56 │ 854.90 │
│            (ms) │           │           │           │           │           │           │        │
│     Inter Token │    112.66 │    111.44 │    114.71 │    113.81 │    113.48 │    112.65 │   0.52 │
│    Latency (ms) │           │           │           │           │           │           │        │
│    Output Token │      8.88 │      8.72 │      8.97 │      8.95 │      8.93 │      8.88 │   0.04 │
│  Throughput Per │           │           │           │           │           │           │        │
│            User │           │           │           │           │           │           │        │
│ (tokens/sec/us… │           │           │           │           │           │           │        │
│      E2E Output │      8.11 │      6.09 │      8.58 │      8.53 │      8.47 │      8.08 │   0.39 │
│           Token │           │           │           │           │           │           │        │
│      Throughput │           │           │           │           │           │           │        │
│ (tokens/sec/us… │           │           │           │           │           │           │        │
│ Output Sequence │    128.00 │    128.00 │    129.00 │    128.00 │    128.00 │    128.00 │   0.07 │
│ Length (tokens) │           │           │           │           │           │           │        │
│  Input Sequence │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │   0.00 │
│ Length (tokens) │           │           │           │           │           │           │        │
│    Output Token │     40.44 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      Throughput │           │           │           │           │           │           │        │
│    (tokens/sec) │           │           │           │           │           │           │        │
│         Request │      0.32 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      Throughput │           │           │           │           │           │           │        │
│  (requests/sec) │           │           │           │           │           │           │        │
│   Request Count │    200.00 │       N/A │       N/A │       N/A │       N/A │       N/A │    N/A │
│      (requests) │           │           │           │           │           │           │        │
└─────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────┘
```

Concurrency 10 | Requests 400 | ISL=1000 OSL=128

```bash
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃        Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to First │    693.50 │    339.93 │  5,854.18 │  4,920.10 │  1,862.41 │    347.66 │   798.06 │
│    Token (ms) │           │           │           │           │           │           │          │
│       Time to │    230.99 │     14.16 │  3,895.41 │    630.42 │    622.93 │    119.63 │   321.35 │
│  Second Token │           │           │           │           │           │           │          │
│          (ms) │           │           │           │           │           │           │          │
│ Time to First │    693.50 │    339.93 │  5,854.18 │  4,920.10 │  1,862.41 │    347.66 │   798.06 │
│  Output Token │           │           │           │           │           │           │          │
│          (ms) │           │           │           │           │           │           │          │
│       Request │ 16,596.76 │ 15,859.77 │ 22,652.91 │ 21,408.27 │ 18,387.86 │ 15,917.39 │ 1,285.35 │
│  Latency (ms) │           │           │           │           │           │           │          │
│   Inter Token │    125.23 │    122.11 │    160.94 │    140.34 │    132.37 │    122.60 │     5.48 │
│  Latency (ms) │           │           │           │           │           │           │          │
│  Output Token │      8.00 │      6.21 │      8.19 │      8.19 │      8.17 │      8.16 │     0.32 │
│    Throughput │           │           │           │           │           │           │          │
│      Per User │           │           │           │           │           │           │          │
│ (tokens/sec/… │           │           │           │           │           │           │          │
│    E2E Output │      7.75 │      5.65 │      8.07 │      8.07 │      8.06 │      8.04 │     0.53 │
│         Token │           │           │           │           │           │           │          │
│    Throughput │           │           │           │           │           │           │          │
│ (tokens/sec/… │           │           │           │           │           │           │          │
│        Output │    128.00 │    127.00 │    128.00 │    128.00 │    128.00 │    128.00 │     0.07 │
│      Sequence │           │           │           │           │           │           │          │
│        Length │           │           │           │           │           │           │          │
│      (tokens) │           │           │           │           │           │           │          │
│         Input │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │  1,000.00 │     0.00 │
│      Sequence │           │           │           │           │           │           │          │
│        Length │           │           │           │           │           │           │          │
│      (tokens) │           │           │           │           │           │           │          │
│  Output Token │     77.07 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│    Throughput │           │           │           │           │           │           │          │
│  (tokens/sec) │           │           │           │           │           │           │          │
│       Request │      0.60 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│    Throughput │           │           │           │           │           │           │          │
│ (requests/se… │           │           │           │           │           │           │          │
│ Request Count │    400.00 │       N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│    (requests) │           │           │           │           │           │           │          │
└───────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘
```
