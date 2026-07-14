#!/usr/bin/env python3
"""
benchmark.py — Disaggregated inference benchmark for DGX Spark lab.
Measures TTFT, ITL, and throughput. Saves results to results/ as JSON.

Usage:
  python3 benchmark.py --model google/gemma-3-12b-it --requests 20 --prompt-len medium --tag disaggregated
  python3 benchmark.py --model google/gemma-3-12b-it --requests 10 --prompt-len long --tag single-node
  python3 benchmark.py --model google/gemma-3-12b-it --requests 10 --prompt-len xlarge --concurrency 10 --tag disaggregated

Arguments:
  --model       HuggingFace model ID
  --requests    Number of requests to run (default: 10)
  --prompt-len  Prompt size: short | medium | long | xlarge | xxlarge (default: medium)
  --concurrency Number of concurrent requests (default: 1)
  --max-tokens  Max output tokens per request (default: 256)
  --tag         Label for this run: disaggregated | single-node (default: disaggregated)
  --endpoint    Dynamo/vLLM endpoint (default: http://192.168.1.26:8000)
  --no-warmup   Skip warmup request
"""

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime

ENDPOINT = "http://192.168.1.26:8000"

# ── Prompt sets ──────────────────────────────────────────────────────────────

PROMPTS = {
    "short": [
        "What is 2+2?",
        "Name the capital of France.",
        "What color is the sky?",
        "How many days in a week?",
        "What is the boiling point of water in Celsius?",
    ],
    "medium": [
        "Explain how a transformer neural network works in 3 sentences.",
        "Write a Python function that checks if a number is prime.",
        "What are the main differences between TCP and UDP protocols?",
        "Describe the water cycle in simple terms.",
        "Write a haiku about machine learning.",
        "What is gradient descent and why is it used?",
        "Explain what a GPU is and why it matters for AI.",
        "Write a SQL query to find the top 5 customers by total purchase amount.",
        "What is the difference between supervised and unsupervised learning?",
        "Explain what CUDA is in one paragraph.",
    ],
    "long": [
        "Write a detailed explanation of how attention mechanisms work in transformer models, including the mathematical formulation of scaled dot-product attention.",
        "Write a Python class implementing a binary search tree with insert, delete, and search methods. Include docstrings and example usage.",
        "Explain the history of GPU computing from fixed-function graphics pipelines to programmable shaders to general-purpose GPU computing. Include key milestones.",
        "Write a comprehensive guide to distributed systems consistency models, covering eventual consistency, strong consistency, causal consistency, and linearizability.",
        "Describe in detail how neural networks learn through backpropagation, including the chain rule, gradient computation, and weight update steps.",
    ],
    "xlarge": [
        # ~16K tokens — repeated technical paragraph
        ("NVIDIA's Grace Blackwell Superchip combines a 20-core ARM CPU with a Blackwell GPU "
         "die via NVLink-C2C, sharing 128GB of LPDDR5X unified memory at 273 GB/s. "
         "The compute capability is sm_121, distinct from the datacenter GB200 (sm_100) "
         "and consumer RTX 5090 (sm_120). This unified memory architecture eliminates "
         "PCIe bottlenecks but introduces a shared memory budget for weights, KV cache, "
         "and OS buffers. Disaggregated serving on this hardware requires careful "
         "MAX_MODEL_LEN tuning to avoid GPU allocator OOM errors at the kernel level. ") * 65
        + "Summarize the key architectural trade-offs described above.",
    ],
    "xxlarge": [
        # ~27.5K tokens — repeated technical paragraph
        ("NVIDIA's Grace Blackwell Superchip combines a 20-core ARM CPU with a Blackwell GPU "
         "die via NVLink-C2C, sharing 128GB of LPDDR5X unified memory at 273 GB/s. "
         "The compute capability is sm_121, distinct from the datacenter GB200 (sm_100) "
         "and consumer RTX 5090 (sm_120). This unified memory architecture eliminates "
         "PCIe bottlenecks but introduces a shared memory budget for weights, KV cache, "
         "and OS buffers. Disaggregated serving on this hardware requires careful "
         "MAX_MODEL_LEN tuning to avoid GPU allocator OOM errors at the kernel level. ") * 110
        + "Summarize the key architectural trade-offs described above.",
    ],
}


# ── Core benchmark logic ─────────────────────────────────────────────────────

def send_request(model, prompt, max_tokens, endpoint):
    """Send a single streaming chat completion request. Returns (ttft_ms, tokens, itl_avg_ms, total_ms)."""
    url = f"{endpoint}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})

    t_start = time.monotonic()
    ttft_ms = None
    token_times = []

    with urllib.request.urlopen(req, timeout=300) as resp:
        buffer = b""
        while True:
            chunk = resp.read(512)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        t_now = time.monotonic()
                        if ttft_ms is None:
                            ttft_ms = (t_now - t_start) * 1000
                        token_times.append(t_now)
                except Exception:
                    pass

    total_ms = (time.monotonic() - t_start) * 1000
    tokens = len(token_times)

    if tokens > 1:
        gaps = [(token_times[i] - token_times[i - 1]) * 1000
                for i in range(1, len(token_times))]
        itl_avg = statistics.mean(gaps)
    else:
        itl_avg = 0.0

    tps = tokens / (total_ms / 1000) if total_ms > 0 else 0.0
    return ttft_ms or 0.0, tokens, tps, itl_avg, total_ms


def run_benchmark(model, num_requests, prompt_len, tag, max_tokens, concurrency, warmup, endpoint):
    prompts = PROMPTS[prompt_len]
    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    print()
    print("=" * 60)
    print("  DYNAMO LAB BENCHMARK")
    print("=" * 60)
    print(f"  Model:       {model}")
    print(f"  Tag:         {tag}")
    print(f"  Requests:    {num_requests}")
    print(f"  Prompt:      {prompt_len}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Endpoint:    {endpoint}")
    print("=" * 60)

    # Health check
    print("  Checking endpoint...")
    try:
        urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10)
    except Exception as e:
        print(f"  ERROR: endpoint not reachable — {e}")
        return

    # Warmup
    if warmup:
        print("  Warming up (1 request)...")
        try:
            send_request(model, "Hello", 10, endpoint)
            print("  Warmup done.")
        except Exception as e:
            print(f"  Warmup failed: {e}")

    # Run requests
    print(f"  Running {num_requests} requests (concurrency={concurrency})...")

    results = [None] * num_requests
    errors = 0
    lock = threading.Lock()

    header = f"  {'#':>3}  {'TTFT':>8}  {'Tokens':>7}  {'tok/s':>7}  {'ITL avg':>8}  {'Total':>8}  Prompt"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def run_one(idx):
        nonlocal errors
        prompt = request_prompts[idx]
        try:
            ttft, tokens, tps, itl, total = send_request(model, prompt, max_tokens, endpoint)
            results[idx] = {
                "ttft_ms": round(ttft, 1),
                "total_ms": round(total, 1),
                "tokens": tokens,
                "tps": round(tps, 1),
                "itl_avg_ms": round(itl, 1),
                "prompt": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            }
            with lock:
                print(f"  {idx+1:>3}  {ttft:>7.0f}ms  {tokens:>7}  {tps:>6.1f}/s  "
                      f"{itl:>7.1f}ms  {total:>7.0f}ms  {prompt[:40]}")
        except Exception as e:
            with lock:
                errors += 1
                print(f"  {idx+1:>3}  ERROR: {e}")

    # Dispatch with concurrency
    threads = []
    sem = threading.Semaphore(concurrency)

    def worker(idx):
        with sem:
            run_one(idx)

    for i in range(num_requests):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Summarize
    valid = [r for r in results if r is not None]
    success = len(valid)

    if not valid:
        print("  No successful requests.")
        return

    ttfts = [r["ttft_ms"] for r in valid]
    tpss = [r["tps"] for r in valid]
    itls = [r["itl_avg_ms"] for r in valid]

    ttft_avg = statistics.mean(ttfts)
    ttft_p50 = statistics.median(ttfts)
    ttft_p90 = sorted(ttfts)[int(len(ttfts) * 0.9)]
    ttft_min = min(ttfts)
    tps_avg = statistics.mean(tpss)
    tps_max = max(tpss)
    itl_avg = statistics.mean(itls)
    itl_p90 = sorted(itls)[int(len(itls) * 0.9)]

    print()
    print("=" * 60)
    print(f"  RESULTS SUMMARY  ({success} successful / {num_requests} total)")
    print("=" * 60)
    print(f"  TTFT:       avg={ttft_avg:>8.0f}ms  p50={ttft_p50:>8.0f}ms  p90={ttft_p90:>8.0f}ms  min={ttft_min:>8.0f}ms")
    print(f"  Throughput: avg={tps_avg:>7.1f}/s  max={tps_max:>7.1f}/s")
    print(f"  ITL avg:    avg={itl_avg:>8.1f}ms  p90={itl_p90:>8.1f}ms")
    print("=" * 60)

    # Save results
    os.makedirs("results", exist_ok=True)
    model_short = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{model_short}_{tag}_{prompt_len}_c{concurrency}_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "model": model,
        "tag": tag,
        "prompt_len": prompt_len,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "endpoint": endpoint,
        "summary": {
            "ttft_avg_ms": round(ttft_avg, 1),
            "ttft_p50_ms": round(ttft_p50, 1),
            "ttft_p90_ms": round(ttft_p90, 1),
            "ttft_min_ms": round(ttft_min, 1),
            "tps_avg": round(tps_avg, 2),
            "tps_max": round(tps_max, 2),
            "itl_avg_ms": round(itl_avg, 1),
            "itl_p90_ms": round(itl_p90, 1),
            "errors": errors,
        },
        "requests": valid,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {filename}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGX Spark Dynamo benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--prompt-len", default="medium",
                        choices=["short", "medium", "long", "xlarge", "xxlarge"])
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tag", default="disaggregated",
                        help="Run label: disaggregated / single-node")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()

    run_benchmark(
        model=args.model,
        num_requests=args.requests,
        prompt_len=args.prompt_len,
        tag=args.tag,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        warmup=not args.no_warmup,
        endpoint=args.endpoint,
    )
