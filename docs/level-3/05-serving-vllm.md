---
description: "Serving Open Models with vLLM — Ollama (module 4) is built for single-user, local convenience. Serving an open-weight model to many concurrent users with…"
---

# 05 · Serving Open Models with vLLM

Ollama (module 4) is built for single-user, local convenience. Serving an
open-weight model to many concurrent users with production-grade
throughput needs a different tool: **vLLM**, an inference server built
around two techniques — PagedAttention and continuous batching — that
make it dramatically more efficient at high concurrency than a naive
serving loop.

## Starting a vLLM server

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 8192
```

This starts an **OpenAI-compatible** HTTP server — the same request/response
shape as the hosted OpenAI API, which is why most client code needs only a
`base_url` change:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain PagedAttention in two sentences."}],
)
print(resp.choices[0].message.content)
```

## PagedAttention

Every request's KV cache (module 1) needs contiguous memory in naive
implementations, sized for the request's *maximum possible* length —
wasteful, because most requests finish well short of the max, and that
reserved memory can't be reused by other requests in the meantime.
PagedAttention borrows the idea of OS virtual memory paging: the KV cache
is split into fixed-size blocks allocated on demand, non-contiguously, and
mapped through a lightweight table — so memory is only consumed as tokens
are actually generated, and can be shared or freed per-block:

```python
# Conceptual sketch — not vLLM's actual internals, but the model to reason from
class PagedKVCache:
    def __init__(self, block_size: int = 16, num_blocks: int = 1000):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_table: dict[str, list[int]] = {}   # request_id -> block ids

    def allocate_block(self, request_id: str) -> int:
        block_id = self.free_blocks.pop()
        self.block_table.setdefault(request_id, []).append(block_id)
        return block_id

    def free(self, request_id: str) -> None:
        for block_id in self.block_table.pop(request_id, []):
            self.free_blocks.append(block_id)
```

The practical effect: far more concurrent requests fit in the same GPU
memory than a naive contiguous-allocation scheme, because you're no longer
reserving worst-case memory for every in-flight request up front.

## Continuous batching

A naive server batches requests that arrive together and waits for the
*whole batch* to finish before accepting new ones — one slow request in a
batch stalls throughput for every other request in it. Continuous batching
instead adds and removes individual requests from the running batch at
every generation step, so a request that finishes early is immediately
replaced by a new one rather than leaving that GPU capacity idle:

```python
# Conceptual sketch of the scheduling loop continuous batching implements
def serve_loop(pending_requests: list, running_batch: list, max_batch_size: int):
    while pending_requests or running_batch:
        while pending_requests and len(running_batch) < max_batch_size:
            running_batch.append(pending_requests.pop(0))

        step_all(running_batch)   # generate exactly one token for every request in the batch

        finished = [r for r in running_batch if r.is_done()]
        for r in finished:
            running_batch.remove(r)
            return_to_client(r)
```

This is the single biggest lever for aggregate throughput under real,
bursty traffic — requests arrive at different times and finish at
different lengths, and continuous batching keeps the GPU's batch full of
*active* work at every single generation step instead of at coarse batch
boundaries.

## Throughput tuning

Key levers, roughly in order of impact:

- **`--gpu-memory-utilization`** — how much GPU memory vLLM reserves for
  the KV cache pool; higher generally means more concurrent requests fit,
  up to your hardware's limit.
- **`--max-num-seqs`** — cap on how many requests run concurrently in one
  batch; too high can thrash memory, too low leaves throughput on the
  table.
- **Quantization** (module 7) — a quantized model needs less memory per
  weight, freeing more room for the KV cache pool and therefore more
  concurrent capacity.
- **Prefix caching** — vLLM can reuse KV cache across requests sharing an
  identical prompt prefix, the open-source-serving equivalent of Level 2's
  hosted prompt caching, enabled with `--enable-prefix-caching`.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --enable-prefix-caching
```

Measure before and after any tuning change with a real load test rather
than guessing — throughput and latency trade off against each other, and
the right setting depends on your traffic pattern (many short requests vs.
few long ones).

## How It Actually Works

vLLM doesn't change the model's computation from module 1 — it's serving
infrastructure around the identical forward-pass mechanism, optimized for
the specific bottleneck that matters at high concurrency: GPU memory
occupied by KV caches. Every in-flight request needs its own growing KV
cache (one key/value vector pair per token, per layer, per attention
head), and naive serving either wastes memory reserving worst-case space
per request or serializes requests to avoid that waste — both leave
expensive GPU compute idle.

PagedAttention's block-based allocation is a memory-management technique,
not a change to attention's mathematics — attention still computes the
same weighted average over keys and values described in module 1; the
paging only affects *where in physical memory* those keys and values are
stored and how they're addressed, transparently to the math. Continuous
batching similarly doesn't change what any individual request's forward
pass computes — it changes the *scheduling* of whose tokens get computed
in which GPU step, maximizing the fraction of every batched matrix
multiplication that's doing useful work for some active request rather
than padding for a request that already finished or hasn't arrived yet.

This is why both techniques compose so well together and why they matter
specifically at scale: a single local request (module 4's Ollama use case)
never contends for shared GPU memory or scheduling slots with other
requests, so neither optimization has anything to bite on — they exist
precisely for the multi-tenant, bursty-arrival regime a production serving
layer has to handle.

## Cheat sheet

| Concept | Key fact |
|---|---|
| vLLM | OpenAI-compatible inference server for open-weight models |
| PagedAttention | Block-based, on-demand KV cache allocation — less wasted GPU memory |
| Continuous batching | Requests join/leave the running batch per-step, not per-batch |
| `--gpu-memory-utilization` | KV cache pool size — biggest lever on concurrent capacity |
| `--enable-prefix-caching` | Self-hosted equivalent of hosted prompt caching |
| Load testing | Always measure real throughput/latency before tuning further |

## Exercise

Start a vLLM server locally (or on a rented GPU instance) with a small
open model, and load-test it with 50 concurrent requests of varying prompt
lengths using a simple `asyncio` client. Record p50/p95 latency and total
throughput (tokens/sec) with `--max-num-seqs` set to 8, then again at 64.
Explain, using PagedAttention and continuous batching, why throughput does
or doesn't scale linearly between the two settings.
