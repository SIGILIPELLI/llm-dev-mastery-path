---
description: "Project — Self-Hosted LLM Stack — This capstone assembles Level 3 into one running stack: a local/served open model, a serving layer built for real…"
---

# 10 · Project — Self-Hosted LLM Stack

This capstone assembles Level 3 into one running stack: a local/served
open model, a serving layer built for real concurrency, guardrails on
input and output, and observability tying it all together — a complete
alternative to calling a hosted API, end to end.

## Architecture

```
Client ──► FastAPI gateway ──► Input guardrail (module 8)
                 │
                 ▼
        vLLM server (module 5) ──► Open model (quantized, module 7)
                 │
                 ▼
        Output guardrail (module 8)
                 │
                 ▼
        Tracer + usage log (module 9) ──► traces.jsonl / usage.db
```

## Serving layer

Start vLLM with prefix caching and quantization for a good
throughput/quality balance on modest hardware:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3-8B-Instruct-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --port 8000
```

## The gateway

Combines guardrails and tracing around every request, presenting one clean
API to callers regardless of what's happening underneath:

```python
# gateway.py
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import uuid, sqlite3, json, time

app = FastAPI()
llm = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
MODEL = "TheBloke/Llama-3-8B-Instruct-AWQ"

class ChatRequest(BaseModel):
    message: str

def check_policy(text: str) -> dict:
    resp = llm.chat.completions.create(
        model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content":
            f"Reply with exactly one word, 'BLOCK' or 'ALLOW', for whether "
            f"this text requests something harmful or illegal:\n\n{text}"}],
    )
    return {"blocked": "BLOCK" in resp.choices[0].message.content.upper()}

def log_trace(trace_id: str, spans: list[dict]) -> None:
    with open("traces.jsonl", "a") as f:
        f.write(json.dumps({"trace_id": trace_id, "spans": spans}) + "\n")

@app.post("/chat")
def chat(req: ChatRequest):
    trace_id = str(uuid.uuid4())
    spans = []

    t0 = time.monotonic()
    input_check = check_policy(req.message)
    spans.append({"name": "input_guardrail", "duration_ms": (time.monotonic() - t0) * 1000, "result": input_check})
    if input_check["blocked"]:
        log_trace(trace_id, spans)
        return {"reply": "I can't help with that.", "trace_id": trace_id}

    t0 = time.monotonic()
    resp = llm.chat.completions.create(model=MODEL, max_tokens=600, messages=[{"role": "user", "content": req.message}])
    reply = resp.choices[0].message.content
    spans.append({"name": "llm_call", "duration_ms": (time.monotonic() - t0) * 1000,
                  "tokens": resp.usage.total_tokens})

    t0 = time.monotonic()
    output_check = check_policy(reply)
    spans.append({"name": "output_guardrail", "duration_ms": (time.monotonic() - t0) * 1000, "result": output_check})
    if output_check["blocked"]:
        log_trace(trace_id, spans)
        return {"reply": "I'm not able to share that response.", "trace_id": trace_id}

    log_trace(trace_id, spans)
    return {"reply": reply, "trace_id": trace_id}
```

## Usage tracking

```python
# usage.py
import sqlite3

def init_db():
    conn = sqlite3.connect("usage.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS usage
                     (trace_id TEXT, tokens INT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()

def log_usage(trace_id: str, tokens: int) -> None:
    conn = sqlite3.connect("usage.db")
    conn.execute("INSERT INTO usage VALUES (?, ?, CURRENT_TIMESTAMP)", (trace_id, tokens))
    conn.commit()
```

Since a self-hosted stack has no per-token API bill, "cost" here is
compute/GPU-time — track total tokens and requests per day as a proxy for
capacity planning (when do you need a second GPU) rather than a literal
invoice.

## Running it end to end

```bash
# Terminal 1: serving layer
python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-3-8B-Instruct-AWQ --quantization awq --port 8000

# Terminal 2: gateway
uvicorn gateway:app --port 9000

# Terminal 3: try it
curl -X POST http://localhost:9000/chat -H "Content-Type: application/json" \
     -d '{"message": "Explain what a self-hosted LLM stack is."}'
```

Inspect `traces.jsonl` afterward to see the full span breakdown — input
guardrail, main call, output guardrail, each with real latency — for that
exact request.

## Why self-host at all

Weigh this against a hosted API honestly before committing to the
operational burden:

| Factor | Self-hosted | Hosted API |
|---|---|---|
| Data residency | Full control — nothing leaves your infra | Data sent to provider |
| Cost at low volume | GPU cost even when idle | Pay only per token used |
| Cost at high, steady volume | Can be cheaper at scale | Scales linearly with usage |
| Model quality ceiling | Bounded by best available open weights | Access to frontier closed models |
| Operational burden | You own uptime, scaling, security patching | Provider owns it |

A hybrid is common in practice: sensitive or high-volume, latency-tolerant
workloads on a self-hosted stack; everything needing frontier
capability or minimal ops burden on a hosted API.

## How It Actually Works

Nothing in this capstone introduces a new mechanism — it is Level 3's five
components (a real transformer, a trained tokenizer, embeddings for any
retrieval step you add, a memory-efficient serving layer, a
quantized model) wired together behind the same guardrail and tracing
patterns Level 3's later modules built independently. The gateway's job is
purely orchestration: call the policy check, call the model, call the
policy check again, record what happened — every one of those calls
resolves to the identical stateless request→response cycle from Level 1,
now pointed at `localhost:8000` instead of a hosted provider's endpoint.

The properties that differ from a hosted setup are entirely about *who
operates* each piece: you now own the GPU provisioning that determines
throughput (module 5), the quantization tradeoff that determines
quality-per-dollar (module 7), and the uptime of every layer in the
diagram — none of which the model's own behavior changes one bit from how
it behaved when called through a hosted API in earlier levels.

## Cheat sheet

| Layer | Module it came from |
|---|---|
| Open model, quantized | Modules 4, 7 |
| vLLM serving | Module 5 |
| Input/output guardrails | Module 8 |
| Tracing + usage logging | Module 9 |
| Gateway orchestration | Plain FastAPI, no new mechanism |

## 🔀 Related lessons on other tracks

- [GitHub & Git — 09 · Self-Hosted Runners & Enterprise GitHub](https://sigilipelli.github.io/github-mastery-path/level-4/09-self-hosted-runners-enterprise/)

## Exercise

Stand up the full stack locally with a small AWQ-quantized model, and run
20 varied requests through the gateway (a mix of benign requests and a few
that should trigger the guardrail). Confirm every request produces a
trace with distinguishable spans, and that blocked requests never reach
the main model call (verify this from the trace, not just the response).
Then swap the guardrail's model for a smaller/faster one than the main
chat model and measure the latency improvement on the two guardrail spans.
