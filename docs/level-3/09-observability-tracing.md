---
description: "Observability & Tracing — By this point your system might have chained calls (Level 2, module 1), subagents (Level 2, module 8), tool use, and retrieval …"
---

# 09 · Observability & Tracing

By this point your system might have chained calls (Level 2, module 1),
subagents (Level 2, module 8), tool use, and retrieval — a single user
request can trigger a dozen model calls across several services. Without
tracing, debugging "why did the agent do that" means reading scattered
print statements after the fact. This module builds real tracing, cost/
token dashboards, and a debugging workflow for production conversations.

## Tracing a single request end-to-end

A trace ties every LLM call, tool execution, and retrieval step within one
user request together under a shared trace ID, with timing and metadata
for each span:

```python
import time, uuid, json
from contextlib import contextmanager

class Tracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def span(self, name: str, **metadata):
        span_id = str(uuid.uuid4())
        start = time.monotonic()
        record = {"span_id": span_id, "name": name, "metadata": metadata, "error": None}
        try:
            yield record
        except Exception as e:
            record["error"] = str(e)
            raise
        finally:
            record["duration_ms"] = round((time.monotonic() - start) * 1000, 1)
            self.spans.append(record)

    def export(self, trace_id: str) -> None:
        with open("traces.jsonl", "a") as f:
            f.write(json.dumps({"trace_id": trace_id, "spans": self.spans}) + "\n")
```

Wrap every model call and tool execution in a span:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

def traced_agent_turn(task: str, tracer: Tracer) -> str:
    with tracer.span("llm_call", model=MODEL) as span:
        resp = client.messages.create(model=MODEL, max_tokens=800, tools=TOOLS,
                                       messages=[{"role": "user", "content": task}])
        span["metadata"]["input_tokens"] = resp.usage.input_tokens
        span["metadata"]["output_tokens"] = resp.usage.output_tokens

    for block in resp.content:
        if block.type == "tool_use":
            with tracer.span("tool_call", tool_name=block.name, args=block.input) as span:
                result = execute_tool(block.name, block.input)
                span["metadata"]["result_preview"] = str(result)[:200]

    return resp.content[0].text

tracer = Tracer()
trace_id = str(uuid.uuid4())
output = traced_agent_turn("What's 15% of 240, and is that above 30?", tracer)
tracer.export(trace_id)
```

Every span records what happened, how long it took, and whether it
errored — enough to reconstruct the full decision path of a specific
request after the fact, without having to reproduce the bug live.

## Token and cost dashboards

Aggregate `usage` data (from every module that's referenced it since
Level 1's module 8) across requests into metrics you can chart over time,
not just check one call at a time:

```python
import sqlite3

def log_usage(trace_id: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    conn = sqlite3.connect("usage.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS usage
                     (trace_id TEXT, model TEXT, input_tokens INT, output_tokens INT,
                      cost_usd REAL, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                  (trace_id, model, input_tokens, output_tokens, cost_usd))
    conn.commit()

def daily_cost_summary() -> list[tuple]:
    conn = sqlite3.connect("usage.db")
    return conn.execute("""
        SELECT date(ts), model, SUM(cost_usd), SUM(input_tokens + output_tokens)
        FROM usage GROUP BY date(ts), model ORDER BY date(ts) DESC
    """).fetchall()
```

Feed this into any dashboarding tool (Grafana, a simple internal page) to
watch for cost spikes, and alert when daily spend crosses a threshold —
the same guardrail spirit as Level 1's per-request cost tracker, extended
to fleet-wide visibility.

## Debugging a production conversation

When a user reports "the agent gave a wrong answer," a good trace lets you
answer three questions without re-running anything live:

1. **What did the model actually see?** — log the full `messages` payload
   per call (redacting PII as needed), not just a summary.
2. **What did it decide to do?** — every `tool_use` block and its
   arguments, in order.
3. **What did each tool actually return?** — the raw tool result, not just
   whether it "succeeded."

```python
def replay_trace(trace_id: str) -> None:
    for line in open("traces.jsonl"):
        record = json.loads(line)
        if record["trace_id"] != trace_id:
            continue
        for span in record["spans"]:
            status = "ERROR" if span["error"] else "ok"
            print(f"[{span['duration_ms']:>7.1f}ms] {span['name']:15s} {status}  {span['metadata']}")
```

Once you can print this for any historical `trace_id`, "reproduce the bug"
often becomes unnecessary — you can see exactly what happened the first
time.

## Integrating with existing observability tools

Purpose-built LLM observability platforms (e.g. LangSmith, Langfuse,
Arize Phoenix) provide this tracing model with less boilerplate,
automatic token/cost aggregation, and a UI for browsing traces — typically
via a decorator or context manager wrapping your existing calls:

```python
# Illustrative — exact API varies by platform
from langfuse.decorators import observe

@observe()
def traced_agent_turn(task: str) -> str:
    ...   # unchanged body — the decorator captures spans automatically
```

Whether you build tracing yourself (as above) or adopt a platform, the
underlying data model — trace, span, metadata, timing — is the same; pick
based on how much UI/aggregation you need versus how much control over
data residency you require.

## How It Actually Works

Tracing adds no new capability to the model — it's instrumentation around
calls you were already making, capturing the same `usage` fields and
`tool_use`/`tool_result` structure that every earlier module already
exposed. The value entirely comes from *persisting and correlating* that
information across many calls and across time, rather than letting each
call's metadata disappear once your process moves to the next line.

This is precisely why traces are effective for debugging non-deterministic
systems: because a model's output is influenced by many aspects of a call
(the exact prompt, model version, tool results at that moment,
temperature/sampling settings), the only reliable way to understand a
specific past decision is to have recorded the *exact input* that produced
it — recreating "roughly the same" conditions and re-running live is not
guaranteed to reproduce the same output even against the same model, since
generation is stochastic. A trace sidesteps that by recording ground truth
once, rather than depending on reproducibility.

Cost dashboards work by the same aggregation logic as any other metrics
pipeline — summing structured `usage` records over time windows — the
LLM-specific detail is only that "usage" here means input/output tokens
rather than, say, database queries, and that token-to-cost conversion
depends on the specific model and rate in effect for each logged call.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Trace | All spans (LLM calls, tool executions) for one user request, tied by trace ID |
| Span | One unit of work with timing, metadata, and error status |
| Token/cost dashboard | Aggregated `usage` data over time, alertable on spend spikes |
| Debugging via trace | Answer "what did it see / decide / get back" without live reproduction |
| Observability platforms | Same data model as hand-rolled tracing, with less boilerplate |

## 🔀 Related lessons on other tracks

- [Azure — 05 · Advanced Observability (App Insights, Tracing)](https://sigilipelli.github.io/azure-mastery-path/level-4/05-advanced-observability/)
- [RAG — 09 · Observability & Tracing](https://sigilipelli.github.io/rag-mastery-path/level-3/09-observability/)

## Exercise

Add the `Tracer` class above to Level 2's planner/executor agent (module
7), tracing the planner call, each executor step, and the final reflection
call as separate spans under one trace ID. Export 5 traces from different
tasks to `traces.jsonl`, then write `replay_trace(trace_id)` output for
each and confirm you can fully explain what happened in each run from the
trace alone, without re-running the agent.
