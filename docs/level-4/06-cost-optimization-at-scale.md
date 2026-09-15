---
description: "Cost Optimization at Scale — At prototype scale, LLM API cost is a rounding error. At platform scale — millions of requests across many teams — it becomes…"
---

# 06 · Cost Optimization at Scale

At prototype scale, LLM API cost is a rounding error. At platform scale
— millions of requests across many teams — it becomes one of the largest
line items in the infrastructure budget, and small per-request savings
compound into large absolute numbers. This module covers the concrete
levers for reducing cost without degrading quality: model tiering, batch
processing, caching, and cost attribution so the right teams see and own
their spend.

## Model tiering: matching model cost to task difficulty

Not every request needs the most capable (and most expensive) model.
Tiering routes requests to the cheapest model that reliably handles them,
escalating only when needed:

```python
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    SMALL = "small"     # cheap, fast, good for classification/extraction/simple tasks
    MEDIUM = "medium"    # balanced, general-purpose default
    LARGE = "large"      # expensive, reserved for genuinely complex reasoning

TIER_COST_PER_1K_TOKENS = {ModelTier.SMALL: 0.0002, ModelTier.MEDIUM: 0.003, ModelTier.LARGE: 0.015}

def select_tier(task_type: str, input_length: int) -> ModelTier:
    if task_type in ("classification", "extraction", "simple_qa"):
        return ModelTier.SMALL
    if task_type in ("complex_reasoning", "code_generation") or input_length > 8000:
        return ModelTier.LARGE
    return ModelTier.MEDIUM
```

A more robust version escalates dynamically based on observed confidence
rather than a static task-type mapping:

```python
def tiered_call_with_escalation(prompt: str, small_model, large_model, confidence_threshold: float = 0.7):
    result = small_model.generate_with_confidence(prompt)
    if result.confidence >= confidence_threshold:
        return result.text, ModelTier.SMALL
    # small model wasn't confident — escalate to the larger model
    return large_model.generate(prompt), ModelTier.LARGE
```

This pattern — try cheap first, escalate on low confidence — routinely
cuts aggregate cost substantially versus routing everything to the large
model by default, because most real traffic is disproportionately simple
requests that a small model handles just as well.

## Batch APIs for non-interactive workloads

Anything that doesn't need a response within the same interaction —
nightly summarization, bulk classification, offline eval scoring (module
5) — is a candidate for batch processing, which most providers price at
a substantial discount versus real-time calls in exchange for
higher-latency (often hours, not seconds) turnaround:

```python
import json

def build_batch_request(items: list[dict]) -> str:
    lines = []
    for item in items:
        lines.append(json.dumps({
            "custom_id": item["id"],
            "params": {"messages": item["messages"], "max_tokens": item.get("max_tokens", 512)},
        }))
    return "\n".join(lines)   # provider-specific JSONL batch format

def submit_batch(batch_jsonl: str, batch_client) -> str:
    batch_job = batch_client.create(input_data=batch_jsonl)
    return batch_job.id

def poll_and_collect(batch_id: str, batch_client) -> dict[str, str]:
    job = batch_client.retrieve(batch_id)
    if job.status != "completed":
        raise RuntimeError(f"Batch {batch_id} not ready: {job.status}")
    return {result.custom_id: result.output for result in job.results}
```

Route any workload that can tolerate hours-not-seconds latency through
batch — it's one of the highest-leverage cost changes available with
zero quality tradeoff, since it's the same model, just priced and
scheduled differently.

## Prompt and response caching

Repeated or near-identical requests (a common system prompt, a
frequently-asked question, a re-run eval) don't need to pay full
inference cost every time:

```python
import hashlib

class ResponseCache:
    def __init__(self, store):
        self.store = store   # any key-value store (Redis, etc.)

    def cache_key(self, messages: list[dict], model: str) -> str:
        payload = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_or_call(self, messages: list[dict], model: str, call_fn, ttl_seconds: int = 3600) -> str:
        key = self.cache_key(messages, model)
        cached = self.store.get(key)
        if cached is not None:
            return cached
        response = call_fn(messages)
        self.store.setex(key, ttl_seconds, response)
        return response
```

Exact-match caching only helps identical requests. For a stable system
prompt or shared document context reused across many calls, provider-side
prompt caching (caching the processed representation of a prompt prefix
across calls) cuts cost on the repeated portion without requiring
identical full requests — check current provider documentation for exact
mechanics and pricing, since implementations vary and evolve.

## Cost attribution across teams

Centralized spend without per-team visibility means no one is
accountable for their own usage. Tag every request at the gateway
(module 3) with the calling team, and aggregate:

```python
from collections import defaultdict

@dataclass
class UsageRecord:
    team_id: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float

def compute_cost(record: UsageRecord, pricing: dict) -> float:
    rates = pricing[record.model]   # {"input": $/1k tokens, "output": $/1k tokens}
    return (record.input_tokens / 1000 * rates["input"]) + (record.output_tokens / 1000 * rates["output"])

def aggregate_by_team(records: list[UsageRecord], pricing: dict) -> dict[str, float]:
    totals = defaultdict(float)
    for record in records:
        totals[record.team_id] += compute_cost(record, pricing)
    return dict(totals)
```

Feed this into the same dashboard as the quality monitoring from module
5 — cost and quality should be reviewed together, since the cheapest
option that also fails quality checks isn't actually cheaper once you
count the cost of the resulting incidents or rework.

## How It Actually Works

Tiering and escalation work because task difficulty in real traffic is
heavily right-skewed: most requests are easy, a small fraction are hard,
and a fixed policy of always using the most capable model pays the
hard-task price on every easy task too. Confidence-based escalation
approximates an oracle router (one that always picks the cheapest
sufficient model) without needing to know task difficulty in advance —
the small model's own uncertainty signal (e.g., low top-token
probability, or an explicit self-rated confidence) serves as a proxy for
"this request is probably harder than what I reliably handle," which
correlates with actual difficulty well enough to be useful even though
it's an imperfect signal.

Batch pricing discounts exist because they change what the provider is
selling: real-time inference must reserve capacity to answer instantly,
which caps how much a given GPU cluster can serve; batch requests can be
scheduled into idle capacity between real-time spikes, so the same
hardware serves more aggregate work over a day. That's a genuine
capacity-utilization efficiency, not just a discount — which is why
batch latency (however long the provider takes to schedule your job into
spare capacity) is the tradeoff, not a quality difference in the
response itself.

## Cheat sheet

| Lever | Mechanism | Best for |
|---|---|---|
| Model tiering | Route to cheapest sufficient model, escalate on low confidence | Mixed-difficulty traffic |
| Batch API | Trade latency (hours) for a substantial per-token discount | Non-interactive, bulk workloads |
| Exact-match caching | Skip inference entirely on identical repeated requests | FAQ-style, repeated queries |
| Prompt caching | Reuse processed prefix across calls with shared context | Long, stable system prompts/docs |
| Cost attribution | Tag every request with team/caller at the gateway | Accountability, budget alerts |

## Exercise

Build the `tiered_call_with_escalation` function against a real small +
large model pair, run it on 50 mixed-difficulty prompts (mostly simple,
a few genuinely hard), and measure: (a) total cost versus always using
the large model, (b) how often escalation actually triggered, and (c)
whether any escalated-but-still-wrong cases suggest the confidence
threshold needs tuning. Report the cost savings percentage and whether
quality (spot-checked against always-large-model answers) held up.
