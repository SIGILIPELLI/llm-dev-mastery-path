---
description: "Latency Engineering & Batching — Cost optimization (module 6) asks 'how much does this request cost?' Latency engineering asks the companion question…"
---

# 07 · Latency Engineering & Batching

Cost optimization (module 6) asks "how much does this request cost?"
Latency engineering asks the companion question: "how fast can this
request return, and how do we keep the *worst* requests fast, not just
the average one?" At scale, the p99 latency (the slowest 1% of requests)
is usually the number that determines whether users perceive your
product as fast or frustrating — and it behaves very differently from
the average.

## Why average latency is the wrong metric

A system with 100ms average latency and a 5-second p99 has a real
problem that the average completely hides: 1% of users are waiting 50x
longer than everyone else, and at scale, 1% of a large request volume is
a lot of actual frustrated users.

```python
import statistics

def latency_report(latencies_ms: list[float]) -> dict:
    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)
    return {
        "p50": sorted_latencies[int(n * 0.50)],
        "p90": sorted_latencies[int(n * 0.90)],
        "p99": sorted_latencies[int(n * 0.99)],
        "mean": statistics.mean(latencies_ms),
        "max": max(latencies_ms),
    }
```

Set latency budgets and alerts on p99 (or even p99.9 for high-volume
systems), not on the mean — the mean is dominated by the common case and
will look fine right up until the tail latency becomes a real user-facing
problem.

## Streaming-first design

The single highest-leverage latency change for perceived responsiveness
is streaming tokens as they're generated rather than waiting for the full
response — time-to-first-token (TTFT) matters more to perceived speed
than total generation time, because users start reading immediately:

```python
def stream_response(client, messages: list[dict]):
    first_token_time = None
    start = time.monotonic()
    for chunk in client.stream(messages):
        if first_token_time is None:
            first_token_time = time.monotonic() - start
            record_metric("ttft_ms", first_token_time * 1000)
        yield chunk.text

# Application code renders each chunk as it arrives instead of blocking
# on the full response — this is why chat UIs feel fast even when total
# generation takes several seconds.
```

Design every user-facing LLM feature to consume a stream by default.
Non-streaming ("wait for the whole response, then show it") should be the
exception, reserved for cases that genuinely need the full response
before doing anything with it (e.g., parsing structured output that isn't
valid until complete).

## Request batching for throughput

Server-side, batching multiple concurrent requests into a single forward
pass through the model dramatically improves GPU utilization (this is
what serving engines like vLLM, covered in Level 3 module 5, do
automatically) — but application-level batching decisions still matter
for workloads you control:

```python
import asyncio
from collections import deque

class RequestBatcher:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: float = 20):
        self.queue: deque = deque()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def add_request(self, request) -> "asyncio.Future":
        future = asyncio.get_event_loop().create_future()
        self.queue.append((request, future))
        return future

    async def batch_loop(self, process_batch_fn):
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)
            if not self.queue:
                continue
            batch = []
            while self.queue and len(batch) < self.max_batch_size:
                batch.append(self.queue.popleft())
            requests = [item[0] for item in batch]
            results = await process_batch_fn(requests)
            for (_, future), result in zip(batch, results):
                future.set_result(result)
```

The `max_wait_ms` parameter is a direct latency/throughput tradeoff:
waiting longer collects a bigger batch (better throughput, better GPU
utilization) but adds fixed latency to every request in the batch. Tune
it against your actual p99 budget, not just theoretical throughput gains.

## Speculative decoding

Speculative decoding speeds up generation by using a small, fast "draft"
model to guess several tokens ahead, then having the large target model
verify all of them in a single forward pass instead of generating them
one at a time:

```python
def speculative_decode_step(draft_model, target_model, context: list[int], num_speculative: int = 4):
    # draft model proposes several tokens cheaply and quickly
    draft_tokens = draft_model.generate(context, max_new_tokens=num_speculative)

    # target model verifies all draft tokens in ONE forward pass
    target_logits = target_model.forward(context + draft_tokens)

    accepted = []
    for i, draft_token in enumerate(draft_tokens):
        target_token = sample(target_logits[i])
        if target_token == draft_token:
            accepted.append(draft_token)   # draft guessed correctly — keep it
        else:
            accepted.append(target_token)  # target disagrees — use its token, stop here
            break
    return accepted
```

When the draft model's guesses are frequently correct (common for
predictable continuations — code completion, formulaic text), this
produces several tokens per target-model forward pass instead of one,
directly cutting wall-clock generation time without changing the target
model's actual output distribution.

## Setting and enforcing latency budgets

```python
@dataclass
class LatencyBudget:
    ttft_ms: float
    total_ms: float

def enforce_budget(request, budget: LatencyBudget, fast_model, capable_model):
    # if the capable model is currently running hot (based on recent p99),
    # fall back to a faster model rather than blow the budget
    if get_recent_p99("capable_model") > budget.total_ms:
        return fast_model.generate(request)
    return capable_model.generate(request)
```

Tie latency budgets to the routing layer from module 3 — a request that
would blow its budget on the primary model is a legitimate reason to
route to a faster fallback, the same way a provider outage is.

## How It Actually Works

Tail latency behaves differently from average latency because of a
statistical effect: when a request depends on several independent steps
(queueing, network, model inference, post-processing), the *overall*
latency is dominated by whichever step happens to be slow on that
particular request — and across many requests, some fraction will hit an
unlucky combination of slow steps even if each step is fast on average.
This is why p99 latency degrades faster than p50 as you add more
sequential dependent steps to a pipeline: each added step is another
chance for that particular request to draw a slow outcome, compounding
multiplicatively rather than just adding its own average latency.

Speculative decoding's speedup comes from a specific asymmetry in
transformer inference cost: generating one token autoregressively
requires a full forward pass through every layer, and forward-pass cost
barely increases when processing several tokens in parallel versus one
(the same matrix multiplications, larger batch dimension) — so verifying
four draft tokens in one forward pass costs roughly the same as
generating one token normally, but produces up to four accepted tokens
when the draft model guesses well. The net speedup is bounded by draft
accuracy: a draft model that's rarely right just wastes the verification
pass, which is why the draft model is chosen specifically to have a
similar output distribution to the target model on the kind of text
actually being generated.

## Cheat sheet

| Concept | Key fact |
|---|---|
| p99 over mean | Tail latency reflects real worst-case user experience; mean hides it |
| Time-to-first-token | Matters more than total time for perceived responsiveness |
| Streaming-first | Default every user-facing feature to stream, not wait-then-return |
| Request batching | Trade a small wait for much better throughput/GPU utilization |
| Speculative decoding | Draft model guesses tokens; target model verifies in one pass |
| Latency budget routing | Fall back to a faster model when the primary is trending slow |

## Exercise

Instrument a mock endpoint with the `latency_report` function above,
generate 200 simulated requests with realistic variance (most fast, a
long tail of slow ones), and compute p50/p90/p99. Then implement
`RequestBatcher` with `max_wait_ms` at 5ms, 20ms, and 100ms, measure
resulting p99 latency and throughput at each setting, and pick the
setting that best balances both for a chat-style interactive workload
versus a bulk-processing workload.
