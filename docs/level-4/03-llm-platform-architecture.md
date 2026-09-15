---
description: "Production LLM Platform Architecture — A single API call to a single model provider works fine for a prototype. A production platform serving many teams…"
---

# 03 · Production LLM Platform Architecture

A single API call to a single model provider works fine for a prototype.
A production platform serving many teams and features needs something in
front of that: a layer that routes requests to the right model, falls
back when a provider is down or rate-limited, enforces quotas so one team
can't starve another, and normalizes differences between providers so
application code doesn't have to know which one is answering. This module
covers how to design that layer — the "LLM gateway" pattern that most
serious LLM platforms converge on.

## Why you need a gateway at all

Without one, every application team writes its own provider client, its
own retry logic, its own fallback handling, and its own cost tracking —
duplicated N times, inconsistently, with no central point to enforce
policy or see aggregate spend. A gateway centralizes:

- **Multi-provider abstraction** — one internal API shape, translated to
  each provider's actual request/response format.
- **Routing and fallback** — which model handles a request, and what
  happens when the primary choice fails or is too slow.
- **Quota and rate limiting** — per-team, per-API-key limits that prevent
  one caller from exhausting shared capacity or budget.
- **Observability** — every request logged in one place (Level 3, module
  9's tracing, applied at the platform level rather than per-app).

## A minimal gateway abstraction

```python
from dataclasses import dataclass
from enum import Enum
import time

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"

@dataclass
class ModelRequest:
    messages: list[dict]
    max_tokens: int = 1024
    team_id: str = "default"

@dataclass
class ModelResponse:
    text: str
    provider: Provider
    latency_ms: float
    input_tokens: int
    output_tokens: int

class ProviderAdapter:
    """Each provider gets one adapter translating the internal shape to its API."""
    def __init__(self, provider: Provider, client):
        self.provider = provider
        self.client = client

    def call(self, request: ModelRequest) -> ModelResponse:
        start = time.monotonic()
        # translate request.messages into this provider's exact schema here
        raw = self.client.send(request.messages, max_tokens=request.max_tokens)
        latency_ms = (time.monotonic() - start) * 1000
        return ModelResponse(
            text=raw["text"],
            provider=self.provider,
            latency_ms=latency_ms,
            input_tokens=raw["usage"]["input_tokens"],
            output_tokens=raw["usage"]["output_tokens"],
        )
```

Every consumer of the gateway works against `ModelRequest`/`ModelResponse`
— they never see provider-specific payloads, which is what makes it
possible to swap, add, or remove a provider without touching application
code.

## Routing and fallback chains

Routing decides which model answers a given request — by task type, cost
tier, or current health — and a fallback chain decides what happens when
the chosen model fails:

```python
class Gateway:
    def __init__(self, adapters: dict[Provider, ProviderAdapter], routing_table: dict[str, list[Provider]]):
        self.adapters = adapters
        self.routing_table = routing_table   # task_type -> ordered list of providers to try

    def route(self, task_type: str, request: ModelRequest) -> ModelResponse:
        chain = self.routing_table.get(task_type, [Provider.ANTHROPIC])
        last_error = None
        for provider in chain:
            try:
                return self.adapters[provider].call(request)
            except Exception as exc:
                last_error = exc
                continue   # try the next provider in the fallback chain
        raise RuntimeError(f"All providers in chain failed for {task_type}: {last_error}")
```

```python
routing_table = {
    "classification": [Provider.LOCAL, Provider.ANTHROPIC],   # cheap local model first
    "complex_reasoning": [Provider.ANTHROPIC, Provider.OPENAI], # primary, then fallback
    "summarization": [Provider.ANTHROPIC],
}
```

The fallback chain is what turns a single provider outage into a
degraded-but-working system instead of a full outage — as long as the
fallback model is genuinely capable of handling the request, not just
available. Route by task type deliberately: sending a task the fallback
model handles poorly is often worse than a clear, fast failure.

## Quota and rate limiting

A shared token-bucket limiter per team or API key prevents one caller
from consuming capacity meant for others:

```python
import threading
import time

class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_per_second = refill_per_second
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def try_consume(self, amount: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.last_refill = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

team_buckets: dict[str, TokenBucket] = {}

def check_quota(team_id: str, requests_per_minute: int = 60) -> bool:
    bucket = team_buckets.setdefault(team_id, TokenBucket(requests_per_minute, requests_per_minute / 60))
    return bucket.try_consume()
```

Enforce quota at the gateway, not in each application — it's the only
place that sees total cross-team demand, and it's the natural place to
reject or queue a request before it ever reaches (and gets billed by) an
upstream provider.

## Normalizing streaming and errors

Providers differ in how they stream tokens and shape error responses.
The gateway should normalize both so application code handles one
consistent contract:

```python
class NormalizedError(Exception):
    def __init__(self, kind: str, retryable: bool, message: str):
        self.kind = kind            # "rate_limit" | "invalid_request" | "server_error" | "timeout"
        self.retryable = retryable
        super().__init__(message)

def normalize_provider_error(provider: Provider, raw_exc: Exception) -> NormalizedError:
    # each adapter maps its provider's actual exception types/status codes here
    if "429" in str(raw_exc):
        return NormalizedError("rate_limit", retryable=True, message=str(raw_exc))
    if "timeout" in str(raw_exc).lower():
        return NormalizedError("timeout", retryable=True, message=str(raw_exc))
    return NormalizedError("server_error", retryable=False, message=str(raw_exc))
```

`retryable` is the field that matters most operationally — it's what lets
generic retry middleware decide whether to retry, fall back, or fail fast
without needing to understand every provider's specific error taxonomy.

## How It Actually Works

The gateway pattern works because it inserts a stable interface at the
one point in the system where instability is guaranteed: the boundary to
external, independently-operated services whose latency, availability,
and exact API shape change on a schedule you don't control. Every
capability above — routing, fallback, quotas, error normalization — is a
variation on the same idea: push provider-specific knowledge to the edge
(the adapter), and keep everything behind that edge working against one
internal contract that changes far less often than any single provider's
API does.

The fallback chain's reliability comes specifically from failing fast and
trying the next option rather than retrying the same failing provider
repeatedly — a provider returning `429` or timing out is, by definition,
not currently able to serve the request, so a fast fallback to a
different provider (or a smaller/cheaper model that's still up) produces a
successful response in the time a naive retry-with-backoff against the
same failing endpoint would still be waiting. The token-bucket limiter's
mechanism — refilling continuously rather than resetting at fixed
intervals — avoids the thundering-herd problem of fixed-window rate
limits, where all capacity resets at once and a burst of queued requests
floods in simultaneously; continuous refill smooths admitted traffic over
time instead.

## Cheat sheet

| Component | Job |
|---|---|
| Provider adapter | Translates internal request/response shape to one provider's API |
| Routing table | Maps task type to an ordered list of providers/models to try |
| Fallback chain | Tries the next provider on failure instead of failing the whole request |
| Token bucket | Per-team/key rate limiting with smooth, continuous refill |
| Normalized errors | One retryable/non-retryable contract across all providers |
| Centralized logging | One place to see cross-team cost, latency, and error rate |

## Exercise

Build the `Gateway` class above with two adapters (a real provider and a
local mock that fails 30% of the time to simulate outages) and a
2-provider fallback chain. Run 100 requests through it, log which
provider actually served each one, and verify the failure rate at the
gateway boundary drops close to zero even though the primary adapter
fails 30% of individual calls. Add a `TokenBucket` per team and confirm a
team exceeding its quota gets rejected before ever reaching an adapter.
