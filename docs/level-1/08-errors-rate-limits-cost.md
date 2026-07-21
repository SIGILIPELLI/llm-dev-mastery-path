# 08 · Errors, Rate Limits & Cost

LLM APIs fail in all the usual ways HTTP APIs fail — plus a few of their own:
rate limits measured in tokens, overload errors at peak times, and bills that
scale with every character you send. Production LLM code is defined less by
its happy path than by how it retries, times out, and accounts for spend.
This module gives you the full defensive toolkit.

## The error taxonomy

The SDK raises typed exceptions — catch specific ones, never string-match
messages:

| Exception | HTTP | Meaning | Retry? |
|-----------|------|---------|--------|
| `BadRequestError` | 400 | Malformed request (bad model id, missing field) | No — fix the code |
| `AuthenticationError` | 401 | Bad/missing API key | No — fix the key |
| `PermissionDeniedError` | 403 | Key lacks access | No |
| `NotFoundError` | 404 | Wrong model name or endpoint | No |
| `RateLimitError` | 429 | Too many requests/tokens per minute | **Yes, with backoff** |
| `InternalServerError` | ≥500 | Provider-side failure | **Yes, with backoff** |
| `APIStatusError` | any | Base class carrying `.status_code` | Depends |
| `APIConnectionError` | — | Network problem before a response | **Yes** |

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

try:
    response = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": "hello"}],
    )
except anthropic.RateLimitError as e:
    retry_after = e.response.headers.get("retry-after")
    print(f"Rate limited — server says wait {retry_after}s")
except anthropic.APIStatusError as e:
    print(f"API error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    print("Network problem — check connectivity")
```

## Retries with exponential backoff

The rule: wait longer after each failure, add random **jitter** so a fleet of
clients doesn't retry in lockstep, and honor the server's `retry-after`
header when present. Only retry what's retryable — a 400 will fail forever.

```python
import random, time

def call_with_retry(max_retries: int = 5, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == max_retries:
                raise
            # Honor the server's hint if it gives one
            hinted = e.response.headers.get("retry-after")
            delay = float(hinted) if hinted else 2 ** attempt + random.random()
        except anthropic.APIStatusError as e:
            if attempt == max_retries or e.status_code < 500:
                raise                       # 4xx (except 429) is not retryable
            delay = 2 ** attempt + random.random()
        except anthropic.APIConnectionError:
            if attempt == max_retries:
                raise
            delay = 2 ** attempt + random.random()
        print(f"retry {attempt + 1}/{max_retries} in {delay:.1f}s")
        time.sleep(delay)
```

!!! tip "The SDK already retries for you"
    The Anthropic SDK automatically retries 429s, 5xx, and connection errors
    twice with backoff. Configure it instead of reinventing it:
    `anthropic.Anthropic(max_retries=5)`. Write your own loop only when you
    need custom behavior (logging, queueing, fallback models).

## Timeouts

The default request timeout is generous (10 minutes). For interactive apps,
tighten it — and prefer streaming for anything long:

```python
client = anthropic.Anthropic(timeout=30.0, max_retries=3)   # seconds

# Or per-request:
client.with_options(timeout=10.0).messages.create(...)
```

A timeout raises `anthropic.APITimeoutError` (and is retried per
`max_retries`).

## Understanding rate limits

Limits come in three flavors, per model: **RPM** (requests/min), **ITPM**
(input tokens/min), and **OTPM** (output tokens/min). Every response includes
headers showing where you stand:

```python
raw = client.messages.with_raw_response.create(
    model=MODEL, max_tokens=100,
    messages=[{"role": "user", "content": "ping"}],
)
for h in ["x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
          "x-ratelimit-remaining-input-tokens", "x-ratelimit-remaining-output-tokens"]:
    print(h, "=", raw.headers.get(h))
response = raw.parse()   # the normal Message object
```

When you hit 429, the `retry-after` header says how long to wait. If you're
*constantly* rate-limited, backoff isn't the fix — a request queue, a higher
usage tier, or a cheaper/faster model is.

## Estimating and tracking cost

Pricing is per million tokens (MTok), input and output priced separately.
Representative Anthropic prices (check the provider's pricing page for
current numbers):

| Model | Input / MTok | Output / MTok |
|-------|--------------|---------------|
| `claude-sonnet-5` | $3.00 | $15.00 |
| `claude-haiku-4-5` | $1.00 | $5.00 |

Track spend from `response.usage` on every call:

```python
PRICES = {  # $ per million tokens: (input, output)
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

class CostTracker:
    def __init__(self):
        self.input_tokens = self.output_tokens = 0
        self.dollars = 0.0

    def record(self, model: str, usage) -> float:
        in_price, out_price = PRICES[model]
        cost = (usage.input_tokens * in_price + usage.output_tokens * out_price) / 1_000_000
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.dollars += cost
        return cost

tracker = CostTracker()
response = client.messages.create(
    model=MODEL, max_tokens=500,
    messages=[{"role": "user", "content": "Summarize why HTTP/2 multiplexing helps."}],
)
print(f"this call: ${tracker.record(MODEL, response.usage):.6f}")
print(f"session total: ${tracker.dollars:.4f}")
```

Estimate *before* an expensive call with `count_tokens` (free) — multiply by
the input price, and remember that in a conversation the whole history is
re-billed every turn (module 6's summarization is a cost feature as much as a
memory feature).

## Caching strategies

Two distinct kinds of caching cut costs dramatically:

- **Your own response cache** — identical requests shouldn't hit the API
  twice. A dict or Redis keyed on `(model, system, messages)` is often
  enough for FAQ-style workloads:

```python
import hashlib, json

_cache: dict[str, str] = {}

def cached_ask(user_text: str, system: str = "") -> str:
    key = hashlib.sha256(json.dumps(
        [MODEL, system, user_text]).encode()).hexdigest()
    if key not in _cache:
        response = client.messages.create(
            model=MODEL, max_tokens=500, system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        _cache[key] = response.content[0].text
    return _cache[key]
```

- **Provider-side prompt caching** — the API can cache a long, stable prompt
  prefix (big system prompt, tool definitions) so repeat requests pay ~10%
  of the input price for the cached part. You mark a breakpoint with
  `cache_control: {"type": "ephemeral"}` and verify hits via
  `response.usage.cache_read_input_tokens`. Level 2 devotes a full module
  to it — for now, know it exists and that it's the single biggest cost
  lever for chat apps.

## Cheat sheet

| Concern | Practice |
|---------|----------|
| Typed errors | Catch `RateLimitError`, `APIStatusError`, `APIConnectionError` — most specific first |
| Retryable | 429, ≥500, connection errors — with exponential backoff + jitter |
| Not retryable | 400/401/403/404 — fix the request, not the retry count |
| `retry-after` | Honor the header on 429s |
| Built-in retries | `anthropic.Anthropic(max_retries=N)` — SDK backs off automatically |
| Timeouts | `timeout=30.0` on the client; stream long responses |
| Rate limit visibility | `x-ratelimit-*` response headers via `with_raw_response` |
| Cost formula | `(in_tokens × in_price + out_tokens × out_price) / 1e6` |
| Cost data | `response.usage` after; `count_tokens` (free) before |
| Cheap tier | Route easy tasks to `claude-haiku-4-5` |
| Caching | App-level response cache + provider prompt caching |

## Exercise

Build `robust_client.py`: a `RobustLLM` class wrapping the SDK with (a) a
30-second timeout, (b) your own retry loop with backoff + jitter that logs
every retry, (c) a `CostTracker` recording per-call and cumulative spend, and
(d) an in-memory response cache. Prove each feature: trigger a
`NotFoundError` with a fake model name and confirm it does *not* retry; ask
the same question twice and confirm the second call is free and instant; and
print a session cost report showing input tokens, output tokens, and dollars
after five varied calls.
