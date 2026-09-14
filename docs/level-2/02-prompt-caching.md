# 02 · Prompt Caching & Cost Optimization

Every request in Level 1 re-sends the full prompt — system instructions,
few-shot examples, long reference documents — and pays full price for every
token, every time. If the same large prefix repeats across calls (a long
system prompt, a big document you're asking many questions about, a
growing conversation history), **prompt caching** lets the server reuse
the already-processed prefix instead of recomputing it, cutting both cost
and latency dramatically.

## Cache breakpoints

You mark where a cacheable prefix ends with `cache_control`. Everything
before (and including) that block becomes a candidate for caching:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

LONG_REFERENCE_DOC = open("product_manual.txt").read()  # e.g. 40,000 tokens

def ask_about_manual(question: str) -> anthropic.types.Message:
    return client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=[
            {
                "type": "text",
                "text": "You are a support agent. Answer only from the manual below.",
            },
            {
                "type": "text",
                "text": LONG_REFERENCE_DOC,
                "cache_control": {"type": "ephemeral"},  # <-- breakpoint
            },
        ],
        messages=[{"role": "user", "content": question}],
    )

r1 = ask_about_manual("How do I reset the device to factory settings?")
print(r1.usage)   # first call: cache_creation_input_tokens is high, cache_read is 0

r2 = ask_about_manual("What's the warranty period?")
print(r2.usage)   # second call: cache_read_input_tokens is high, input_tokens is tiny
```

The first call *writes* the cache (slightly more expensive than a normal
call); subsequent calls within the cache's lifetime *read* it at a steep
discount — commonly a fraction of the normal input price for the cached
portion. You only pay full price again once the cache expires or the
prefix changes.

## Prefix stability — the rule that matters most

Caching only helps if the tokens **before** the breakpoint are byte-for-byte
identical across calls. Anything that changes per-request must go *after*
the last `cache_control` block:

```python
# GOOD — stable system + docs cached, only the question varies below it
messages=[{"role": "user", "content": question}]

# BAD — putting today's date or a random request id inside the cached
# system block invalidates the cache on every single call
system=[{"type": "text", "text": f"Today is {date.today()}. ...", "cache_control": {...}}]
```

Order matters too: put your longest, most stable content first (system
prompt, tool definitions, reference documents), and your shortest, most
volatile content last (the actual user question). For multi-turn chats,
cache the growing conversation prefix by placing a breakpoint on the last
message of the *previous* turn each time you re-send history — the shared
prefix (everything up to that point) gets served from cache even as new
turns are appended.

```python
def chat_turn(messages: list[dict], new_user_text: str) -> tuple[str, list[dict]]:
    messages = messages + [{"role": "user", "content": new_user_text}]
    # mark the last message so this whole prefix can be reused next turn
    messages[-1] = {
        "role": "user",
        "content": [{"type": "text", "text": new_user_text, "cache_control": {"type": "ephemeral"}}],
    }
    resp = client.messages.create(model=MODEL, max_tokens=500, messages=messages)
    messages.append({"role": "assistant", "content": resp.content})
    return resp.content[0].text, messages
```

## Verifying cache hits

Never assume caching is working — check `usage` on every response. A
regression here (someone adds a timestamp to the cached block, say) is
silent and just shows up as a cost spike weeks later unless you monitor it:

```python
def log_cache_efficiency(resp: anthropic.types.Message, label: str) -> None:
    u = resp.usage
    created = getattr(u, "cache_creation_input_tokens", 0) or 0
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    total_cacheable = created + read
    hit_rate = read / total_cacheable if total_cacheable else 0
    print(f"[{label}] input={u.input_tokens} cache_write={created} "
          f"cache_read={read} hit_rate={hit_rate:.0%}")

log_cache_efficiency(r1, "first call")
log_cache_efficiency(r2, "second call")
```

If `hit_rate` stays near 0% across repeated calls that should share a
prefix, something upstream of the breakpoint is changing — diff the raw
request payloads to find it.

## Cutting input costs at scale

Combine caching with two other levers:

- **Batch what doesn't need to be synchronous.** Non-interactive workloads
  (nightly summarization, bulk classification) can use a batch endpoint at
  a further discount in exchange for completing within a longer window
  instead of immediately.
- **Right-size `max_tokens` and trim context.** Caching reduces the cost of
  the *input* tokens you send; it does nothing for tokens you didn't need
  to send in the first place. Summarize or drop stale conversation turns
  (module 6) before they become part of a cached-but-still-billed prefix.

## How It Actually Works

A transformer forward pass computes, for every token in the input, a set of
intermediate "key" and "value" vectors in every attention layer (the KV
cache — previewed conceptually in Level 3's transformer internals module).
Normally, each new API call recomputes those vectors from scratch for the
*entire* input, token by token, because the server has no memory of your
previous request.

Prompt caching works by having the server persist that intermediate KV
state, keyed by the exact token sequence up to your breakpoint, for a short
time window (commonly a few minutes, refreshed on reuse). When your next
request arrives with an identical prefix, the server skips recomputing keys
and values for those tokens and reuses the stored ones, only running the
full forward computation for the new tokens after the breakpoint. This is
why prefix stability is not a suggestion but a hard requirement: caching is
implemented as an exact-match lookup on the token sequence — even one
different token anywhere before the breakpoint (a timestamp, a reordered
field, different whitespace) produces a different token sequence, which is
a cache miss for the whole prefix, not a partial hit.

It also explains the pricing asymmetry: writing the cache still costs a
full forward pass (hence the write premium), while reading it skips that
computation almost entirely (hence the steep discount) — you are paying
for GPU compute you avoided, not for a fixed "API feature."

## Cheat sheet

| Concept | Key fact |
|---|---|
| `cache_control` | Marks the end of a cacheable prefix segment |
| Prefix rule | Everything before the breakpoint must be byte-identical across calls |
| Ordering | Stable content (system, docs) first; volatile content (question) last |
| First call | Pays a small write premium (`cache_creation_input_tokens`) |
| Later calls | Steep discount on `cache_read_input_tokens` |
| Verification | Always log `usage` — never assume the cache is hitting |
| Multi-turn | Cache the growing history by breakpointing the latest message each turn |

## Exercise

Take module 1's chained pipeline (or any prompt using a long static system
prompt) and add a cache breakpoint after the system instructions. Run the
same request 3 times in a loop, print `usage` each time, and confirm
`cache_read_input_tokens` climbs on calls 2 and 3. Then deliberately
insert `datetime.now()` into the cached block and re-run — verify the hit
rate drops to zero, and explain in a comment why.
