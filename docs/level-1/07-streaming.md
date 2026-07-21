# 07 · Streaming

A non-streaming API call makes the user stare at nothing until the entire
response is generated — easily 10–30 seconds for a long answer. Streaming
delivers tokens as they're produced, so the first words appear in well under
a second. Every serious chat UI streams; after this module, yours will too.

## Why streaming matters

- **Perceived latency.** Time-to-first-token is typically a few hundred
  milliseconds; time-to-last-token can be 30s+. Streaming turns a dead wait
  into visible progress.
- **Long outputs need it.** Very large `max_tokens` on a non-streaming
  request risks HTTP timeouts — the SDK will actually refuse extreme values
  unless you stream.
- **Responsiveness.** You can render, cancel, or post-process output as it
  arrives.

## The streaming helper

The Python SDK's `client.messages.stream()` context manager is the easy path
— `text_stream` yields plain text chunks:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

with client.messages.stream(
    model=MODEL,
    max_tokens=1000,
    messages=[{"role": "user", "content": "Explain DNS like I'm a curious 12-year-old."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)   # flush=True — or nothing appears until newline
    print()
```

`flush=True` matters: without it Python buffers output and the "stream"
appears in bursts, defeating the purpose.

## Getting the full message afterwards

Streaming doesn't mean giving up the response object. After the stream ends,
`get_final_message()` returns the complete message — usage, stop_reason and
all:

```python
with client.messages.stream(
    model=MODEL, max_tokens=1000,
    messages=[{"role": "user", "content": "Write a haiku about compilers."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()

print(f"\n[{final.usage.output_tokens} output tokens, stop: {final.stop_reason}]")
# Append final.content to your history exactly as in non-streaming code
```

This is the pattern to internalize: **stream for the user, collect the final
message for your program** (history, cost tracking, tool detection).

## Under the hood: the event stream

`text_stream` is a convenience over a typed event sequence. When you need
more than text — thinking indicators, tool-call progress — iterate events:

```python
with client.messages.stream(
    model=MODEL, max_tokens=1000,
    messages=[{"role": "user", "content": "Compare TCP and UDP briefly."}],
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            print(f"\n[block: {event.content_block.type}]")
        elif event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)
```

The event order is always: `message_start` → (`content_block_start` →
`content_block_delta`* → `content_block_stop`)* → `message_delta` (carries
`stop_reason` and usage) → `message_stop`. Under raw HTTP this arrives as
Server-Sent Events (SSE) — the same mechanism you'd proxy to a browser.

## Streaming into a CLI chat

Combining module 6's conversation manager with streaming gives a genuinely
pleasant terminal chatbot:

```python
# chat.py — a streaming CLI chat with memory
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"
SYSTEM = "You are a helpful, concise assistant running in a terminal."

def main() -> None:
    messages: list[dict] = []
    print("Chat started. Ctrl+C or 'quit' to exit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user or user.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": user})
        print("assistant> ", end="", flush=True)
        try:
            with client.messages.stream(
                model=MODEL, max_tokens=1500, system=SYSTEM, messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                final = stream.get_final_message()
            print()
            messages.append({"role": "assistant", "content": final.content})
        except KeyboardInterrupt:
            # User cancelled mid-response: keep history consistent
            messages.pop()
            print("\n[cancelled]")

if __name__ == "__main__":
    main()
```

Note the cancellation handling: if the user Ctrl+C's mid-stream, we remove
the pending user turn so the history stays valid. Handling partial output is
the price of streaming — a truncated stream should never leave your
`messages` list ending with an unanswered turn or half-recorded reply.

## Streaming + tools

Streaming and tool calling compose: stream the turn, then inspect the final
message for `stop_reason == "tool_use"`, run the tools, and stream the next
turn. The agent in module 9 stays non-streaming for clarity; the capstone
wires streaming in.

## Cheat sheet

| Concept | Key fact |
|---------|----------|
| Why stream | First token in ~0.5s instead of full-response latency; long outputs require it |
| Easy path | `with client.messages.stream(...) as stream:` |
| Text chunks | `for text in stream.text_stream: print(text, end="", flush=True)` |
| Full message | `stream.get_final_message()` after the loop — usage, stop_reason, content |
| History | Append `final.content` exactly as in non-streaming code |
| Events | `content_block_delta` carries `text_delta`; `message_delta` carries stop_reason/usage |
| Wire format | Server-Sent Events (SSE) — proxyable to browsers |
| Partial output | On cancel/disconnect, repair your history; never leave a dangling turn |
| `flush=True` | Without it, terminal buffering hides the streaming |

## Exercise

Enhance `chat.py` in three steps: (1) print a live token counter — count
streamed chunks and display `[{n} chunks]` on a status line after each
response, plus the true `output_tokens` from the final message; (2) add a
`/long` command that asks the model for a ~2000-word essay, and measure
time-to-first-token vs. time-to-last-token with `time.monotonic()` — print
both; (3) verify your cancellation handling: Ctrl+C mid-essay, then continue
chatting and confirm the model isn't confused by the aborted turn.
