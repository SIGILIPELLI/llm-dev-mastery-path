---
description: "Building LLM Web Apps — Every example so far has been a script printing to a terminal. Shipping an LLM feature to actual users means wrapping it in a web…"
---

# 09 · Building LLM Web Apps

Every example so far has been a script printing to a terminal. Shipping an
LLM feature to actual users means wrapping it in a web backend: an HTTP API
that streams tokens to a browser as they arrive, and session handling so a
user's conversation persists across requests instead of resetting every
time.

## Wrapping the model in FastAPI

A minimal chat endpoint mirrors Level 1's conversation loop, just behind an
HTTP route instead of a `while True` in a terminal:

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()
app = FastAPI()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    history = load_history(req.session_id)          # from module 6's approach, backed by a DB now
    history.append({"role": "user", "content": req.message})

    resp = client.messages.create(model=MODEL, max_tokens=800, messages=history)
    reply = resp.content[0].text

    history.append({"role": "assistant", "content": resp.content})
    save_history(req.session_id, history)
    return {"reply": reply}
```

Run it with `uvicorn app:app --reload` and it's immediately callable from
any HTTP client — the web framework's only job is routing and
(de)serialization; the LLM logic underneath is unchanged from earlier
modules.

## Streaming to the browser with server-sent events

A blocking `/chat` endpoint makes users stare at a spinner for the full
response time. Stream tokens as they're generated using Server-Sent Events
(SSE), the same mechanism Level 1's module 7 used for the terminal, now
piped to an HTTP response:

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history = load_history(req.session_id)
    history.append({"role": "user", "content": req.message})

    def event_generator():
        full_text = ""
        with client.messages.stream(model=MODEL, max_tokens=800, messages=history) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'delta': text})}\n\n"
        history.append({"role": "assistant", "content": full_text})
        save_history(req.session_id, history)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

On the frontend, an `EventSource` (or a `fetch` reading the response body
as a stream) appends each `delta` to the visible message as it arrives:

```javascript
const evtSource = new EventSource(`/chat/stream?...`);
evtSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.done) { evtSource.close(); return; }
  messageDiv.textContent += data.delta;
};
```

This is the single highest-leverage change for perceived responsiveness —
users see the first words within a few hundred milliseconds instead of
waiting for the entire response to finish generating.

## Session management

Sessions need real persistence once you're past a single-process demo —
in-memory dictionaries lose all history on restart and don't work across
multiple server instances behind a load balancer:

```python
import redis, json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
SESSION_TTL_SECONDS = 60 * 60 * 24   # expire idle sessions after a day

def load_history(session_id: str) -> list[dict]:
    raw = r.get(f"session:{session_id}")
    return json.loads(raw) if raw else []

def save_history(session_id: str, history: list[dict]) -> None:
    r.setex(f"session:{session_id}", SESSION_TTL_SECONDS, json.dumps(history))
```

A key-value store (Redis here) is a natural fit: sessions are read/written
as a whole blob keyed by ID, need automatic expiry, and don't require
relational queries — though a real database works too once you need to
query across sessions (analytics, moderation review).

## Handling concurrent requests and backpressure

A web server serves many users at once; without limits, a burst of
requests can exhaust your API rate limit or your server's memory holding
open streaming connections:

```python
import asyncio

SEMAPHORE = asyncio.Semaphore(20)   # cap concurrent in-flight model calls

@app.post("/chat")
async def chat(req: ChatRequest):
    async with SEMAPHORE:
        history = load_history(req.session_id)
        history.append({"role": "user", "content": req.message})
        resp = await client_async.messages.create(model=MODEL, max_tokens=800, messages=history)
        ...
```

Combine this with module 8's error handling (Level 1) at the boundary —
catch rate-limit and overload errors from the provider and translate them
into a clean HTTP 503 with a `Retry-After` header, rather than letting a
raw exception surface to the browser.

## How It Actually Works

None of this changes the model interaction itself — every request still
resolves to the same stateless `messages.create` call from Level 1, with
the identical request→response (or request→stream-of-deltas) cycle. What a
web framework adds is entirely infrastructure around that call: routing an
HTTP request to the right handler, serializing JSON in and out, and
(critically) making the conversation's state — which the API itself never
stores — durable across independent HTTP requests that may hit different
server processes.

Streaming over SSE works by keeping the underlying HTTP response open and
writing chunks to it incrementally instead of buffering the full body
before responding — this maps directly onto the token-by-token generation
mechanism from Level 1's streaming module: the provider is emitting
partial output as it's generated, your server relays each chunk to the
open connection the instant it arrives, and the browser's `EventSource`
API is built to parse exactly this "multiple small messages over one
long-lived connection" pattern.

Session persistence exists purely because the API is stateless per call
(Level 1, module 6): the *only* place a conversation's history lives is
wherever your application chooses to store it between requests. A
key-value store like Redis is not adding capability the model doesn't
have — it's substituting for the fact that the model provider retains
nothing from one API call to the next, so "memory" of a conversation is
entirely your infrastructure's responsibility, reproduced faithfully by
reconstructing and re-sending the full history on every turn.

## Cheat sheet

| Concern | Approach |
|---|---|
| Basic endpoint | FastAPI route wrapping `messages.create` |
| Perceived latency | `StreamingResponse` + SSE, token-by-token |
| Session state | External store (Redis/DB) keyed by session id, with TTL |
| Concurrency limits | `asyncio.Semaphore` capping in-flight model calls |
| Provider errors | Catch and translate to clean HTTP status + retry guidance |

## Exercise

Build a FastAPI app with `/chat/stream` (SSE) backed by Redis-persisted
session history, plus a minimal HTML page using `EventSource` to render
tokens as they stream in. Load-test it with 30 concurrent simulated users
(a simple `asyncio.gather` of requests) with the semaphore set to 5, and
observe requests queueing rather than all firing at once — then remove the
semaphore and note what happens to your API rate-limit error rate.
