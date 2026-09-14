# 04 · Running Local Models with Ollama

Every previous module called a hosted API. Sometimes you need the model
running on hardware you control — privacy-sensitive data that can't leave
your network, offline development, or just avoiding per-token cost during
heavy experimentation. **Ollama** packages open-weight models for easy
local execution, with an HTTP API shaped closely enough to what you've
already used that switching is mostly a base-URL change.

## Installing and pulling a model

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (downloads weights, quantized by default — see module 7)
ollama pull llama3.1:8b

# Quick sanity check from the CLI
ollama run llama3.1:8b "Explain what a race condition is in one sentence."
```

The tag after the colon (`8b`) picks the parameter count / quantization
variant — larger tags need more RAM and are slower but generally more
capable; `ollama list` shows what you have locally.

## Calling Ollama from Python

Ollama exposes an HTTP API on `localhost:11434` with both its native
format and an OpenAI-compatible endpoint:

```python
import requests

def generate(prompt: str, model: str = "llama3.1:8b") -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["response"]

print(generate("List three uses for a hash map."))
```

For multi-turn conversation, the `/api/chat` endpoint accepts the same
`messages` list shape used throughout this course:

```python
def chat(messages: list[dict], model: str = "llama3.1:8b") -> str:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
    )
    return resp.json()["message"]["content"]

history = [{"role": "user", "content": "What's the time complexity of binary search?"}]
reply = chat(history)
print(reply)
```

## Streaming from Ollama

Setting `"stream": True` switches the response to newline-delimited JSON
chunks, mirroring Level 1's streaming module but over a local HTTP
connection instead of the hosted API:

```python
def chat_stream(messages: list[dict], model: str = "llama3.1:8b"):
    with requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = requests.compat.json.loads(line)
            yield chunk["message"]["content"]
            if chunk.get("done"):
                break

for token in chat_stream(history):
    print(token, end="", flush=True)
```

## Tool calling with local models

Newer Ollama-packaged models support tool calling with a schema shape
compatible with what you built in Level 1 — but be aware reliability
varies far more by model than with frontier hosted models, so validate
tool-use accuracy against your own eval set (Level 2, module 6) before
trusting it in a pipeline:

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={"model": "llama3.1:8b", "messages": history, "tools": TOOLS, "stream": False},
)
message = resp.json()["message"]
if "tool_calls" in message:
    for call in message["tool_calls"]:
        print(call["function"]["name"], call["function"]["arguments"])
```

## Using the OpenAI-compatible endpoint

If you already have code written against an OpenAI-shaped client, Ollama's
compatibility layer lets you point it at localhost with minimal changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is unused, but required by the client

resp = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Summarize the plot of a heist movie in two sentences."}],
)
print(resp.choices[0].message.content)
```

This is often the fastest way to prototype "what would this cost/behave
like locally" without rewriting your application's call sites.

## Hardware sizing

Rough rule of thumb for a quantized model (module 7 covers quantization in
depth): you need roughly the model's parameter count in GB of RAM/VRAM at
4-bit quantization (an 8B model needs ~5-6GB, a 70B model needs ~40GB+),
plus headroom for context and activations. Running on CPU works but is
noticeably slower than GPU/unified-memory (Apple Silicon) execution —
acceptable for development, often too slow for interactive production use
at larger model sizes.

## How It Actually Works

Ollama is not a different kind of model — it's a runtime that loads the
same transformer architecture from module 1 (attention, feed-forward
layers, embeddings) from a weights file into local memory and runs the
forward pass on your own CPU/GPU instead of a remote data center's. The
API shape difference from the hosted Claude API you've used throughout
this course is purely a software convention chosen by each provider; the
underlying computation — tokenize input, run it through the stacked
transformer blocks, sample from the resulting logits — is architecturally
the same process either way.

The practical differences you'll actually notice all trace back to one
fact: you now own the compute. Latency depends on your hardware's raw
throughput rather than a provider's fleet of large accelerators, which is
why local generation is often visibly slower token-by-token than the
hosted API for a comparably-sized model. Tool-calling reliability
differences trace back to training, not the runtime: a model only calls
tools well if it was specifically fine-tuned on tool-use examples in a
format the runtime's chat template reproduces faithfully — Ollama runs
whatever chat template ships with the model file, so a model's tool
reliability locally is a property of that specific open-weight model's
training, not of Ollama itself.

## Cheat sheet

| Concept | Key fact |
|---|---|
| `ollama pull` | Downloads a model's weights, quantized by tag |
| `/api/generate` | Single-prompt completion |
| `/api/chat` | Multi-turn, same `messages` shape as the hosted API |
| Streaming | `"stream": true` → newline-delimited JSON chunks |
| Tool calling | Supported by newer models; validate reliability per model |
| OpenAI-compat endpoint | `base_url="http://localhost:11434/v1"` |
| Hardware rule of thumb | ~parameter count in GB of RAM at 4-bit quantization |

## Exercise

Pull two differently-sized Ollama models (e.g. an 8B and a smaller 3B
variant), and run the same 10-question eval set (reuse Level 2 module 6's
golden-dataset pattern) against both plus the hosted Claude API. Compare
accuracy, latency, and — using `time.monotonic()` around each call — total
wall-clock time for the full set. Write up which model you'd choose for
an offline-only feature versus a latency-sensitive interactive one.
