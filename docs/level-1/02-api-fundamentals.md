# 02 · API Fundamentals

Every request you send to an LLM API has the same anatomy: a model, a token
limit, an optional system prompt, and a list of messages. Mastering this
shape — and what each knob does — is the foundation for everything else in
this course. All snippets below need `ANTHROPIC_API_KEY` set up as in
[module 1](01-llm-landscape-setup.md).

## The messages format: roles

A conversation is a list of messages, each with a `role` and `content`:

- **`user`** — what the human (or your application) says.
- **`assistant`** — what the model said in earlier turns. You send these
  back so the model remembers the conversation (the API is stateless!).
- **system prompt** — instructions from *you, the developer*, that frame the
  whole conversation. In the Anthropic API it's a top-level `system`
  parameter rather than a message in the list; in OpenAI's API it's a
  message with `role: "system"`. Same concept.

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=300,
    system="You are a terse assistant for busy engineers. Answer in at most two sentences.",
    messages=[
        {"role": "user", "content": "What is a context window?"}
    ],
)
print(response.content[0].text)
```

The system prompt is the highest-leverage line of code in most LLM apps: it
sets persona, rules, format, and boundaries, and the model treats it as more
authoritative than user text.

## Multi-turn message lists

The API has no memory between calls. To continue a conversation, you resend
the whole history, alternating `user` / `assistant`, always ending on a
`user` turn:

```python
messages = [
    {"role": "user", "content": "My name is Priya and I write Go."},
    {"role": "assistant", "content": "Nice to meet you, Priya! Go is a great language."},
    {"role": "user", "content": "What's my name and what do I write?"},
]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=100,
    messages=messages,
)
print(response.content[0].text)
# "Your name is Priya and you write Go."
```

The model "remembers" only because turn 1 and 2 are literally in the request.
Module 6 builds a proper conversation manager around this.

## max_tokens: the output cap

`max_tokens` is a **hard limit on output**, not a request for a length. If
the model hits it mid-sentence, generation simply stops and the response's
`stop_reason` is `"max_tokens"`:

```python
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=10,  # absurdly small, to demonstrate
    messages=[{"role": "user", "content": "Explain HTTP in detail."}],
)
print(response.stop_reason)        # "max_tokens" — output was truncated
print(response.content[0].text)    # cut off mid-thought
```

Always check `stop_reason`. `"end_turn"` means the model finished naturally;
`"max_tokens"` means you truncated it — raise the limit or stream (module 7).
A sensible default while learning is 1000–4000.

## Temperature and sampling — the concept

LLMs generate text by sampling from a probability distribution over the next
token. **Temperature** classically controls how adventurous that sampling is:
low values make output focused and repeatable-ish, high values make it more
varied and creative. Related knobs `top_p` and `top_k` restrict sampling to
the most likely tokens.

Two practical notes for modern APIs:

- **The newest Claude models (Sonnet 5, Opus 4.8) no longer accept sampling
  parameters** — they manage sampling internally, and you steer style through
  prompting ("give a creative, unexpected answer" / "answer precisely").
  Sending a non-default `temperature` to these models returns a 400 error.
- Older models and most other providers still expose `temperature` (typically
  0.0–1.0). Even `temperature=0` never guaranteed identical outputs.

```python
# Temperature still works on e.g. claude-haiku-4-5:
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    temperature=1.0,  # more varied output
    messages=[{"role": "user", "content": "Invent a name for a coffee-loving robot."}],
)
print(response.content[0].text)
```

The portable lesson: know what temperature *means*, and prefer prompt-based
steering — it works on every model.

## Tokens, counting, and context windows

Everything is measured in tokens. Two numbers matter:

- **Context window** — the maximum tokens of *input the model can see at
  once* (Claude Sonnet 5: 1,000,000 tokens — several novels' worth).
- **Output cap** — the maximum tokens a single response can produce
  (up to 128K for Sonnet 5, bounded by your `max_tokens`).

Count tokens *before* sending with the free `count_tokens` endpoint:

```python
count = client.messages.count_tokens(
    model="claude-sonnet-5",
    system="You are a terse assistant.",
    messages=[{"role": "user", "content": "What is a context window?"}],
)
print(count.input_tokens)  # e.g. 25
```

And *after* the call, the response reports exactly what you used and will be
billed for:

```python
print(response.usage.input_tokens)   # tokens you sent
print(response.usage.output_tokens)  # tokens the model generated
```

!!! tip "Don't use tiktoken for Claude"
    `tiktoken` is OpenAI's tokenizer and miscounts Claude tokens badly. Each
    provider tokenizes differently — always use the provider's own counter.

## The response object

A response carries more than text:

```python
response.id            # unique request id — log it for debugging
response.model         # which model actually served the request
response.stop_reason   # "end_turn" | "max_tokens" | "tool_use" | ...
response.content       # list of content blocks
response.usage         # input/output token counts
```

## Cheat sheet

| Concept | Key fact |
|---------|----------|
| Roles | `user` (human), `assistant` (model's prior turns), system prompt (developer rules) |
| System prompt | Top-level `system=` parameter; sets persona, rules, format |
| Statelessness | The API remembers nothing — resend full history every call |
| `max_tokens` | Hard output cap; check `stop_reason == "max_tokens"` for truncation |
| `stop_reason` | `end_turn` = finished; `max_tokens` = truncated; `tool_use` = wants a tool |
| Temperature | Sampling randomness; removed on newest Claude models — steer via prompts |
| Context window | Max input size; 1M tokens for Claude Sonnet 5 |
| Count before | `client.messages.count_tokens(...)` — free |
| Count after | `response.usage.input_tokens` / `.output_tokens` |

## Exercise

Write a script `interview.py` that holds a fixed three-turn conversation with
the model about a topic of your choice by building the `messages` list
manually (turn 1 user, turn 2 the model's real reply that you append, turn 3
a follow-up question that only makes sense if the model remembers turn 1).
Print the `usage` numbers for each call and observe how input tokens grow as
history accumulates. Then use `count_tokens` to predict the third call's
input tokens before making it — how close was your prediction?
