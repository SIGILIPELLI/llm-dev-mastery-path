# 01 · The LLM Landscape & Setup

A large language model (LLM) is a neural network trained on enormous amounts
of text to predict the next token in a sequence. That simple objective, at
scale, produces a system that can summarize, translate, write code, answer
questions, and follow instructions. As an application developer you don't
need to know how to train one — you need to know how to **call** one well,
which is what this entire level teaches.

## Hosted APIs vs. local models

There are two ways to get an LLM into your application:

| | Hosted API | Local / open-weight model |
|---|---|---|
| Examples | Anthropic (Claude), OpenAI (GPT), Google (Gemini) | Llama, Mistral, Qwen via Ollama or vLLM |
| Quality | Frontier-level | Good, improving fast, below frontier |
| Setup | An API key and an HTTP call | Download weights, need a capable GPU/CPU |
| Cost model | Pay per token | Pay for hardware/electricity |
| Data location | Provider's servers | Your machine |
| Best for | Most products, fastest path to quality | Privacy-critical or offline use, high fixed volume |

This course uses the **Anthropic API** for examples because the concepts —
messages, system prompts, tools, streaming — are identical across providers.
Level 3 covers running local models with Ollama and vLLM.

## The provider landscape (high level)

- **Anthropic** — the Claude family. Current workhorse model:
  `claude-sonnet-5`; `claude-haiku-4-5` is the fast/cheap tier and
  the Opus family is the high-capability tier.
- **OpenAI** — the GPT family, with a near-identical "chat completions /
  responses" API shape.
- **Google** — Gemini models, integrated with Google Cloud (Vertex AI).
- **Open-weight** — Meta's Llama, Mistral, Qwen, DeepSeek and others you can
  run yourself.

Everything you learn here transfers: every major provider has messages with
roles, a system prompt, `max_tokens`, tool calling, and streaming.

## Get an API key

1. Sign up at [platform.claude.com](https://platform.claude.com/) (the
   Anthropic developer console).
2. Add a small amount of billing credit ($5 is plenty for this level).
3. Create an API key under **API Keys** and copy it — it's shown only once.

The key is a secret. Anyone who has it can spend your credit. That leads to
rule one of LLM development:

## Never hardcode keys — use environment variables

Create a project folder and a virtual environment:

```bash
mkdir llm-course && cd llm-course
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install anthropic python-dotenv
```

Store the key in a `.env` file that **never gets committed**:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

```bash
# .gitignore
.env
.venv/
```

`python-dotenv` loads `.env` into the process environment, and the Anthropic
SDK automatically picks up `ANTHROPIC_API_KEY` from the environment — you
never pass the key in code.

## Your first API call

Create `hello_claude.py`:

```python
# hello_claude.py — requires ANTHROPIC_API_KEY in .env
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads .env into environment variables

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY automatically

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Say hello and tell me one fun fact about language models."}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

Run it:

```bash
python hello_claude.py
# Hello! Here's a fun fact: language models don't see words — they see
# "tokens," small chunks of text ...
```

Three things to notice:

- **`model`** — which model handles the request. Model choice is a
  quality/cost/speed dial you'll tune constantly.
- **`max_tokens`** — a required hard cap on how much the model may generate.
- **`messages`** — a list of conversation turns. Even a single question is a
  one-item conversation. The next module dissects this format.

The response's `content` is a **list of blocks** (text, tool calls, and other
types you'll meet later), which is why we loop instead of assuming a single
string.

## What a "token" is (just enough for now)

Models read and write **tokens** — chunks of roughly 3–4 English characters.
"Hello, world!" is 4 tokens; a page of text is ~500. You pay per token, the
model has a maximum context window measured in tokens, and `max_tokens`
limits output tokens. Module 2 covers counting them precisely.

## Cheat sheet

| Item | Detail |
|------|--------|
| Install | `pip install anthropic python-dotenv` |
| Key storage | `.env` file with `ANTHROPIC_API_KEY=...`, listed in `.gitignore` |
| Load key | `from dotenv import load_dotenv; load_dotenv()` |
| Client | `client = anthropic.Anthropic()` — reads the key from the environment |
| Basic call | `client.messages.create(model=..., max_tokens=..., messages=[...])` |
| Workhorse model | `claude-sonnet-5` |
| Cheap/fast model | `claude-haiku-4-5` |
| Response text | iterate `response.content`, use blocks where `block.type == "text"` |
| Golden rule | Never hardcode or commit an API key |

## Exercise

Set up the project exactly as above, then modify `hello_claude.py` to ask the
model to explain, in exactly three bullet points, the difference between a
hosted LLM API and a local model. Run it twice — once with
`model="claude-sonnet-5"` and once with `model="claude-haiku-4-5"` — and
compare the answers and the response speed. Then check your `.gitignore`
actually works: run `git init && git status` and confirm `.env` is not listed
as an untracked file.
