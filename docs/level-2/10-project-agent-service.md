# 10 · Project — Production-Ready Agent Service

This capstone combines everything in Level 2 into one deployable service: a
FastAPI agent backend with prompt caching, an eval suite gating changes,
tool use, and streaming — the shape of a real production agent, not a
demo script.

## Architecture

```
Browser ── SSE ──► FastAPI app ──► Agent loop (tools + planner) ──► Claude API
                        │                    │
                        ▼                    ▼
                  Redis (sessions)     Cached system prompt + docs
                        │
                        ▼
                 evals/ (CI-gated regression suite)
```

## Project layout

```
agent_service/
├── app.py                  # FastAPI routes
├── agent.py                # planner/executor loop + tools
├── sessions.py             # Redis-backed history
├── prompts/
│   └── agent_system/v1.txt
├── evals/
│   ├── golden.jsonl
│   └── run_eval.py
└── .github/workflows/eval.yml
```

## The agent core

Combines module 4's strict tool schemas, module 7's planner/executor split,
and module 2's caching on the (large, stable) system prompt:

```python
# agent.py
from dotenv import load_dotenv
import anthropic, json
from pathlib import Path

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"
SYSTEM_PROMPT = Path("prompts/agent_system/v1.txt").read_text()

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for relevant articles.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 3}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_ticket",
        "description": "File a support ticket when the user's issue needs human follow-up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 10},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["summary", "priority"],
            "additionalProperties": False,
        },
    },
]

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "search_knowledge_base":
            return json.dumps(search_kb(args["query"]))       # your KB search impl
        if name == "create_ticket":
            return json.dumps(file_ticket(**args))             # your ticketing impl
        return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Error: {e}"

def build_system() -> list[dict]:
    return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

def run_agent_turn(history: list[dict], max_iterations: int = 6):
    for _ in range(max_iterations):
        resp = client.messages.create(
            model=MODEL, max_tokens=1000,
            system=build_system(), tools=TOOLS, messages=history,
        )
        if resp.stop_reason != "tool_use":
            return resp.content[0].text, history

        history = history + [{"role": "assistant", "content": resp.content}]
        results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": execute_tool(b.name, b.input)}
            for b in resp.content if b.type == "tool_use"
        ]
        history = history + [{"role": "user", "content": results}]

    return "I wasn't able to finish within the step budget — let me file a ticket for a human to follow up.", history
```

## The service layer

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sessions import load_history, save_history
from agent import run_agent_turn

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    history = load_history(req.session_id)
    history.append({"role": "user", "content": req.message})
    reply, updated_history = run_agent_turn(history)
    save_history(req.session_id, updated_history)
    return {"reply": reply}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

## The gating eval

```python
# evals/run_eval.py
import json, sys
from agent import run_agent_turn

THRESHOLD = 0.85

def load_golden(path="evals/golden.jsonl"):
    return [json.loads(l) for l in open(path)]

def main():
    golden = load_golden()
    passed = 0
    for case in golden:
        reply, _ = run_agent_turn([{"role": "user", "content": case["input"]}])
        if case["expected_tool"] is None or case["expected_tool"] in reply.lower():
            passed += 1
    rate = passed / len(golden)
    print(f"pass rate: {rate:.1%} (threshold {THRESHOLD:.0%})")
    sys.exit(0 if rate >= THRESHOLD else 1)

if __name__ == "__main__":
    main()
```

```yaml
# .github/workflows/eval.yml
name: agent-eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python evals/run_eval.py
```

Any PR touching `prompts/`, `agent.py`, or the tool schemas has to clear
this gate before merge — the same protection a code test suite gives you,
applied to prompt and agent-behavior changes.

## Deploying

Containerize the service so it runs identically in CI, staging, and
production:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run it behind a process manager or orchestrator with `ANTHROPIC_API_KEY`
injected as a secret (never baked into the image), a Redis instance
reachable from the container, and the `/healthz` route wired into your
platform's liveness check.

## How It Actually Works

This project doesn't introduce anything new — it's every mechanism from
modules 1-9 composed into one running system, each piece doing exactly
what its own module explained: the system prompt is cached because its
token sequence is identical on every request (module 2); tool arguments
are constrained by strict JSON Schema (module 4); the agent loop terminates
either on a non-`tool_use` stop reason or a hard iteration cap (Level 1,
module 9); and the eval script is a plain deterministic threshold check
run in CI, gating merges the same way a unit-test suite would.

The one property worth naming explicitly: nothing in this service is
"smart" about deployment or scaling — the model call is still one
stateless request per turn, and everything that makes the *service*
production-grade (session durability, concurrency limits, CI gating,
container reproducibility) is ordinary backend engineering wrapped around
that one call, which is exactly why Level 2's modules could be taught
independently and then assembled here without any new mechanism to learn.

## Cheat sheet

| Layer | Module it came from |
|---|---|
| Cached system prompt | Module 2 |
| Strict tool schemas | Module 4 |
| Agent loop + iteration cap | Level 1, module 9 |
| SSE streaming | Module 9 |
| Redis session persistence | Module 9 |
| CI-gated eval suite | Module 6 |

## Exercise

Stand up the full service locally (FastAPI + Redis via `docker compose`),
add the `/chat/stream` SSE route from module 9 on top of `run_agent_turn`,
and write a 10-case golden set covering both tool paths
(`search_knowledge_base`, `create_ticket`) plus 2 cases where neither tool
should fire. Wire `evals/run_eval.py` into a local pre-commit hook so a
prompt regression is caught before it's even pushed, not just in CI.
