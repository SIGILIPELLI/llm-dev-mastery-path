# 09 · A Simple Agent Loop

An "agent" sounds exotic, but you've already built every ingredient: tools
(module 5), conversation state (module 6), and error handling (module 8). An
agent is just those pieces in a loop — the model **plans** what to do,
**acts** by calling tools, **observes** the results, and repeats until the
task is done. This module assembles a minimal but real agent, adds the
guardrails that keep loops from running away, and — importantly — discusses
when an agent is the wrong tool.

## Plan → Act → Observe

```
        ┌────────────────────────────┐
        │  model plans next step     │◄──────────┐
        └─────────────┬──────────────┘           │
                      │ tool_use?                │
              yes     ▼        no → final answer │
        ┌────────────────────────────┐           │
        │  your code executes tools  │           │
        └─────────────┬──────────────┘           │
                      │ tool results             │
                      └──────────── observe ─────┘
```

The model drives the control flow: *it* decides which tool, in what order,
and when to stop. Your code supplies the tools, enforces the limits, and
keeps the transcript.

## The tools

Three small tools give the agent real abilities — listing files, reading
them, and doing math:

```python
from dotenv import load_dotenv
from pathlib import Path
import anthropic, json

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"
WORKSPACE = Path("./workspace").resolve()   # the agent may only touch this dir

TOOLS = [
    {
        "name": "list_files",
        "description": "List files in the workspace directory.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a text file from the workspace. Returns its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

def safe_path(filename: str) -> Path:
    """Confine file access to the workspace — never trust model-supplied paths."""
    p = (WORKSPACE / filename).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ValueError("path escapes workspace")
    return p

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "list_files":
            return json.dumps([f.name for f in WORKSPACE.iterdir() if f.is_file()])
        if name == "read_file":
            return safe_path(args["filename"]).read_text()[:8000]
        if name == "calculator":
            from lesson05 import safe_eval          # module 5's safe evaluator
            return str(safe_eval(args["expression"]))
        return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Error: {e}"
```

## The agent loop with guardrails

Two guardrails are non-negotiable: a **maximum iteration count** (so a
confused model can't loop forever on your credit card) and
**confirm-before-dangerous-actions** (any tool that writes, deletes, sends,
or spends should require a human yes).

```python
SYSTEM = """You are a careful analyst agent working in a file workspace.
Use the tools to gather facts before answering; use the calculator for ALL
arithmetic. When you have enough information, give your final answer as text.
If the task is impossible with the available tools, say so plainly."""

MAX_ITERATIONS = 10
DANGEROUS_TOOLS = set()          # e.g. {"write_file", "send_email"} when you add them

def confirm(name: str, args: dict) -> bool:
    reply = input(f"  agent wants to run {name}({args}) — allow? [y/N] ")
    return reply.strip().lower() == "y"

def run_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=2000,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )

        if response.stop_reason != "tool_use":
            return next(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[{iteration}] {block.name}({block.input})")
            if block.name in DANGEROUS_TOOLS and not confirm(block.name, block.input):
                output = "Error: user denied permission for this action."
            else:
                output = execute_tool(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the maximum number of steps without finishing."
```

Try it on a task that genuinely requires multi-step planning:

```python
# Put a few .txt invoices in ./workspace first, e.g. each containing "total: 123.45"
print(run_agent(
    "Read every invoice file in the workspace and tell me the combined total, "
    "plus which single invoice was largest."
))
# [1] list_files({})
# [2] read_file({'filename': 'invoice_a.txt'})
# [2] read_file({'filename': 'invoice_b.txt'})      ← parallel tool calls
# [3] calculator({'expression': '123.45 + 87.10'})
# "The combined total is 210.55; invoice_a.txt was the largest at 123.45."
```

Notice you never told it *how* — list-then-read-then-add was the model's own
plan. That's the agent difference.

## More guardrails worth adding

- **Budget cap** — accumulate `response.usage` with module 8's `CostTracker`
  and abort past a dollar threshold.
- **Wall-clock timeout** — `time.monotonic()` check per iteration.
- **Tool allow-lists** — expose only what the task needs; an agent can't
  misuse a tool it doesn't have.
- **Read-only by default** — anything mutating goes in `DANGEROUS_TOOLS`.
- **Transcript logging** — persist `messages` to a file; you'll need it the
  first time the agent does something weird.

## When agents are overkill

An agent loop adds latency (many round trips), cost (the transcript grows
every step), and nondeterminism (different runs take different paths). Reach
for the simplest structure that solves the problem:

| Situation | Use instead |
|-----------|-------------|
| Fixed steps known in advance ("summarize, then translate") | A plain pipeline of 2 API calls |
| One capability needed ("classify this ticket") | A single call, maybe with one forced tool |
| Deterministic transformation ("extract these fields") | Structured output (module 4) |
| Genuinely open-ended, multi-step, tool-dependent task | An agent |

Rule of thumb: **if you can write the steps as code, write code and call the
model per step.** Use an agent only when the model must decide the steps.

## Cheat sheet

| Concept | Key fact |
|---------|----------|
| Agent | Model plans → your code executes tools → model observes → repeat |
| Loop exit | `stop_reason != "tool_use"` → final answer |
| Iteration cap | Always bound the loop (`MAX_ITERATIONS`) — runaway loops burn money |
| Dangerous actions | Human confirmation before anything that writes/sends/spends |
| Path safety | Resolve + containment-check every model-supplied filename |
| Errors to the model | Return tool errors as results; the agent adapts |
| Cost guardrail | Track usage per iteration; abort past a budget |
| Overkill test | If steps are known in advance, use a pipeline, not an agent |

## Exercise

Add a `write_file(filename, content)` tool, register it in
`DANGEROUS_TOOLS`, and give the agent this task: "Read all invoices, then
write a file `report.txt` summarizing each invoice and the grand total."
Confirm you get prompted before the write happens — approve it once and
verify `report.txt`; run it again and deny, and check the agent explains it
couldn't complete the write. Finally, set `MAX_ITERATIONS = 2` and watch the
guardrail fire on the same task.
