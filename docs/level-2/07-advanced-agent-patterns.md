---
description: "Advanced Agent Patterns — Level 1's agent loop (module 9) works well for short, tool-bounded tasks. Once tasks span many steps — a research report, a…"
---

# 07 · Advanced Agent Patterns

Level 1's agent loop (module 9) works well for short, tool-bounded tasks.
Once tasks span many steps — a research report, a multi-file refactor — a
single model juggling planning, execution, and self-checking in one
undifferentiated loop starts to drift, lose track of the original goal, or
blow past its context window. This module covers the patterns that keep
long-horizon agents on track: planner/executor splits, reflection, task
management, and context editing.

## Planner/executor split

Instead of one model doing everything, separate "decide what to do" from
"do it." A planner call produces a structured plan; an executor loop
carries it out step by step, only consulting the planner again if the plan
needs revising:

```python
from dotenv import load_dotenv
import anthropic, json

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

PLAN_TOOL = {
    "name": "set_plan",
    "description": "Record the step-by-step plan to accomplish the task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["steps"],
    },
}

def make_plan(task: str) -> list[str]:
    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        tools=[PLAN_TOOL], tool_choice={"type": "tool", "name": "set_plan"},
        messages=[{"role": "user", "content":
            f"Break this task into 3-7 concrete, ordered steps:\n\n{task}"}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")["steps"]

def execute_step(step: str, context: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=800, tools=TOOLS,
        messages=[{"role": "user", "content":
            f"Context so far:\n{context}\n\nNow do this step: {step}"}],
    )
    # (tool-use loop from Level 1 module 9 goes here)
    return resp.content[0].text

plan = make_plan("Research three vector databases and recommend one for a 50M-vector RAG system.")
context = ""
for step in plan:
    result = execute_step(step, context)
    context += f"\n[{step}] -> {result}"
```

The planner uses a cheap, forced-schema call to produce a plan you can log,
edit, or show to a user for approval *before* any tool executes — turning
an opaque agent into an auditable, checkpointable workflow.

## Reflection

After executing a step (or the whole task), have the agent explicitly
compare its result against the original goal and decide whether to
continue, retry, or revise the plan — a self-critique loop (module 1)
applied to multi-step execution rather than single-shot text:

```python
REFLECT_TOOL = {
    "name": "reflect",
    "description": "Assess progress toward the goal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "goal_met": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "next_action": {"type": "string", "enum": ["continue", "retry_step", "revise_plan", "done"]},
        },
        "required": ["goal_met", "issues", "next_action"],
    },
}

def reflect(goal: str, context: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        tools=[REFLECT_TOOL], tool_choice={"type": "tool", "name": "reflect"},
        messages=[{"role": "user", "content":
            f"Goal: {goal}\n\nWork so far:\n{context}\n\n"
            f"Has the goal been met? What issues remain?"}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")
```

Wiring reflection into the loop after each step (or every few steps) lets
the agent catch its own dead ends — a tool that returned an error it
silently ignored, a step that technically ran but didn't advance the
goal — instead of marching through the rest of a doomed plan.

## Long-horizon task management

For tasks spanning dozens of steps, keep an explicit task list as
structured state outside the model's free-form context, and have the agent
update it rather than relying on the conversation transcript to convey
progress:

```python
class TaskList:
    def __init__(self, steps: list[str]):
        self.tasks = [{"step": s, "status": "pending"} for s in steps]

    def mark(self, index: int, status: str) -> None:
        self.tasks[index]["status"] = status

    def summary(self) -> str:
        return "\n".join(f"[{t['status']}] {t['step']}" for t in self.tasks)

tasks = TaskList(plan)
for i, task in enumerate(tasks.tasks):
    tasks.mark(i, "in_progress")
    execute_step(task["step"], context=tasks.summary())
    tasks.mark(i, "done")
```

Feeding `tasks.summary()` back into each call — instead of the full raw
transcript — gives the model a compact, always-current view of overall
progress, which stays cheap and legible even as the task grows to 50 steps.

## Context editing

As a long-running agent accumulates tool results, the transcript grows
past the point where every detail is still useful — most of the value of
an old tool result is already captured in a later summary. Periodically
compact history instead of letting it grow unbounded:

```python
def compact_history(messages: list[dict], keep_last: int = 4) -> list[dict]:
    if len(messages) <= keep_last:
        return messages
    old, recent = messages[:-keep_last], messages[-keep_last:]
    old_text = "\n".join(str(m["content"])[:500] for m in old)
    summary = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": f"Summarize the key facts and decisions from this agent history:\n\n{old_text}"}],
    ).content[0].text
    return [{"role": "user", "content": f"[Earlier context summary]: {summary}"}] + recent
```

This trades some fidelity for staying within the context window
indefinitely — reserve it for genuinely long-running agents, since
summarization is lossy and can drop a detail a later step needed.

## How It Actually Works

None of these patterns give the model a capability it didn't have — they
are all ways of shaping what goes into its input on each call, using
mechanisms already covered: forced tool schemas (module 4) for the planner
and reflector, and plain prompt engineering for summarization. The
"planner" and "executor" are not different models or different modes; they
are the same underlying model, called twice with different framing and
different `tool_choice` constraints.

Reflection helps for the same reason self-critique does (module 1):
verifying "did this step accomplish X" against an already-completed action
is an easier discrimination task than getting every step right the first
time, so a dedicated reflection call catches drift that would otherwise
only surface as a wrong final answer several steps later.

Context editing matters because of a hard architectural constraint, not a
design choice: the model has no state between calls beyond the token
sequence you send it (Level 1 module 9's "How It Actually Works"), and that
sequence has a fixed maximum length. A task that runs long enough will
eventually generate more tool-result text than fits in the context window
at all, at which point *something* must be dropped or compressed — context
editing just makes that drop deliberate (a summary of old steps) rather
than an abrupt truncation error partway through a run.

## Cheat sheet

| Pattern | Purpose |
|---|---|
| Planner/executor split | Auditable, checkpointable multi-step plans |
| Reflection | Catch drift/dead-ends before they compound |
| Task list as external state | Compact, always-current progress view |
| Context editing | Stay within the context window on long-running agents |

## Exercise

Extend Level 1's invoice agent (module 9) into a planner/executor agent:
the planner produces a numbered plan for "read every invoice, flag any
duplicate, and produce a summary report," the executor runs each step, and
a reflection call runs after the final step to confirm the goal was met.
Log the plan, each step's result, and the final reflection verdict
separately so you can inspect each layer independently.
