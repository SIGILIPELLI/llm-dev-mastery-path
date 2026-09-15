---
description: "Multi-Agent Systems — One agent, one context window, one job — that's the limit of module 7's patterns. Some tasks genuinely benefit from splitting work…"
---

# 08 · Multi-Agent Systems

One agent, one context window, one job — that's the limit of module 7's
patterns. Some tasks genuinely benefit from splitting work across several
agents with different roles: an orchestrator that delegates, specialized
subagents that each focus on one kind of work, and a shared workspace or
messages they use to coordinate.

## Orchestrator / subagent architecture

The orchestrator never does the work itself — it decides *which* subagent
should handle a piece of the task, delegates via a tool call, and
assembles the results:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

DELEGATE_TOOLS = [
    {
        "name": "delegate_to_researcher",
        "description": "Send a research question to the researcher subagent, "
                       "which can search and read documents but cannot write code.",
        "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    },
    {
        "name": "delegate_to_coder",
        "description": "Send a coding task to the coder subagent, which can write "
                       "and run code but has no research/browsing ability.",
        "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
    },
]

def researcher_subagent(question: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=600,
        system="You are a research specialist. Answer using only verifiable facts; "
               "say clearly when you are uncertain.",
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text

def coder_subagent(task: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=800,
        system="You are a coding specialist. Write correct, tested code with brief "
               "comments. Do not explain at length — show the code.",
        messages=[{"role": "user", "content": task}],
    )
    return resp.content[0].text

def orchestrate(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=1000, tools=DELEGATE_TOOLS, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return resp.content[0].text

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.name == "delegate_to_researcher":
                output = researcher_subagent(block.input["question"])
            elif block.name == "delegate_to_coder":
                output = coder_subagent(block.input["task"])
            else:
                output = "Error: unknown subagent"
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

print(orchestrate(
    "Find out the time complexity of Python's Timsort, then write a Python "
    "function that documents that complexity in its docstring."
))
```

Each subagent is *just* a `messages.create` call with its own narrow system
prompt — "delegation" is the orchestrator treating another model call as a
tool, structurally identical to Level 1's tool calling, just with the tool
implementation being an LLM instead of a Python function.

## Why specialize subagents at all

A narrow system prompt and a restricted toolset make a subagent more
reliable at its one job than a single generalist prompt trying to do
everything — the same reasoning that motivates prompt chaining in module
1, applied to agents instead of single calls. It also isolates context: the
coder subagent's window fills with code and errors, the researcher's fills
with search results and citations, and neither pollutes the other's
context with irrelevant history — useful once either role's transcript
would otherwise grow large enough to crowd out the orchestrator's own
context.

## Shared workspace

When subagents need to build on each other's output beyond what fits in a
single delegation message, use a shared, external workspace (a directory,
a database, a shared object) rather than routing every intermediate result
back through the orchestrator's context:

```python
import json
from pathlib import Path

WORKSPACE = Path("./agent_workspace")
WORKSPACE.mkdir(exist_ok=True)

def write_artifact(name: str, content: str) -> str:
    (WORKSPACE / name).write_text(content)
    return f"Wrote {name} ({len(content)} chars)"

def read_artifact(name: str) -> str:
    return (WORKSPACE / name).read_text()
```

Give each subagent `write_artifact`/`read_artifact` tools scoped to this
directory; the orchestrator's job becomes coordinating *which* subagent
reads/writes *which* file and in what order, rather than physically
carrying every byte of intermediate output through its own context.

## Inter-agent messaging

For subagents that need to negotiate rather than just execute a single
delegated task (e.g. a "critic" subagent reviewing a "writer" subagent's
draft, back and forth), model the exchange as an explicit message log both
sides read and append to — the same message-list structure used for
conversation state in Level 1, just with two model-driven participants
instead of one model and one human:

```python
def writer_critic_loop(brief: str, max_rounds: int = 3) -> str:
    log = [{"role": "writer", "content": brief}]
    draft = ""
    for _ in range(max_rounds):
        draft = client.messages.create(
            model=MODEL, max_tokens=600,
            messages=[{"role": "user", "content": f"Brief: {brief}\n\nPrior feedback:\n" +
                       "\n".join(f"{m['role']}: {m['content']}" for m in log[1:])}],
        ).content[0].text
        log.append({"role": "writer", "content": draft})

        critique = client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": f"Critique this draft against the brief. "
                       f"Reply 'APPROVED' if it's ready.\n\nBrief: {brief}\n\nDraft:\n{draft}"}],
        ).content[0].text
        log.append({"role": "critic", "content": critique})
        if "APPROVED" in critique:
            break
    return draft
```

This is deliberately just module 1's self-critique loop with the critic
role framed as a separate participant — multi-agent systems are, at bottom,
compositions of the same single-model-call primitives, arranged so each
call sees a narrower, purpose-built context.

## How It Actually Works

There is no mechanism here beyond what earlier modules already
established: every "agent" is a `messages.create` call with its own system
prompt and its own message list, and "multi-agent" simply means your
orchestrating code makes several such calls, in some cases treating one
model's output as another model call's input. No information passes
between agents except through the text (or artifacts) your code explicitly
routes between them — there is no shared memory, shared attention, or
awareness between separate API calls.

This is precisely why context isolation is the real payoff of the
architecture: since each subagent call only ever sees what your code puts
in its `messages` list, giving the coder subagent a clean, code-focused
context (rather than the orchestrator's full delegation history) is not an
optimization detail — it's the entire reason to use a subagent at all
instead of just appending "now act like a coder" to a single, ever-growing
conversation. A single bloated context degrades quality gradually as
irrelevant history crowds out attention on the immediately relevant part
of the input; separate agent calls each get a full, undiluted context
budget for their own narrow task.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Orchestrator | Delegates via tool calls; never does the work itself |
| Subagent | A `messages.create` call with a narrow system prompt, invoked as a tool |
| Specialization payoff | Narrower prompt + isolated context = higher reliability |
| Shared workspace | Files/DB rows for artifacts too large to pass through messages |
| Inter-agent messaging | An explicit message log both sides read/append to |
| No hidden coordination | Agents only share what your code explicitly routes between them |

## Exercise

Build an orchestrator with two subagents: a "planner" (drafts a 3-step
outline for a blog post) and a "writer" (expands one outline step into
2-3 paragraphs). Have the orchestrator delegate the outline first, then
delegate each step to the writer in turn, assembling the final post. Add a
third "editor" subagent that reviews the assembled post and returns
specific line-edit suggestions, and print each subagent's isolated context
window (i.e., exactly what `messages` you sent it) to confirm none of them
saw the others' full transcripts.
