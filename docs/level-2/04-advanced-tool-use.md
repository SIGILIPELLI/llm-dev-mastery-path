---
description: "Advanced Tool Use Patterns — Level 1's tool calling covered the request→execute→respond loop for a handful of simple tools. Production tool use adds…"
---

# 04 · Advanced Tool Use Patterns

Level 1's tool calling covered the request→execute→respond loop for a
handful of simple tools. Production tool use adds parallelism, strict
schemas that eliminate malformed calls, finer control over when tools fire,
and server-side tools the model can invoke without a round trip through
your code at all.

## Parallel tool calls, done right

The model can request several tools in one turn. Level 1 already handled
this by returning all results in one message — the advanced part is
executing them **concurrently** when they're independent, instead of
serially:

```python
from dotenv import load_dotenv
import anthropic, concurrent.futures, json

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

def execute_tool(name: str, args: dict) -> str:
    ...  # dispatch as in Level 1

def run_tools_parallel(tool_use_blocks: list) -> list[dict]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(execute_tool, b.name, b.input): b
            for b in tool_use_blocks
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            block = futures[future]
            try:
                output = future.result(timeout=15)
            except Exception as e:
                output = f"Error: {e}"
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
    return results
```

Only parallelize tools that are actually safe to run out-of-order and
concurrently (read-only lookups, independent API calls). Anything with
side effects that could conflict (two writes to the same record) should
stay serial even if the model requested them together.

## Strict schemas

A loose JSON Schema lets the model emit technically-valid-but-useless
arguments (an empty string where you needed an enum value, a number where
you needed an ISO date). Tighten the schema itself rather than relying on
the model to infer the constraint from a description:

```python
TOOLS = [{
    "name": "create_ticket",
    "description": "File a support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "category": {"type": "string", "enum": ["billing", "technical", "account"]},
            "due_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "summary": {"type": "string", "minLength": 10, "maxLength": 200},
        },
        "required": ["priority", "category", "summary"],
        "additionalProperties": False,
    },
}]
```

`enum`, `pattern`, `minLength`/`maxLength`, and `additionalProperties: False`
all narrow the space of arguments the model can produce — the schema itself
is the validation, not a comment telling the model to behave. Still
validate on receipt (a schema constrains generation strongly but you should
never treat any external input, model-generated or not, as pre-validated
before it touches your business logic).

## Fine-grained `tool_choice`

Beyond Level 1's `auto`/`any`/`tool`/`none`, use `tool_choice` deliberately
to shape behavior per call rather than leaving every call fully open:

```python
# Force structured extraction — the "tool" is really just an output schema
resp = client.messages.create(
    model=MODEL, max_tokens=300,
    tools=[EXTRACT_INVOICE_FIELDS_TOOL],
    tool_choice={"type": "tool", "name": "extract_invoice_fields"},
    messages=[{"role": "user", "content": invoice_text}],
)

# Disable parallel calls when a workflow genuinely requires strict ordering
resp = client.messages.create(
    model=MODEL, max_tokens=500,
    tools=TOOLS,
    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    messages=messages,
)
```

Forcing a specific tool for extraction-shaped tasks is often more reliable
than module 4's plain structured-output prompting, because the schema is
enforced by the tool-calling mechanism itself rather than by instructions
in the prompt.

## Server-side tools

Some tools run on the provider's infrastructure instead of yours — web
search and code execution being the common examples — so you never
implement `execute_tool` for them; you only enable them and read the
result blocks:

```python
resp = client.messages.create(
    model=MODEL, max_tokens=800,
    tools=[{"type": "web_search_20250305", "name": "web_search"}],
    messages=[{"role": "user", "content": "What's the latest stable version of PostgreSQL, and when was it released?"}],
)
for block in resp.content:
    if block.type == "text":
        print(block.text)
    elif block.type == "server_tool_use":
        print(f"[server ran] {block.name}({block.input})")
```

Server-side tools remove your infra burden (no search API key, no sandboxed
execution environment to maintain) but also remove your ability to
intercept or modify the call before it runs — treat them as trusted
extensions of the model's own capability, not as tools you control the
same way as your own `execute_tool` dispatcher.

## How It Actually Works

Parallel tool *requests* are a property of generation, not execution: the
model, in a single forward pass over the growing output, can emit multiple
separate `tool_use` blocks before it stops — nothing about the model
itself runs concurrently. Whether those independent requests are then
executed one-at-a-time or concurrently is entirely a property of your
client code (the thread pool above), because from the API's perspective
all it requires is that every `tool_use_id` gets exactly one matching
`tool_result` in the next message, in any order.

Schema strictness works upstream of generation: providing `enum`,
`pattern`, and type constraints changes what the model is being asked to
produce, and generation of structured tool arguments uses the same
constrained decoding mechanism referenced in Level 1's structured-output
module — the sampler is restricted to tokens that keep the output a valid
instance of the schema at every step, so an `enum` genuinely makes it
impossible for the model to emit a value outside the list, unlike a
sentence in the description merely making it less likely.

Server-side tools work by the provider running an inference loop
functionally identical to your own client-side agent loop (Level 1, module
9), just on their infrastructure between forming the request and returning
the final response to you — the model still only ever *requests* an action
in `server_tool_use` form; it is the provider's orchestration code, not the
model, that executes the search or code and feeds the result back in for a
continued turn, exactly the same request→execute→respond mechanism, just
with the execution step relocated off your machine.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Parallel execution | Model batches requests; your code decides serial vs. concurrent |
| Strict schema | `enum`/`pattern`/`additionalProperties: False` constrain generation directly |
| Forced tool | `tool_choice: {"type": "tool", "name": ...}` — reliable structured extraction |
| Disable parallel | `disable_parallel_tool_use: True` when ordering matters |
| Server-side tools | Provider executes; you only enable and read results |
| Trust boundary | Validate tool arguments on receipt even with a strict schema |

## Exercise

Take three independent read-only tools (e.g. `get_weather`, `get_stock_price`,
`get_time_in_timezone`) and give the model a prompt that requires calling
all three. Measure wall-clock time executing them serially vs. with the
`ThreadPoolExecutor` pattern above, and print the speedup. Then add a strict
schema to one tool (an `enum` on a units field) and show, with a
deliberately ambiguous prompt, that the model can no longer produce an
invalid unit value.
