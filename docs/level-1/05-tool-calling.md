# 05 · Tool / Function Calling

An LLM by itself can only emit text — it can't check the weather, query your
database, or do reliable arithmetic. **Tool calling** (also called function
calling) fixes that: you describe functions to the model, the model *asks*
to call one with specific arguments, your code executes it and returns the
result, and the model uses that result in its answer. This request→execute→
respond loop is the mechanism behind every AI assistant that "does things,"
and the foundation for agents in module 9.

The crucial mental model: **the model never runs code**. It only produces a
structured request ("call `get_weather` with `{"city": "Paris"}`"). Your
program stays in control of what actually executes.

## Defining tools with JSON Schema

A tool definition has a name, a description (the model reads this to decide
*when* to use it), and a JSON Schema for its parameters:

```python
from dotenv import load_dotenv
import anthropic, json

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. "
                       "Use this for ANY math instead of computing in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression using + - * / ** ( ), e.g. '17.5 * 12'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
]
```

Description quality matters enormously — say **when** to call the tool, not
just what it does. `enum` pins down fixed choices.

## Implementing the tools

Plain Python functions, plus a dispatcher:

```python
import ast, operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def safe_eval(expr: str) -> float:
    """Evaluate arithmetic without eval() — never eval model output!"""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported syntax: {ast.dump(node)}")
    return walk(ast.parse(expr, mode="eval").body)

def get_weather(city: str, unit: str = "celsius") -> dict:
    # Real code would call a weather API; we fake it deterministically.
    fake_temp_c = (sum(map(ord, city.lower())) % 35)
    temp = fake_temp_c if unit == "celsius" else round(fake_temp_c * 9 / 5 + 32)
    return {"city": city, "temp": temp, "unit": unit, "conditions": "partly cloudy"}

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "calculator":
            return str(safe_eval(args["expression"]))
        if name == "get_weather":
            return json.dumps(get_weather(**args))
        return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Error: {e}"      # return errors as text — the model can recover
```

## The tool-use loop

When the model wants a tool, the response has `stop_reason == "tool_use"` and
the content contains `tool_use` blocks. You execute them and send back
`tool_result` blocks (matched by `tool_use_id`) in a **user** message, then
call the API again:

```python
def chat_with_tools(user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1000, tools=TOOLS, messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Model is done — return its final text
            return next(b.text for b in response.content if b.type == "text")

        # 1. Echo the assistant turn (including its tool_use blocks) into history
        messages.append({"role": "assistant", "content": response.content})

        # 2. Execute every requested tool; collect ALL results in ONE user message
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool] {block.name}({block.input})")
                output = execute_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # must match the request
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
        # loop: the model now sees the results and continues


print(chat_with_tools(
    "If it's above 15°C in Madrid, what's 15% tip on a €84.60 dinner there?"
))
# [tool] get_weather({'city': 'Madrid'})
# [tool] calculator({'expression': '84.60 * 0.15'})
# "It's 22°C in Madrid, so... a 15% tip on €84.60 is €12.69."
```

Three rules that trip everyone up:

1. **Append the assistant message first**, tool_use blocks included — the
   API requires every `tool_result` to answer a `tool_use` in the history.
2. **One user message with all results.** The model may request several
   tools in parallel; return every result in a single message, not one
   message each.
3. **Return errors as results**, optionally with `"is_error": True` — the
   model will retry or work around them. Raising an exception kills the
   conversation instead.

## Forcing or forbidding tools

`tool_choice` controls the model's freedom:

```python
tool_choice={"type": "auto"}                      # default — model decides
tool_choice={"type": "any"}                       # must call some tool
tool_choice={"type": "tool", "name": "calculator"} # must call this tool
tool_choice={"type": "none"}                      # text only this turn
```

Forcing a specific tool is handy for extraction pipelines where the "tool"
is really just a schema you want filled in.

## Cheat sheet

| Concept | Key fact |
|---------|----------|
| Tool definition | `{name, description, input_schema}` — schema is JSON Schema |
| Descriptions | Say *when* to use the tool; the model chooses based on them |
| Detection | `response.stop_reason == "tool_use"` |
| Request block | `block.type == "tool_use"` with `.id`, `.name`, `.input` (already parsed) |
| Result message | `{"type": "tool_result", "tool_use_id": ..., "content": ...}` in a **user** turn |
| Parallel calls | Multiple `tool_use` blocks in one response → all results in one message |
| Errors | Return as result text with `is_error: True`; never raise |
| Control | `tool_choice`: `auto` / `any` / `tool` / `none` |
| Safety | The model only *requests*; your code decides what runs. Never `eval()` model output |

## Exercise

Add a third tool `unit_convert(value, from_unit, to_unit)` supporting km/mi,
kg/lb, and °C/°F (use an `enum` in the schema for the units). Then ask a
question that requires **all three tools in one conversation** — e.g. "It's
68°F in Boston; convert that to Celsius, and if it's warmer than 15°C, how
many km is a 26.2-mile marathon and what's 26.2 × 3.1?" Print each tool
invocation as it happens. Finally, break your calculator on purpose (make it
return `"Error: division by zero"` for some input) and watch how the model
handles the error result gracefully.
