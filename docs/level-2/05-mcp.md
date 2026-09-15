---
description: "Model Context Protocol (MCP) — Every tool you've defined so far has been hand-written and wired directly into your own TOOLS list and execute_tool…"
---

# 05 · Model Context Protocol (MCP)

Every tool you've defined so far has been hand-written and wired directly
into your own `TOOLS` list and `execute_tool` dispatcher. That doesn't
scale once you want to reuse the same tool across multiple apps, or plug in
tools someone else maintains (a database connector, a Slack integration).
**MCP (Model Context Protocol)** standardizes how an LLM application
discovers and calls tools exposed by an independent server — write a tool
once as an MCP server, use it from any MCP-compatible client.

## The shape of the protocol

An MCP server exposes three kinds of things over a client/server
connection (typically stdio for local processes, or HTTP for remote ones):

- **Tools** — callable functions with a name, description, and JSON Schema,
  conceptually identical to Level 1's tool definitions.
- **Resources** — readable data (files, DB rows, API responses) addressed
  by URI, for the client to pull into context.
- **Prompts** — reusable, parameterized prompt templates the server
  provides.

The client (your application) connects, asks the server "what tools do you
have," and gets back JSON Schema definitions it can pass straight through
to the model's `tools` parameter — the server is the single source of
truth for what a tool does and how to call it.

## Writing a minimal MCP server

Using the Python SDK, a server exposing one tool looks like this:

```python
# weather_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather for a city."""
    # Real implementation would call a weather API.
    fake_temp_c = sum(map(ord, city.lower())) % 35
    temp = fake_temp_c if unit == "celsius" else round(fake_temp_c * 9 / 5 + 32)
    return {"city": city, "temp": temp, "unit": unit}

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

The decorator inspects the function's type hints and docstring to build the
JSON Schema automatically — you write a normal Python function, and the
protocol-level wiring (schema generation, request parsing, response
serialization) is handled by the SDK.

## Connecting to an MCP server from your app

Your application acts as an MCP **client**, launching the server as a
subprocess and forwarding its tools into a normal `messages.create` call:

```python
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic, asyncio

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

async def run():
    server_params = StdioServerParameters(command="python", args=["weather_mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tools = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            } for t in tools_response.tools]

            messages = [{"role": "user", "content": "What's the weather in Lisbon?"}]
            resp = client.messages.create(model=MODEL, max_tokens=300, tools=tools, messages=messages)

            for block in resp.content:
                if block.type == "tool_use":
                    result = await session.call_tool(block.name, block.input)
                    print(f"{block.name}({block.input}) -> {result.content}")

asyncio.run(run())
```

This is the same request→execute→respond loop from Level 1 — the only
difference is that `execute_tool` is replaced by `session.call_tool`,
which forwards the call over the protocol to the server process instead of
running a local Python function directly.

## Writing your own MCP tools for reuse

The payoff for the extra layer is reuse: a `github_mcp_server.py` you write
once can back a CLI assistant, a Slack bot, and a web app — each app is a
thin MCP client that connects to the same server and never re-implements
the GitHub API calls, auth, or pagination logic. When designing a server
for reuse:

- **Keep tool descriptions self-contained.** The consuming application may
  not know the domain — write descriptions as if the model has never seen
  your system before, because from a fresh client's perspective, it hasn't.
- **Return structured, not prose, results.** Let the calling model decide
  how to phrase things to its own user; a server returning `{"temp": 18,
  "unit": "celsius"}` is more reusable across apps than one returning
  "It's a pleasant 18°C in Lisbon today!"
- **Version your server** the same way you'd version an API — tool schema
  changes are breaking changes for every client depending on them.

## How It Actually Works

MCP does not change how the model calls tools — the model still only ever
emits a `tool_use` block naming a tool and filling arguments against a JSON
Schema, exactly as in Level 1. What MCP standardizes is everything
*around* that: a wire format (JSON-RPC messages over stdio or HTTP) for a
client to ask a server "what tools exist" and "run this one," so that tool
*discovery* and *invocation* are decoupled from any single application's
codebase.

This decoupling is why an MCP tool is reusable in a way a hand-written
`execute_tool` branch isn't: the server owns both the schema definition and
the implementation, and any client that speaks the protocol can fetch the
current schema at connection time via `list_tools()` rather than a
developer hand-copying it into a `TOOLS` list that can drift out of sync
with the actual function signature. The `@mcp.tool()` decorator's job is
purely mechanical — introspecting Python type hints to generate the JSON
Schema your app would otherwise have written by hand, and marshalling
requests and responses across the process boundary.

## Cheat sheet

| Concept | Key fact |
|---|---|
| MCP | Protocol standardizing tool discovery + invocation across apps |
| Tools / Resources / Prompts | The three things an MCP server can expose |
| `list_tools()` | Client fetches current schemas at connection time |
| `call_tool()` | Client-side equivalent of Level 1's `execute_tool` |
| Transport | stdio for local subprocess servers, HTTP for remote ones |
| Reuse payoff | One server backs many client applications |

## Exercise

Turn Level 1's calculator and weather tools into a single MCP server using
`FastMCP`, then write a client script that connects to it, lists its
tools, and runs a conversation requiring both. Confirm the schemas the
client receives from `list_tools()` match what you'd have hand-written in
Level 1 — then add a third tool to the server *without touching the client
code* and confirm the client picks it up automatically on the next
`list_tools()` call.
