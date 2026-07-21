# Level 1 · Entry <span class="level-badge">Foundations</span>

Goal: get comfortable calling an LLM from Python, engineering prompts that
work reliably, wiring in tools, and shipping a small working AI assistant.

This level is about **building with** LLMs — you'll treat the model as a
powerful API and learn the engineering patterns around it. All examples use
the Anthropic API (Claude) with Python; the concepts map 1:1 to OpenAI,
Google, and other providers.

## Modules

1. [The LLM Landscape & Setup](01-llm-landscape-setup.md)
2. [API Fundamentals](02-api-fundamentals.md)
3. [Prompt Engineering](03-prompt-engineering.md)
4. [Structured Output](04-structured-output.md)
5. [Tool / Function Calling](05-tool-calling.md)
6. [Conversation State & Memory](06-conversation-state-memory.md)
7. [Streaming](07-streaming.md)
8. [Errors, Rate Limits & Cost](08-errors-rate-limits-cost.md)
9. [A Simple Agent Loop](09-simple-agent-loop.md)
10. [Capstone — CLI Personal Assistant](10-capstone-project.md)

## What you'll need

- **Python 3.10+** and a code editor.
- **An Anthropic API key** — created in [module 1](01-llm-landscape-setup.md).
  Every code snippet that talks to the API needs it; snippets that are pure
  Python (parsing, loops, cost math) run without one.
- A few dollars of API credit. The whole level can be completed for well
  under $5 using the models suggested in the lessons.

By the end of this level you'll have built a CLI personal assistant with
tools, streaming responses, conversation memory, cost tracking, and a small
evaluation suite — the core skill set behind almost every LLM product.
