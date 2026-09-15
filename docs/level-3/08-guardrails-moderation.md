---
description: "Guardrails & Content Moderation — An agent with tools (Level 1, module 9) and a chatbot facing real users both need defenses beyond 'trust the model's…"
---

# 08 · Guardrails & Content Moderation

An agent with tools (Level 1, module 9) and a chatbot facing real users
both need defenses beyond "trust the model's judgment": input filtering
before a request reaches the model, moderation on what the model
produces, and a policy-enforcement layer that's auditable independent of
either model's behavior.

## Input filtering

Check user input *before* it reaches the model — cheaper (a filter call is
faster/cheaper than a full generation you'd then have to discard) and
prevents obviously-bad input from ever entering context at all:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

MODERATION_TOOL = {
    "name": "classify_input",
    "description": "Classify user input for policy violations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "violates_policy": {"type": "boolean"},
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": ["harassment", "self_harm", "illegal_activity", "pii_request", "none"]},
            },
        },
        "required": ["violates_policy", "categories"],
    },
}

def check_input(user_text: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=200,
        tools=[MODERATION_TOOL], tool_choice={"type": "tool", "name": "classify_input"},
        messages=[{"role": "user", "content": f"Classify this message:\n\n{user_text}"}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")

verdict = check_input("How do I pick a lock to break into my own house?")
if verdict["violates_policy"]:
    print(f"Blocked: {verdict['categories']}")
else:
    # proceed to the main model call
    pass
```

Using a forced-tool-schema classification call (module 4, Level 2) rather
than asking the main conversational model "is this okay?" as free text
keeps the policy decision structured, loggable, and separable from the
conversational response itself.

## Output filtering

Check the model's *own* output before returning it to the user — a model
can still produce a problematic response even from benign input (it might
misunderstand context, or a jailbreak attempt might partially succeed):

```python
def check_output(assistant_text: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=200,
        tools=[MODERATION_TOOL], tool_choice={"type": "tool", "name": "classify_input"},
        messages=[{"role": "user", "content": f"Classify this AI-generated response:\n\n{assistant_text}"}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")

def safe_chat(user_text: str) -> str:
    input_check = check_input(user_text)
    if input_check["violates_policy"]:
        return "I can't help with that request."

    response = client.messages.create(model=MODEL, max_tokens=500, messages=[{"role": "user", "content": user_text}]).content[0].text

    output_check = check_output(response)
    if output_check["violates_policy"]:
        return "I'm not able to provide that response — let me know if I can help differently."

    return response
```

Two moderation calls per turn adds latency and cost — budget for it the
same way you budget for any other production safety check, and consider a
cheaper/faster model for the moderation calls than for the primary
conversational response.

## Dedicated moderation models vs. general LLM classification

A general-purpose model doing classification (as above) works but a
dedicated moderation model or API, purpose-trained and continuously
updated against new abuse patterns, is often more reliable and cheaper for
this narrow task at scale:

```python
import requests

def moderate_with_dedicated_api(text: str) -> dict:
    resp = requests.post(
        "https://api.moderation-provider.example/v1/moderate",
        json={"input": text},
        headers={"Authorization": "Bearer YOUR_KEY"},
    )
    return resp.json()   # {"flagged": bool, "categories": {...}, "category_scores": {...}}
```

Layer both when stakes are high: a fast dedicated moderation pass for
common, well-understood categories, plus your own policy-specific
LLM-based classifier for domain rules a generic moderation API wouldn't
know about (e.g. "never discuss unreleased product features").

## Policy enforcement as a separate layer

Treat policy as data, not scattered `if` statements inline in your agent
code — a table of rules the enforcement layer consults, so policy changes
don't require rewriting application logic:

```python
POLICIES = [
    {"category": "pii_request", "action": "block", "message": "I can't share personal information."},
    {"category": "illegal_activity", "action": "block", "message": "I can't help with that."},
    {"category": "self_harm", "action": "escalate", "message": "If you're in crisis, please reach out to a crisis line."},
]

def enforce(verdict: dict) -> tuple[str, str] | None:
    for category in verdict["categories"]:
        for policy in POLICIES:
            if policy["category"] == category:
                return policy["action"], policy["message"]
    return None
```

Log every enforcement decision (input, verdict, action taken) to an
audit trail separate from the conversational transcript — when a policy
needs revisiting (too strict, too lax, missing a category), you need
records of what was actually blocked and why, not just the eventual user
complaint.

## How It Actually Works

Input and output filtering both work through the same forced-schema
classification mechanism from module 4 (Level 2) and self-critique (module
1, Level 2): a model call whose job is narrowly "does this text match one
of these categories," constrained to a fixed enum output. This is a
fundamentally easier and more reliable task than open-ended generation,
for the same reason judge models are more reliable graders than
generators are error-free writers — classification only requires
recognizing a pattern against explicit categories already present in the
prompt, not producing novel correct content.

The reason a *separate* output check matters, beyond the input check, is
architectural: nothing about the model prevents a benign-looking input
from producing a problematic completion, because the model's response is
still just next-token prediction conditioned on the full conversation —
an input classifier judges the prompt in isolation, before the model has
generated anything, and simply cannot see failure modes that only emerge
in the generated text itself (a subtly manipulative multi-turn setup, a
jailbreak that succeeds gradually across turns). Checking the actual
output closes that gap because it evaluates what was actually produced,
not what was predicted might be produced from the input alone.

Treating policy as external data rather than inline code matters for the
same reason externalizing prompts as versioned files did in Level 2,
module 1: a policy change becomes an auditable, reviewable diff to a rule
table instead of a code change buried in conditional logic, and the same
enforcement function can be tested against a golden set of known
violations exactly like any other eval (Level 2, module 6).

## Cheat sheet

| Layer | Purpose |
|---|---|
| Input filtering | Block or flag before the request reaches the main model |
| Output filtering | Catch problematic completions the input check couldn't predict |
| Dedicated moderation API | Cheaper, purpose-trained for common abuse categories |
| Custom LLM classifier | Domain-specific policy rules a generic API doesn't know |
| Policy table | Rules as data — auditable, versionable, separate from app code |
| Audit logging | Record every enforcement decision, independent of the chat transcript |

## Exercise

Build `safe_chat` with both input and output classification as shown, a
policy table with at least 4 categories and different actions
(block/escalate/allow-with-warning), and an audit log written to a JSONL
file. Test it against 10 benign inputs, 5 inputs that should be blocked
outright, and 2 inputs that would pass the input check but should trigger
the output check (craft a prompt-injection-style input where the danger
only appears in what the model generates). Confirm the audit log captures
enough detail to explain every decision after the fact.
