# 01 · Advanced Prompting Patterns

Level 1's prompting (module 3) covered single-shot instructions. Real
applications chain prompts together, have the model critique its own work,
and manage prompt *libraries* that grow into the hundreds. This module
covers the patterns that make that manageable: meta-prompting, prompt
chaining, self-critique loops, and versioned prompt libraries.

## Meta-prompting: using the model to write prompts

Instead of hand-crafting a prompt for a narrow task, ask a strong model to
draft one, then refine it against real examples:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

def draft_prompt(task_description: str, example_inputs: list[str]) -> str:
    meta = f"""You write system prompts for other LLM calls. Write a system
prompt for a model that will: {task_description}

Here are example inputs it will see:
{chr(10).join(f"- {e}" for e in example_inputs)}

Output ONLY the system prompt text, no commentary."""
    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content": meta}],
    )
    return resp.content[0].text

system_prompt = draft_prompt(
    "classify a support ticket as billing, technical, or account, and "
    "extract the customer's stated urgency (low/medium/high)",
    ["My card was charged twice", "The app crashes on launch", "I can't log in"],
)
print(system_prompt)
```

Treat the output as a first draft, not gospel — meta-prompting gives you a
reasonable starting point in seconds, but you still validate it against a
labeled eval set (module 6) before shipping it.

## Prompt chaining

Break a complex task into a pipeline of focused calls, each with a narrow
job and clean output, rather than one giant prompt trying to do everything
at once:

```python
def summarize(text: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": f"Summarize in 3 bullet points:\n\n{text}"}],
    )
    return r.content[0].text

def extract_action_items(summary: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content":
            f"From this summary, list concrete action items as a numbered list. "
            f"If there are none, say 'None.'\n\n{summary}"}],
    )
    return r.content[0].text

def draft_reply(action_items: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content":
            f"Write a short, professional email reply that addresses these "
            f"action items:\n\n{action_items}"}],
    )
    return r.content[0].text

meeting_notes = "..."  # long raw transcript
summary = summarize(meeting_notes)
actions = extract_action_items(summary)
reply = draft_reply(actions)
```

Each link is independently testable, independently promptable, and — because
the output of one becomes deterministic-ish text input to the next — you can
insert validation, caching, or a human checkpoint between any two links.
The cost is more round trips and more tokens (each step re-sends its input);
that tradeoff is usually worth it once a single-prompt version starts
producing inconsistent results on some steps but not others.

## Self-critique (generator-critic) loops

Have the model check its own work with a second, differently-instructed
call before you accept the output:

```python
def generate_with_critique(prompt: str, max_rounds: int = 2) -> str:
    draft = client.messages.create(
        model=MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

    for _ in range(max_rounds):
        critique = client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content":
                f"Critique this response for factual errors, missing edge "
                f"cases, and unclear wording. If it's genuinely fine, reply "
                f"with exactly 'OK'.\n\nOriginal task: {prompt}\n\nResponse:\n{draft}"}],
        ).content[0].text

        if critique.strip() == "OK":
            break

        draft = client.messages.create(
            model=MODEL, max_tokens=600,
            messages=[{"role": "user", "content":
                f"Revise this response to address the critique.\n\n"
                f"Response:\n{draft}\n\nCritique:\n{critique}"}],
        ).content[0].text

    return draft

print(generate_with_critique(
    "Write a Python function that merges two sorted lists into one sorted list."
))
```

Self-critique catches a meaningful fraction of errors (missing an edge case,
an off-by-one, an unsupported claim) because criticizing is an easier task
than generating from scratch — the critic only has to *check* a claim
against the text, not invent the right answer. It is not a substitute for
grounding against real sources; a self-critique loop can still confidently
agree with a wrong answer if the error isn't visible from the text alone.

## Managing a prompt library

Once you have a dozen system prompts, keep them as versioned, testable
artifacts instead of inline strings scattered across the codebase:

```python
# prompts/ticket_classifier/v3.txt   ← plain files, one per prompt+version
# prompts/ticket_classifier/meta.json
import json
from pathlib import Path

PROMPT_DIR = Path("prompts")

def load_prompt(name: str, version: str | None = None) -> str:
    meta = json.loads((PROMPT_DIR / name / "meta.json").read_text())
    version = version or meta["current"]        # pin or float to latest
    return (PROMPT_DIR / name / f"{version}.txt").read_text()

system = load_prompt("ticket_classifier")        # uses meta["current"]
system_pinned = load_prompt("ticket_classifier", version="v2")  # A/B or rollback
```

`meta.json` records which version is "current," and each version file is
diffable in git and reviewable in a PR like any other code change — because
a prompt edit is a behavior change, and should go through the same review
and eval-gating (module 6) as a code change.

## How It Actually Works

None of these patterns change what a single call to the model does — each
one is still exactly the stateless "predict the next tokens given this
input" mechanism from Level 1. What changes is the *shape of the input* the
orchestrating code constructs before each call.

Meta-prompting works because "write a good system prompt for task X" is
itself just a prompt-completion task the model was trained on (technical
writing, instruction-following examples), so it produces plausible
instructions the same way it produces plausible prose — it is not reasoning
about your specific downstream failure modes, it is pattern-completing what
a reasonable prompt looks like.

Chaining works because each link's output becomes literal text inside the
next link's input — there is no hidden state carried between calls beyond
whatever text you explicitly pass along. This is also why chains are
debuggable: you can print, log, or replay the exact string at each hop.

Self-critique works because generation and evaluation draw on different
slices of the same underlying next-token distribution: producing a novel,
correct answer requires getting every token right, while judging a given
answer only requires recognizing whether it's consistent with patterns of
correct/incorrect text seen in training — a strictly easier discrimination
task, which is why a second pass over already-generated text catches errors
the first pass didn't.

## Cheat sheet

| Pattern | Use it for |
|---|---|
| Meta-prompting | Fast first draft of a new system prompt |
| Prompt chaining | Complex tasks with clear intermediate outputs |
| Self-critique | Catching errors before returning output to users |
| Prompt library | Versioning, reviewing, and rolling back prompt changes |

## Exercise

Build a 3-step chain that takes a raw product review, (1) extracts
sentiment and specific complaints as JSON (module 4's structured output),
(2) drafts a customer-service reply addressing each complaint, and (3)
critiques the reply for tone before returning it. Store the three prompts
as versioned files under a `prompts/` directory with a `meta.json` per
prompt, and print the intermediate JSON and critique so you can see each
hop.
