# 03 · Prompt Engineering

Prompt engineering is not magic incantations — it's writing unambiguous
specifications for a very literal, very capable reader. The difference
between a mediocre and a great LLM feature is usually the prompt, not the
model. This module covers the handful of techniques that account for most of
the wins, and — just as important — how to iterate on prompts systematically
instead of guessing.

Setup for all snippets:

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

def ask(system: str, user: str, max_tokens: int = 500) -> str:
    response = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text
```

## 1. Clear, specific instructions

Vague prompts get vague answers. State the task, the audience, the format,
and the constraints:

```python
# Weak
print(ask("You are a helpful assistant.", "Summarize this: " + article))

# Strong
print(ask(
    "You are an editor for a technical newsletter.",
    f"""Summarize the article below for busy software engineers.

Requirements:
- Exactly 3 bullet points, each under 20 words
- Focus on what changed and why it matters, not background
- No marketing language

Article:
{article}""",
))
```

A good mental test: would a smart new intern produce the right output from
your prompt alone, with no chance to ask questions? If not, the model
probably won't either.

## 2. Structure with XML-style delimiters

When a prompt mixes instructions with data (articles, code, user input),
delimit the data so the model can't confuse the two. XML-style tags work
extremely well with Claude:

```python
user_prompt = f"""Classify the customer message into exactly one category.

<categories>
billing, bug_report, feature_request, other
</categories>

<message>
{customer_message}
</message>

Respond with only the category name."""
```

Tags also let you refer to sections by name ("using the rules in `<policy>`
..."), and they defang injection-ish inputs: text inside `<message>` is
clearly data, not instructions.

## 3. Few-shot examples

Showing beats telling. A couple of input→output examples pin down format and
edge-case behavior far better than prose:

```python
system = """You extract meeting actions from notes. Output one action per line
as: OWNER — ACTION — DUE. If a field is unknown, write "?".

<examples>
Input: "Sam will send the deck by Friday. We should revisit pricing."
Output:
Sam — send the deck — Friday
? — revisit pricing — ?

Input: "No actions this week."
Output:
NONE
</examples>"""

print(ask(system, "Input: \"Ana to file the bug today; Lee reviews it next sprint.\"\nOutput:"))
# Ana — file the bug — today
# Lee — review the bug — next sprint
```

Notice the examples include the awkward cases (missing owner, no actions) —
that's where few-shot earns its keep.

## 4. Role prompting

Assigning a role focuses tone, vocabulary, and priorities:

```python
system = (
    "You are a senior site reliability engineer doing a blameless incident "
    "review. You are precise, calm, and always distinguish observed facts "
    "from hypotheses."
)
print(ask(system, incident_notes))
```

Roles are most useful for *style and judgment*; they don't add knowledge the
model doesn't have.

## 5. Chain of thought: let the model think

For math, logic, and multi-step analysis, quality jumps when the model
reasons before answering. Ask it to think first — and keep the reasoning
separate from the final answer so your code can parse it:

```python
user_prompt = f"""A customer was billed $84.30 on a $70/month plan with 3 days
of a $2/day add-on and 8% tax. Is the bill correct?

Think through the calculation step by step inside <thinking> tags,
then give your final verdict inside <answer> tags as CORRECT or INCORRECT
with the expected amount."""

reply = ask("You are a billing auditor.", user_prompt, max_tokens=800)
answer = reply.split("<answer>")[1].split("</answer>")[0].strip()
print(answer)
```

Modern Claude models also have a built-in *adaptive thinking* mode that does
this internally; the explicit-tags pattern still matters because it works on
every model and makes the reasoning inspectable.

## 6. Iterate systematically, don't guess

The biggest prompt-engineering mistake is tweaking words at random and
eyeballing one output. Treat prompts like code:

1. **Build a tiny test set first** — 5–20 representative inputs, including
   nasty edge cases, with expected outputs where possible.
2. **Change one thing at a time** — one instruction, one example. If you
   change five things and quality moves, you've learned nothing.
3. **Run the whole set after each change** — a prompt tweak that fixes one
   case often breaks another.
4. **Version your prompts** — keep them in files under git, not inline
   strings scattered through code.

```python
TEST_CASES = [
    ("I was double charged this month!!", "billing"),
    ("app crashes when I rotate my phone", "bug_report"),
    ("would love a dark mode", "feature_request"),
    ("hi", "other"),
]

def run_evals(system_prompt: str) -> float:
    correct = 0
    for text, expected in TEST_CASES:
        got = ask(system_prompt, f"<message>{text}</message>").strip().lower()
        ok = got == expected
        correct += ok
        print(f"{'PASS' if ok else 'FAIL'}  expected={expected}  got={got}")
    return correct / len(TEST_CASES)

print("accuracy:", run_evals(CLASSIFIER_PROMPT_V2))
```

This is a miniature **eval harness** — the capstone project ships one, and
Level 2 covers full evaluation frameworks.

## Cheat sheet

| Technique | Use when | One-liner |
|-----------|----------|-----------|
| Specific instructions | Always | State task, audience, format, constraints explicitly |
| XML delimiters | Prompt mixes instructions and data | Wrap data in `<tags>`; refer to sections by name |
| Few-shot examples | Format or edge cases matter | 2–5 input→output pairs, including hard cases |
| Role prompting | Tone/judgment matters | "You are a senior X who values Y" |
| Chain of thought | Math, logic, multi-step analysis | "Think in `<thinking>`, answer in `<answer>`" |
| Output constraints | Downstream code parses the reply | "Respond with only the category name" |
| Systematic iteration | Always | Test set + one change at a time + rerun everything |

## Exercise

Build a sentiment classifier for product reviews with three labels
(`positive`, `negative`, `mixed`). Start with a naive one-line prompt and a
test set of 8 reviews you write yourself — include at least two genuinely
mixed reviews and one sarcastic one. Measure accuracy with a `run_evals`
harness like the one above. Then improve the prompt in three separate,
single-change steps (add delimiters, add few-shot examples, add an output
constraint), re-running the harness after each. Record the accuracy after
every step — you should see it climb.
