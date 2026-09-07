# 04 · Structured Output

The moment an LLM's answer feeds into *code* instead of a human's eyes, prose
stops being acceptable — you need JSON that parses every time and matches a
schema your program expects. This module builds up the reliable-JSON toolkit:
schema-in-prompt, pydantic validation, retry-on-invalid, and the API's native
structured-output mode.

Setup:

```bash
pip install anthropic python-dotenv pydantic
```

```python
from dotenv import load_dotenv
import anthropic, json

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"
```

## Step 1: Ask for JSON, show the schema

The baseline technique: describe the exact schema in the prompt and forbid
everything else.

```python
SYSTEM = """You extract structured data from job postings.

Respond with ONLY a JSON object — no markdown fences, no commentary — matching:
{
  "title": string,
  "company": string,
  "remote": boolean,
  "salary_min": integer or null,
  "salary_max": integer or null,
  "skills": string[]        // lowercase, deduplicated
}"""

posting = """Senior Backend Engineer at Fjordworks (fully remote, EU hours).
$140k–$175k. You'll own our Go services and Postgres schemas; Kubernetes a plus."""

response = client.messages.create(
    model=MODEL, max_tokens=500, system=SYSTEM,
    messages=[{"role": "user", "content": posting}],
)
raw = response.content[0].text
data = json.loads(raw)          # works... usually
print(data["skills"])           # ['go', 'postgres', 'kubernetes']
```

This works most of the time — and "most of the time" is exactly the problem.
Occasionally you'll get a markdown fence, a trailing comment, or a field with
the wrong type. Production code needs guarantees.

## Step 2: Tolerant extraction

A cheap robustness layer: strip fences and grab the outermost JSON object
before parsing.

```python
def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]          # drop the fence
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in: {text[:120]!r}")
    return json.loads(text[start : end + 1])
```

## Step 3: Validate with pydantic

Parsing proves it's JSON; **validation** proves it's *your* JSON. Pydantic
models declare the schema once and enforce types, requiredness, and
constraints:

```python
from pydantic import BaseModel, Field, ValidationError

class JobPosting(BaseModel):
    title: str
    company: str
    remote: bool
    salary_min: int | None
    salary_max: int | None
    skills: list[str] = Field(min_length=1)

try:
    job = JobPosting.model_validate(data)
    print(job.title, "at", job.company)
except ValidationError as e:
    print("model returned wrong shape:", e)
```

Bonus: pydantic can *generate* the JSON schema for your prompt, so prompt and
validator can never drift apart:

```python
schema_str = json.dumps(JobPosting.model_json_schema(), indent=2)
SYSTEM = f"Respond with ONLY a JSON object matching this JSON Schema:\n{schema_str}"
```

## Step 4: Retry on invalid

When validation fails, don't give up — feed the error back and let the model
fix its own output. One retry resolves the vast majority of failures:

```python
def get_structured(user_text: str, max_attempts: int = 3) -> JobPosting:
    messages = [{"role": "user", "content": user_text}]
    for attempt in range(max_attempts):
        response = client.messages.create(
            model=MODEL, max_tokens=500, system=SYSTEM, messages=messages,
        )
        raw = response.content[0].text
        try:
            return JobPosting.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:
            # Append the bad answer + the error, and ask again
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"That was invalid: {e}\nRespond again with ONLY corrected JSON."})
    raise RuntimeError(f"no valid JSON after {max_attempts} attempts")

job = get_structured(posting)
```

This *generate → validate → repair* loop is one of the most important
patterns in LLM engineering — you'll reuse it for tools, agents, and evals.

## Step 5: Native structured outputs (the modern way)

Recent Anthropic API versions can **enforce** a schema server-side, so the
response is guaranteed-valid JSON — no extraction or retries needed. The SDK
even wires it straight into pydantic:

```python
response = client.messages.parse(
    model=MODEL,
    max_tokens=500,
    messages=[{"role": "user", "content": posting}],
    output_format=JobPosting,          # pydantic model in, pydantic model out
)
job = response.parsed_output           # a validated JobPosting instance
print(job.salary_max)                  # 175000
```

So why learn steps 1–4 at all? Because (a) not every provider or model
supports native structured output, (b) schema features like recursion or
numeric ranges aren't always supported so prompt+validate remains the
fallback, and (c) the repair-loop pattern generalizes far beyond JSON.

## How It Actually Works

The model fundamentally only knows how to emit one token at a time from
its vocabulary; it has no native concept of "JSON" as a data type. Getting
reliable structured output relies on one of two underlying mechanisms.

The first — prompting the model to "return JSON" — works purely because
JSON is extremely common in the model's training data, so the learned
distribution over next-tokens naturally favors JSON-shaped continuations
after a JSON-shaped instruction and example. It's still just next-token
prediction; there's no validation happening, which is exactly why it can
still emit a trailing comma, an unescaped quote, or prose before the `{`.

Native structured-output / JSON-mode features work differently and far
more reliably: they use **constrained decoding**. Instead of sampling
freely from the full vocabulary at each step, the server intersects the
model's output distribution with a grammar or JSON-schema-derived state
machine, zeroing out the probability of any token that would make the
output invalid at that position (e.g., zeroing every token except `"`,
`}`, or a digit if the schema expects a number next). The model still
picks among only the tokens consistent with your schema, sampled according
to its normal (renormalized) probabilities — so it's guaranteed
syntactically valid, but the *values* it fills in are still generated
predictions, not database lookups, and can still be factually wrong even
though they're structurally perfect.

This is also why constrained decoding is a server-side feature: it
requires access to the raw logits before sampling, something a client
calling the API over HTTP cannot do once it only receives the finished
text.

## Cheat sheet

| Technique | Guarantee | Cost |
|-----------|-----------|------|
| Schema in prompt | None — usually works | Free |
| "ONLY JSON, no fences" instruction | Fewer format slips | Free |
| `extract_json` tolerant parsing | Survives fences/prose | Free |
| Pydantic `model_validate` | Correct types & fields, or a clear error | Free |
| Retry with error feedback | Self-heals most failures | Extra API calls |
| `client.messages.parse(output_format=Model)` | Schema-valid by construction | Native, Anthropic API |
| `model_json_schema()` in prompt | Prompt and validator never drift | Free |

## Exercise

Build an expense-report extractor. Define a pydantic model `Expense` with
`merchant: str`, `date: str`, `total: float`, `currency: str` (3-letter
code), and `category` restricted to a `Literal` of five values. Write ten
messy one-line expense descriptions ("ubr 14.20 eur airport→hotel 3/7",
"Team lunch at Nonna's — $84.10, March 9"), then process them with the
full `get_structured` repair loop. Log every retry that happens. Finally,
switch to `client.messages.parse` and compare: how many retries did the
native mode eliminate?
