# 06 · Evaluation Frameworks

"It looked good when I tried it" is not a testing strategy. Once a prompt
or agent ships, you need a repeatable way to know whether a change made
things better or worse — the LLM equivalent of a test suite. This module
builds golden datasets, LLM-as-judge scoring, and a regression suite that
runs in CI.

## Golden datasets

A golden dataset is a fixed set of representative inputs with known-good
(or known-acceptable) outputs, checked into version control like test
fixtures:

```python
# evals/ticket_classifier/golden.jsonl
# {"input": "My card was charged twice this month", "expected_category": "billing"}
# {"input": "The app crashes every time I open settings", "expected_category": "technical"}
# {"input": "I can't remember my password", "expected_category": "account"}

import json

def load_golden(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]

golden = load_golden("evals/ticket_classifier/golden.jsonl")
```

Build this set from real production examples (anonymized) plus
deliberately-chosen edge cases — ambiguous inputs, adversarial phrasing,
known past failures. A golden set that's all easy cases will pass forever
and tell you nothing.

## Exact-match and rubric-based scoring

For tasks with a single correct answer (classification, extraction),
score with plain code — no LLM needed:

```python
def run_classifier_eval(golden: list[dict]) -> float:
    correct = 0
    for case in golden:
        predicted = classify_ticket(case["input"])       # your function under test
        correct += predicted == case["expected_category"]
    return correct / len(golden)

accuracy = run_classifier_eval(golden)
print(f"accuracy: {accuracy:.1%}")
```

For open-ended tasks (summaries, drafted replies, explanations) there's
rarely one correct string, so exact match is meaningless — that's where
LLM-as-judge comes in.

## LLM-as-judge

Use a second model call to grade output against a rubric, returning a
structured verdict you can aggregate like any other metric:

```python
from dotenv import load_dotenv
import anthropic, json

load_dotenv()
client = anthropic.Anthropic()
JUDGE_MODEL = "claude-sonnet-5"

JUDGE_SCHEMA = {
    "name": "record_verdict",
    "description": "Record the evaluation verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "meets_rubric": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "score_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["meets_rubric", "reasoning", "score_1_to_5"],
    },
}

def judge(task_input: str, candidate_output: str, rubric: str) -> dict:
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=400,
        tools=[JUDGE_SCHEMA], tool_choice={"type": "tool", "name": "record_verdict"},
        messages=[{"role": "user", "content":
            f"Rubric:\n{rubric}\n\nTask input:\n{task_input}\n\n"
            f"Candidate output:\n{candidate_output}\n\n"
            f"Evaluate the candidate output strictly against the rubric."}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")

verdict = judge(
    task_input="Draft a reply to: 'My order arrived damaged.'",
    candidate_output="Sorry about that! We'll send a replacement today, no need to return the damaged item.",
    rubric="Reply must: (1) apologize, (2) offer a concrete resolution, (3) not blame the customer.",
)
print(verdict)
```

Judge reliability matters as much as the thing being judged — write the
rubric as concretely as you'd write a code review checklist, use a
consistent judge model across runs so scores are comparable over time, and
periodically spot-check judge verdicts against human judgment to catch
drift or a rubric that's ambiguous enough to produce inconsistent grading.

## Regression suites for prompts

Wire the golden set + scoring into a script (or CI job) that fails the
build when a prompt or model change regresses quality:

```python
# evals/run_eval.py
import sys

THRESHOLD = 0.90

def main():
    golden = load_golden("evals/ticket_classifier/golden.jsonl")
    accuracy = run_classifier_eval(golden)
    print(f"accuracy: {accuracy:.1%} (threshold: {THRESHOLD:.0%})")
    if accuracy < THRESHOLD:
        print("FAIL: regression detected")
        sys.exit(1)
    print("PASS")

if __name__ == "__main__":
    main()
```

```yaml
# .github/workflows/eval.yml
- name: Run prompt regression eval
  run: python evals/run_eval.py
```

Run this on every PR that touches a prompt, a model version, or a tool
schema — the same discipline as running unit tests on every code PR, for
the same reason: prompts are code, and untested prompt changes regress
silently in production instead of failing a build.

## Eval-driven development

Once a suite exists, invert the workflow: write the failing eval case
*first* when you discover a bug ("customer said 'billing issue'
lowercase, misclassified as technical"), confirm it fails, then fix the
prompt or code until it passes — exactly TDD, applied to prompts instead
of functions.

## How It Actually Works

Exact-match evaluation needs no special mechanism — it's ordinary
assertion-based testing where the "function under test" happens to call an
LLM. LLM-as-judge is more subtle: it works because judging text against an
explicit rubric is, like self-critique in module 1, a comparison task
rather than a generation-from-nothing task — the judge model conditions on
both the rubric and the candidate output already being present in its
input, and only has to produce a token sequence expressing agreement or
disagreement with specific stated criteria, which is a narrower and more
constrained prediction than open-ended writing.

This is also exactly why judge quality degrades with vague rubrics: an
ambiguous rubric under-constrains the judge's next-token distribution the
same way an ambiguous system prompt under-constrains a generator's, so the
judge's verdict becomes as inconsistent run-to-run as any other
under-specified completion task — tightening the rubric is doing the same
job as tightening a tool schema in module 4, just applied to the grading
step instead of the answering step.

Regression detection works purely as software engineering: a numeric
threshold gate on a deterministic (or averaged-over-N-runs, since LLM
outputs vary) score is no different in kind from a code-coverage or
performance-budget gate — the novelty is entirely in what's being measured
(a model's behavior on a fixed dataset), not in any new evaluation
mechanism.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Golden dataset | Fixed inputs + expected outputs, versioned like test fixtures |
| Exact-match scoring | Use for classification/extraction with one correct answer |
| LLM-as-judge | Use for open-ended output; grade against an explicit rubric |
| Judge consistency | Pin the judge model; spot-check against human judgment |
| Regression suite | Threshold-gated script run in CI on every prompt/model change |
| Eval-driven dev | Write the failing eval case first when you find a bug |

## Exercise

Build a golden dataset of 15 support-ticket examples (5 each: billing,
technical, account) including at least 3 deliberately ambiguous ones.
Write an exact-match eval for a classifier prompt, then write an
LLM-as-judge eval scoring a "draft a reply" prompt against a 3-point
rubric. Wire both into a `run_eval.py` with a pass/fail threshold, and
show it failing when you intentionally weaken the classifier's system
prompt.
