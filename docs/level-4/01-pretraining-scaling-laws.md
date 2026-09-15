---
description: "Pretraining & Scaling Laws Overview — Every model you've called throughout this course started as a randomly initialized transformer (Level 3, module 1)…"
---

# 01 · Pretraining & Scaling Laws Overview

Every model you've called throughout this course started as a randomly
initialized transformer (Level 3, module 1) and became useful through
**pretraining** — a massive, expensive, one-time process this module
explains at the level a practitioner needs: what the data pipeline looks
like, and what scaling laws let you predict before spending the compute.

## The pretraining objective

Pretraining trains the transformer on one deceptively simple task: predict
the next token, given every token before it, across a huge and diverse
text corpus. No labels, no task-specific supervision — the "label" for
every position is just the token that actually came next in real text:

```python
import numpy as np

def next_token_loss(logits: np.ndarray, target_token_id: int) -> float:
    """Cross-entropy loss for one position: -log P(correct token)."""
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return -np.log(probs[target_token_id] + 1e-12)

# Illustrative: one position's loss, given the model's predicted distribution
vocab_logits = np.array([2.1, 0.3, -1.0, 4.5, 0.8])   # 5-token toy vocab
actual_next_token = 3
loss = next_token_loss(vocab_logits, actual_next_token)
print(f"loss at this position: {loss:.3f}")
```

Averaged over every position in every document in a training batch, this
loss is what gradient descent minimizes — trillions of times, across a
training run that can last weeks to months on thousands of accelerators.
Nothing about the objective mentions "reasoning," "helpfulness," or
"following instructions" — those behaviors are downstream effects of what
next-token prediction over the right corpus, at sufficient scale, turns
out to produce, refined further by the alignment stage (module 2).

## The data pipeline

Raw internet-scale text isn't training-ready — a real pretraining pipeline
runs several stages before a document ever reaches the model:

```python
def pretraining_pipeline_stage(raw_document: str) -> str | None:
    # 1. Deduplication — near-identical documents waste compute and can
    #    cause memorization of specific repeated content.
    if is_near_duplicate(raw_document):
        return None

    # 2. Quality filtering — heuristic and model-based filters remove
    #    boilerplate, spam, and very low-quality text.
    if quality_score(raw_document) < QUALITY_THRESHOLD:
        return None

    # 3. Safety/PII filtering — remove or redact clearly harmful content
    #    and personally identifiable information before training ever sees it.
    cleaned = redact_pii(raw_document)

    # 4. Mixing — the final corpus blends sources (web text, books, code,
    #    academic papers) in deliberately chosen proportions, not naturally
    #    occurring ones — code and high-quality reference text are typically
    #    over-represented relative to their share of raw internet volume.
    return cleaned
```

The mixing ratios are themselves a major design decision with measurable
downstream effects — a corpus over-weighted toward code measurably
improves logical/structured reasoning even on non-code tasks, which is why
"what's in the training mix" is one of the first questions worth asking
when comparing models' differing strengths.

## Scaling laws

Scaling laws are empirical relationships — discovered by training many
smaller models at different sizes and data volumes and fitting a curve —
predicting how loss decreases as you increase model size, dataset size, or
compute, before you spend the money to train the large run itself:

```python
def predicted_loss(compute_flops: float, a: float = 400, alpha: float = 0.34) -> float:
    """Simplified power-law fit: loss ~ a * compute^-alpha (illustrative constants)."""
    return a * compute_flops ** -alpha

for flops in [1e18, 1e20, 1e22, 1e24]:
    print(f"compute={flops:.0e}  predicted_loss~{predicted_loss(flops):.3f}")
```

The key practical use of scaling laws is deciding, for a *fixed* compute
budget, the optimal split between model size and training data volume —
this is the "compute-optimal" question (famously addressed by the
Chinchilla scaling analysis): training a smaller model on proportionally
more data can reach lower loss for the same compute spend than training a
larger model on less data, which reshaped how labs allocate training
budgets industry-wide.

```python
def compute_optimal_split(compute_budget_flops: float) -> tuple[float, float]:
    """Illustrative: roughly balanced scaling of params and tokens with compute."""
    params = (compute_budget_flops / 6) ** 0.5
    tokens = compute_budget_flops / (6 * params)
    return params, tokens

params, tokens = compute_optimal_split(1e23)
print(f"~{params:.2e} parameters, ~{tokens:.2e} training tokens")
```

## Why scaling laws matter practically

You will likely never pretrain a frontier model yourself, but scaling laws
explain observable facts about the models you *do* call via API:

- **Why bigger models are usually more capable, with diminishing
  returns** — loss decreases as a power law, not linearly, so each further
  doubling of size/compute yields a smaller absolute improvement.
- **Why data quality has become as important as data quantity** — once
  labs run low on unique high-quality text at the scale scaling laws call
  for, filtering and mixing (above) become the lever instead of raw volume.
- **Why smaller, well-trained models can rival older larger ones** — a
  model trained compute-optimally on more, better data can match a larger,
  under-trained model's quality at a fraction of the inference cost.

## How It Actually Works

Pretraining is gradient descent on the transformer weights from Level 3,
module 1, applied at a scale that differs from the fine-tuning you did in
Level 3, module 6 only in degree, not in kind: the same forward
pass→loss→backward pass→weight update loop, run over a training corpus
many orders of magnitude larger, for a training duration many orders of
magnitude longer, spread across thousands of accelerators communicating
gradient updates with each other rather than one.

Scaling laws are possible because loss curves as a function of model
size/data/compute have empirically followed smooth power-law relationships
across many independent training runs, at many labs, across many
architectures — a genuinely surprising empirical regularity, not something
derived from first principles of the transformer architecture. This is
also their limitation: a scaling law is a fit to observed data, valid
within the regime it was measured in, and extrapolating far beyond that
regime (a very different data mix, an architecture change) is not
guaranteed to hold — which is why labs still run smaller-scale
confirmation experiments before committing a full training budget, rather
than trusting extrapolation blindly.

The reason next-token prediction alone produces behaviors as varied as
translation, summarization, and code generation is that all of these are,
in a sufficiently large and diverse training corpus, literally represented
as "what comes next" in real documents — a training corpus containing
millions of examples of text followed by its French translation, or code
followed by its explanation, gives the objective genuine signal to learn
those mappings, without ever being told "translation" is a task; scale
matters because larger models can represent and generalize more of these
implicit patterns simultaneously without interfering with each other.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Pretraining objective | Predict the next token — no task-specific labels |
| Data pipeline | Dedup → quality filter → PII/safety filter → deliberate source mixing |
| Scaling laws | Empirical power-law relationships between loss, size, data, compute |
| Compute-optimal | For fixed compute, there's a size/data split that minimizes loss |
| Diminishing returns | Power-law scaling means each doubling helps less than the last |
| Practical takeaway | Data quality/mix now rivals raw scale as the main quality lever |

## Exercise

Using the toy `predicted_loss` and `compute_optimal_split` functions
above, plot predicted loss against compute budget on a log-log scale for
budgets from 1e18 to 1e26 FLOPs, and separately plot the compute-optimal
parameter count and token count over the same range. Write a short
explanation, in your own words, of why the compute-optimal parameter
count grows more slowly than total compute — tie your answer back to the
power-law loss curve, not just the arithmetic.
