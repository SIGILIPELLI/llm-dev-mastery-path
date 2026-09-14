# 05 · Advanced Evaluation at Scale

Level 2's eval module covered building a golden-example regression suite
you run before shipping a prompt change. That works well at the scale of
one team shipping occasionally. At platform scale — many teams, frequent
model and prompt changes, live traffic that doesn't match any offline
test set exactly — evaluation needs to become continuous and partly
online, not just a pre-deploy checklist. This module covers that scaling.

## Why offline eval suites stop being enough

A static golden set catches regressions on the exact cases you thought to
write down. It can't catch:

- **Distribution drift** — real user inputs shift over time in ways your
  eval set, written months ago, doesn't reflect.
- **Silent model updates** — a provider updates a model version behind a
  stable API name, and behavior shifts on inputs your eval set doesn't
  happen to cover.
- **Interaction effects at scale** — a prompt change that looks fine on
  50 golden examples but degrades a rare-but-real input pattern that only
  shows up at 1-in-10,000 request volume.

Continuous evaluation extends the golden-set approach with a live,
online layer sampling real production traffic.

## Continuous eval pipeline architecture

```python
from dataclasses import dataclass
import random

@dataclass
class ProductionSample:
    request_id: str
    prompt: str
    response: str
    model_version: str
    timestamp: float

def sample_for_eval(request: ProductionSample, sample_rate: float = 0.01) -> bool:
    # sample a small percentage of live traffic for ongoing quality review —
    # not every request, since scoring every request is expensive at scale
    return random.random() < sample_rate

def enqueue_for_scoring(sample: ProductionSample, queue):
    if sample_for_eval(sample):
        queue.push({
            "request_id": sample.request_id,
            "prompt": sample.prompt,
            "response": sample.response,
            "model_version": sample.model_version,
        })
```

Sampled requests flow into a scoring pipeline — typically a mix of
automated LLM-as-judge scoring (Level 2's pattern, run continuously
rather than on a fixed golden set) and periodic human review of a smaller
subset, since automated judges themselves need calibration checks against
human judgment.

```python
def score_sampled_batch(samples: list[dict], judge_model, rubric: str) -> list[dict]:
    scored = []
    for sample in samples:
        verdict = judge_model.judge(
            f"{rubric}\n\nPrompt: {sample['prompt']}\nResponse: {sample['response']}\n"
            f"Score 1-5 and give one sentence of reasoning."
        )
        scored.append({**sample, "score": parse_score(verdict), "reasoning": verdict})
    return scored
```

## A/B testing model and prompt changes

Rolling out a prompt or model change to 100% of traffic at once means
you find out about a regression from user complaints. Splitting traffic
lets you measure the difference statistically before full rollout:

```python
import hashlib

def assign_variant(user_id: str, variant_a: str, variant_b: str, split: float = 0.5) -> str:
    # deterministic hash-based assignment: same user always gets the same
    # variant across requests, which matters for consistent experience
    # and for measuring per-user effects rather than per-request noise
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return variant_a if (h % 10000) / 10000 < split else variant_b

def record_experiment_outcome(user_id: str, variant: str, outcome_metrics: dict, store):
    store.append({
        "user_id": user_id,
        "variant": variant,
        "metrics": outcome_metrics,   # e.g. thumbs_up, task_completed, latency_ms
        "timestamp": time.time(),
    })
```

```python
import statistics

def analyze_experiment(results_a: list[float], results_b: list[float]) -> dict:
    mean_a, mean_b = statistics.mean(results_a), statistics.mean(results_b)
    # a real analysis uses a proper significance test (e.g. Welch's t-test);
    # this sketch shows the structure of what you're measuring
    pooled_std = statistics.pstdev(results_a + results_b)
    effect_size = (mean_b - mean_a) / pooled_std if pooled_std else 0
    return {"mean_a": mean_a, "mean_b": mean_b, "effect_size": effect_size,
            "n_a": len(results_a), "n_b": len(results_b)}
```

Don't ship a change based on a difference that isn't statistically
significant relative to your sample size — a 2% improvement measured on
50 samples per variant is noise, not signal; use a proper significance
test and a pre-registered minimum sample size before drawing conclusions.

## Quality monitoring dashboards and alerting

Continuous scoring is only useful if degradation actually gets noticed.
Track eval scores as a time series, not a one-off number, and alert on
meaningful drops:

```python
from collections import deque

class RollingQualityMonitor:
    def __init__(self, window_size: int = 500, alert_threshold: float = 0.15):
        self.scores = deque(maxlen=window_size)
        self.baseline = None
        self.alert_threshold = alert_threshold

    def record(self, score: float):
        self.scores.append(score)

    def check_drift(self) -> bool:
        if self.baseline is None or len(self.scores) < self.scores.maxlen:
            if len(self.scores) == self.scores.maxlen:
                self.baseline = sum(self.scores) / len(self.scores)
            return False
        current_avg = sum(self.scores) / len(self.scores)
        relative_drop = (self.baseline - current_avg) / self.baseline
        return relative_drop > self.alert_threshold
```

Alert on relative drops against a rolling baseline rather than an
absolute fixed threshold — absolute thresholds need constant manual
retuning as the underlying task or model changes, while a rolling
baseline adapts automatically and still flags a genuine regression.

## How It Actually Works

The core reason continuous eval catches things static suites miss is a
sampling argument: a golden set of even a few hundred examples is a fixed,
tiny sample of an effectively unbounded input space, chosen by humans who
can only anticipate the failure modes they've already thought of. Live
traffic sampling, even at a low rate, draws from the *actual* distribution
of real usage, including the long tail of inputs no one wrote a test case
for — so it surfaces failure modes that are structurally invisible to any
offline set, no matter how carefully curated, simply because those
failure modes only exist in the live distribution.

A/B testing's statistical logic matters for the same underlying reason
eval sets need held-out data (Level 2/3): any single side-by-side
comparison is subject to random variation in which users, prompts, and
edge cases happened to land in each variant. The point of a significance
test and minimum sample size is to distinguish "this metric moved because
the change matters" from "this metric moved because of which particular
100 users happened to be sampled into each bucket" — the same
overfitting-to-noise risk as trusting training loss over held-out eval
score, applied to a live traffic split instead of a training run.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Continuous eval | Sample live traffic continuously, not just a static golden set |
| LLM-as-judge at scale | Automates scoring of sampled production traffic |
| A/B testing | Deterministic hash-based assignment; measure with a significance test |
| Effect size | Whether a measured difference is meaningful, not just non-zero |
| Rolling baseline | Alert on relative drift from a moving baseline, not a fixed threshold |
| Why sampling matters | Live traffic covers failure modes no static golden set anticipated |

## Exercise

Instrument a mock chat endpoint to sample 5% of requests into a scoring
queue, score them with an LLM-as-judge rubric, and feed scores into the
`RollingQualityMonitor` above. Simulate a quality regression by
injecting worse responses into 20% of one traffic segment, and confirm
the monitor's `check_drift()` fires within a reasonable number of
samples. Then design an A/B test comparing two prompt variants on the
same endpoint, and compute the effect size needed before you'd consider
rolling the winning variant out to 100% of traffic.
