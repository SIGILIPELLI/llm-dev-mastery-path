---
description: "Capstone Project — This capstone pulls together every module in Level 4 into one coherent build: a small but genuinely production-shaped LLM platform…"
---

# 10 · Capstone Project

This capstone pulls together every module in Level 4 into one coherent
build: a small but genuinely production-shaped LLM platform serving
multiple simulated teams, with routing, cost controls, safety layers,
evaluation, and compliance all present — not as separate toy exercises,
but as one integrated system where each piece depends on the others the
way they do in a real deployment.

## What you're building

A "Platform Service" that sits in front of one or more models and serves
requests from at least two simulated internal "teams" (e.g., a
`support-bot` team and an `internal-tools` team), with these components
wired together end to end:

1. A **gateway** (module 3) with provider adapters, a routing table, and
   quota enforcement per team.
2. **Model tiering with escalation** (module 6) so simple requests use a
   cheap model and only escalate when needed.
3. A **safety layer** (modules 4 and 8): input classification for
   obvious attack patterns, output filtering, and least-privilege tool
   scoping for any agentic team.
4. **Continuous evaluation** (module 5): a sampling pipeline that scores
   a percentage of live traffic and a rolling quality monitor that
   alerts on drift.
5. **Cost attribution** (module 6): per-team spend tracking, visible in
   one place.
6. **An audit trail and PII redaction layer** (module 9): every
   consequential action logged in a tamper-evident chain, with PII
   redacted before storage.
7. **Latency budgets** (module 7): p99 tracked per team, with a
   documented fallback path when a budget is at risk of being blown.

## Reference architecture

```python
from dataclasses import dataclass, field
from enum import Enum
import time

class ModelTier(Enum):
    SMALL = "small"
    LARGE = "large"

@dataclass
class PlatformRequest:
    team_id: str
    task_type: str
    messages: list[dict]
    user_id: str

@dataclass
class PlatformResponse:
    text: str
    tier_used: ModelTier
    latency_ms: float
    cost: float
    flagged: bool = False

class LLMPlatform:
    def __init__(self, small_model, large_model, gateway_quota, audit_log,
                 quality_monitor, pii_redactor, tool_scopes: dict):
        self.small_model = small_model
        self.large_model = large_model
        self.gateway_quota = gateway_quota
        self.audit_log = audit_log
        self.quality_monitor = quality_monitor
        self.pii_redactor = pii_redactor
        self.tool_scopes = tool_scopes

    def handle_request(self, request: PlatformRequest) -> PlatformResponse:
        # 1. Quota check (module 3)
        if not self.gateway_quota.check_quota(request.team_id):
            raise PermissionError(f"Team {request.team_id} exceeded quota")

        # 2. Safety input check (modules 4, 8)
        if self._looks_like_injection(request.messages):
            self.audit_log.record(request.user_id, "blocked_injection_attempt", {"team": request.team_id})
            raise PermissionError("Request blocked by input safety filter")

        # 3. PII redaction before anything is logged or sent onward (module 9)
        redacted_messages, pii_found = self._redact_messages(request.messages)

        # 4. Tiered routing with escalation (module 6)
        start = time.monotonic()
        result = self.small_model.generate_with_confidence(redacted_messages)
        tier_used = ModelTier.SMALL
        if result.confidence < 0.7:
            result = self.large_model.generate_with_confidence(redacted_messages)
            tier_used = ModelTier.LARGE
        latency_ms = (time.monotonic() - start) * 1000

        # 5. Output safety check (module 8)
        flagged = self._output_flagged(result.text)

        # 6. Cost + audit (modules 6, 9)
        cost = self._compute_cost(tier_used, redacted_messages, result.text)
        self.audit_log.record(request.user_id, "request_served", {
            "team": request.team_id, "tier": tier_used.value, "cost": cost,
            "pii_redacted": pii_found, "flagged": flagged,
        })

        # 7. Sample for continuous eval (module 5)
        self.quality_monitor.maybe_sample(request, result.text)

        return PlatformResponse(text=result.text, tier_used=tier_used,
                                  latency_ms=latency_ms, cost=cost, flagged=flagged)
```

This is deliberately a sketch, not a finished library — every method
prefixed `_` is where you plug in the real implementation from the
corresponding module (`_looks_like_injection` from module 8's patterns,
`_redact_messages` from module 9's `redact_pii`, `_compute_cost` from
module 6's cost attribution, and so on).

## Build phases

1. **Phase 1 — the gateway skeleton.** Get `LLMPlatform.handle_request`
   running end to end with a single mock model, no safety or eval logic
   yet — just quota checking and a hardcoded response. Verify two teams
   get independently enforced quotas.
2. **Phase 2 — tiering and cost.** Add the small/large model escalation
   and per-team cost attribution. Verify cost is correctly summed per
   team and that escalation actually triggers on low-confidence
   responses.
3. **Phase 3 — safety.** Add input injection detection (module 8) and an
   output content filter (module 4). Build a 20-case red-team suite and
   confirm the platform blocks the attack cases while passing the
   control cases.
4. **Phase 4 — compliance.** Add PII redaction before storage and the
   hash-chained audit log. Verify a deletion request can locate and
   remove a specific user's data across every store the platform writes
   to.
5. **Phase 5 — continuous eval and latency.** Wire up traffic sampling
   into an LLM-as-judge scoring pipeline and a rolling quality monitor.
   Add p99 latency tracking per team and a documented fallback when a
   team's p99 is trending toward its budget.

## Acceptance checklist

Treat this as the capstone's actual "done" bar — each item should be
independently verifiable, not just asserted:

- [ ] Two teams have independently enforced quotas; exceeding one doesn't
      affect the other.
- [ ] A request that a small model handles confidently never invokes the
      large model (verified via a call counter).
- [ ] A 20-case red-team suite (10 attacks, 10 controls) passes with the
      attacks blocked and the controls answered normally.
- [ ] PII in a test request never appears unredacted in any log or audit
      entry.
- [ ] The audit log's `verify_chain` correctly detects tampering when a
      historical entry is modified.
- [ ] A per-team cost report sums correctly against a known set of
      requests with known token counts.
- [ ] p99 latency is tracked and visible per team, separate from the
      mean.
- [ ] Running `mkdocs`-style documentation (or a README) explains the
      architecture well enough that someone unfamiliar with the build
      could operate it.

## How It Actually Works

The reason this capstone is structured as one integrated system rather
than nine separate exercises is that the modules in this level are not
independent features bolted onto a model — they're different views of
the same underlying constraint: a model is a powerful, imperfect,
non-deterministic component, and a production platform's job is to wrap
it with enough structure that the *system's* behavior is more reliable,
accountable, and controllable than the model's raw behavior alone. Quota
enforcement, tiering, safety filtering, and audit logging all sit at the
same architectural layer — around the model, not inside it — because none
of them can be reliably achieved by prompting the model differently; they
require software that constrains what the model is allowed to affect,
independent of what the model itself decides to do.

This is also why the acceptance checklist emphasizes independent
verification over "it seems to work": every property on that list — quota
isolation, tamper detection, redaction completeness — is a property of
the *system*, observable only by testing the system's boundaries
directly (trying to exceed a quota, trying to tamper with a log, trying
to inject an attack), not by inspecting any single model response. That
is the central lesson of Level 4: at platform scale, correctness lives in
the architecture surrounding the model at least as much as in the model
itself.

## Wrapping up

Completing this capstone means you've built — end to end, even if at
small scale — the same architectural shape used by real production LLM
platforms: a gateway, tiered routing, layered safety, continuous
evaluation, cost accountability, and compliance-grade logging, all
integrated rather than siloed. That shape is the actual transferable
skill; the specific model, provider, and scale will keep changing, but a
platform built this way absorbs those changes at the adapter and
routing-table layer without requiring the rest of the system to be
rebuilt.

## Exercise

Implement all five build phases above against a mock model pair (a fast
"small" model and a slower "large" model, both simple functions you
control so behavior is predictable), run the full acceptance checklist,
and write up a short incident postmortem for one deliberately-injected
failure: pick one guardrail (quota, safety filter, or audit log), disable
it, run the same red-team and load tests, and document exactly what
breaks and how the system's own monitoring would have surfaced the
regression.
