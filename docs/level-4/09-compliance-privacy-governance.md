---
description: "Compliance, Privacy & Governance — Everything covered so far in Level 4 — alignment, safety, security, cost, latency — is about making an LLM system work…"
---

# 09 · Compliance, Privacy & Governance

Everything covered so far in Level 4 — alignment, safety, security, cost,
latency — is about making an LLM system work well and safely. This
module covers the layer that makes it *operable under scrutiny*: data
retention policy, PII handling, audit trails, and the general shape of
regulatory considerations that apply once an LLM system touches real
user data at any meaningful scale. This is not legal advice, and specific
regulatory obligations vary by jurisdiction and industry — the goal here
is the engineering patterns that make compliance possible, so a legal or
compliance team has something real to work with.

## Data retention: knowing what you're keeping, and for how long

An LLM platform touches data at several distinct points — inputs,
outputs, logs, traces (Level 3, module 9), eval samples (module 5),
cached responses (module 6) — and each can silently accumulate
sensitive data indefinitely if retention isn't designed deliberately:

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RetentionPolicy:
    data_category: str
    retention_days: int
    reason: str

retention_policies = [
    RetentionPolicy("raw_prompts_with_pii", 30, "Minimize PII exposure window"),
    RetentionPolicy("aggregated_metrics", 730, "No PII; useful for long-term trend analysis"),
    RetentionPolicy("eval_samples_anonymized", 365, "PII stripped before storage"),
    RetentionPolicy("audit_logs", 2555, "Common 7-year regulatory retention for audit trails"),
]

def is_expired(created_at: datetime, policy: RetentionPolicy) -> bool:
    return datetime.utcnow() - created_at > timedelta(days=policy.retention_days)

def purge_expired(records: list[dict], policy: RetentionPolicy) -> list[dict]:
    return [r for r in records if not is_expired(r["created_at"], policy)]
```

The pattern that matters: define retention *per data category*, not one
blanket policy — raw prompts containing PII, anonymized aggregates, and
audit logs have genuinely different retention needs, and a single global
retention setting is either too short for audit purposes or too long for
PII minimization.

## PII detection and redaction

Detecting and redacting personally identifiable information before it's
stored, logged, or sent to a third-party model provider reduces exposure
at the source:

```python
import re

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    found_counts = {}
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        matches = pattern.findall(redacted)
        if matches:
            found_counts[label] = len(matches)
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted, found_counts
```

Regex-based detection catches well-structured PII (emails, phone
numbers, SSNs) but misses unstructured PII (a name mentioned in prose, an
address written informally) — for higher-stakes applications, pair regex
with a dedicated PII-detection model or service, and treat regex
detection as a fast first pass, not a complete solution. Apply redaction
*before* data leaves your own boundary — before it's sent to a
third-party provider or written to a log store, not after.

## Audit trails for agentic actions

Once an LLM system takes real actions (Level 3's tool use, this level's
security module), a complete, tamper-evident record of what happened and
why is both a security control and, in regulated contexts, often a
compliance requirement:

```python
import json
import hashlib
from datetime import datetime

class AuditLog:
    def __init__(self, store):
        self.store = store
        self.last_hash = "0" * 64   # genesis hash

    def record(self, actor: str, action: str, details: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "actor": actor,
            "action": action,
            "details": details,
            "prev_hash": self.last_hash,
        }
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = hashlib.sha256(entry_bytes + self.last_hash.encode()).hexdigest()
        self.last_hash = entry["hash"]
        self.store.append(entry)
        return entry

    def verify_chain(self, entries: list[dict]) -> bool:
        # each entry's hash depends on the previous entry's hash — tampering
        # with any past entry breaks every hash after it, making tampering detectable
        prev_hash = "0" * 64
        for entry in entries:
            expected = entry["hash"]
            entry_copy = {k: v for k, v in entry.items() if k != "hash"}
            entry_bytes = json.dumps(entry_copy, sort_keys=True).encode()
            actual = hashlib.sha256(entry_bytes + prev_hash.encode()).hexdigest()
            if actual != expected:
                return False
            prev_hash = expected
        return True
```

Log every consequential action an agent takes — not just the final
outcome, but which tool was called, with what arguments, on whose
authority, and what the model's stated reasoning was — because the audit
trail is what lets you reconstruct *why* an agent did something months
later, which is exactly what an incident review or regulatory inquiry
needs.

## Consent, purpose limitation, and data minimization

Three governance principles recur across most privacy regulation
(GDPR, CCPA, and sector-specific rules alike), independent of the exact
legal text in any one jurisdiction:

- **Purpose limitation** — data collected for one purpose (e.g.,
  answering a support query) shouldn't be silently reused for another
  (e.g., training a model) without separate basis or consent.
- **Data minimization** — send a model only the data actually needed for
  the task, not everything available "in case it helps" — a support
  agent summarizing a ticket doesn't need the user's full order history
  in its context unless the task requires it.
- **User rights** — access, correction, and deletion rights mean your
  system needs a real mechanism to locate and remove a specific user's
  data across every store it touched (raw logs, caches, eval samples,
  vector indexes), not just the primary database.

```python
def handle_deletion_request(user_id: str, data_stores: list):
    results = {}
    for store in data_stores:
        # every store that might hold user data needs to implement this —
        # primary DB, logs, cache, vector index, eval samples, everywhere
        deleted_count = store.delete_by_user_id(user_id)
        results[store.name] = deleted_count
    return results
```

Design for deletion from day one — retrofitting "find every place this
user's data might exist" across logs, caches, vector stores, and eval
snapshots after the fact is a much larger project than building each
store with a `user_id`-indexed deletion path from the start.

## How It Actually Works

The audit log's hash-chaining mechanism (each entry's hash depends on
the previous entry's hash, forming a chain) is the same structural idea
behind blockchains and Git commit history: it makes tampering with any
past entry detectable, because changing one entry changes its hash,
which invalidates every subsequent entry's hash in the chain. This
doesn't prevent someone with write access from tampering — it makes
tampering *evident* upon verification, which is the property audit trails
actually need: not that data can never be altered, but that alteration
can always be detected after the fact.

The reason PII minimization needs to happen before data leaves your
boundary, not after, is that once data has been transmitted to a
third-party provider, logged to a store you don't fully control, or used
in a training run, deletion or correction becomes structurally harder or
sometimes impossible — a model that has already been trained on data
can't have that specific data "un-trained" out of its weights the way a
database row can be deleted, since the influence is distributed across
many parameters rather than stored at a retrievable address. This
asymmetry (easy to prevent inclusion, hard to reverse it after the fact)
is why data minimization and purpose limitation are enforced as upstream
gates rather than downstream cleanup.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Retention policy | Define per data category, not one global rule |
| PII redaction | Apply before data leaves your boundary; regex + model for full coverage |
| Audit trail | Hash-chained, tamper-evident log of every consequential action |
| Purpose limitation | Don't silently repurpose data collected for one use into another |
| Data minimization | Send models only what the task needs, not everything available |
| Deletion by design | Every store needs a `user_id`-indexed deletion path from day one |

## Exercise

Implement `redact_pii` and run it against 20 synthetic sample prompts
containing a mix of emails, phone numbers, and SSNs (never use real
personal data — generate synthetic examples). Report the detection rate
and any false positives. Then build the `AuditLog` class, record 10
sequential agent actions, verify the chain with `verify_chain`, and
demonstrate that tampering with one historical entry causes verification
to fail from that point forward.
