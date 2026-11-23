---
title: "LLM Development Cheat Sheet: Quick Reference, Beginner to Advanced"
description: "A free LLM Development cheat sheet: 39 quick-reference tables from every level of the LLM Development Mastery Path, each linked to its full lesson."
---

# LLM Development Cheat Sheet

Quick-reference tables from **39 lessons** of the free LLM Development Mastery Path, in course order from beginner to advanced. Each section links to the full lesson with explanations and runnable examples.

## Level 1 · Entry

### [01 · The LLM Landscape &amp; Setup](level-1/01-llm-landscape-setup.md)

| Item | Detail |
|------|--------|
| Install | `pip install anthropic python-dotenv` |
| Key storage | `.env` file with `ANTHROPIC_API_KEY=...`, listed in `.gitignore` |
| Load key | `from dotenv import load_dotenv; load_dotenv()` |
| Client | `client = anthropic.Anthropic()` — reads the key from the environment |
| Basic call | `client.messages.create(model=..., max_tokens=..., messages=[...])` |
| Workhorse model | `claude-sonnet-5` |
| Cheap/fast model | `claude-haiku-4-5` |
| Response text | iterate `response.content`, use blocks where `block.type == "text"` |
| Golden rule | Never hardcode or commit an API key |

### [02 · API Fundamentals](level-1/02-api-fundamentals.md)

| Concept | Key fact |
|---------|----------|
| Roles | `user` (human), `assistant` (model's prior turns), system prompt (developer rules) |
| System prompt | Top-level `system=` parameter; sets persona, rules, format |
| Statelessness | The API remembers nothing — resend full history every call |
| `max_tokens` | Hard output cap; check `stop_reason == "max_tokens"` for truncation |
| `stop_reason` | `end_turn` = finished; `max_tokens` = truncated; `tool_use` = wants a tool |
| Temperature | Sampling randomness; removed on newest Claude models — steer via prompts |
| Context window | Max input size; 1M tokens for Claude Sonnet 5 |
| Count before | `client.messages.count_tokens(...)` — free |
| Count after | `response.usage.input_tokens` / `.output_tokens` |

### [03 · Prompt Engineering](level-1/03-prompt-engineering.md)

| Technique | Use when | One-liner |
|-----------|----------|-----------|
| Specific instructions | Always | State task, audience, format, constraints explicitly |
| XML delimiters | Prompt mixes instructions and data | Wrap data in `<tags>`; refer to sections by name |
| Few-shot examples | Format or edge cases matter | 2–5 input→output pairs, including hard cases |
| Role prompting | Tone/judgment matters | "You are a senior X who values Y" |
| Chain of thought | Math, logic, multi-step analysis | "Think in `<thinking>`, answer in `<answer>`" |
| Output constraints | Downstream code parses the reply | "Respond with only the category name" |
| Systematic iteration | Always | Test set + one change at a time + rerun everything |

### [04 · Structured Output](level-1/04-structured-output.md)

| Technique | Guarantee | Cost |
|-----------|-----------|------|
| Schema in prompt | None — usually works | Free |
| "ONLY JSON, no fences" instruction | Fewer format slips | Free |
| `extract_json` tolerant parsing | Survives fences/prose | Free |
| Pydantic `model_validate` | Correct types & fields, or a clear error | Free |
| Retry with error feedback | Self-heals most failures | Extra API calls |
| `client.messages.parse(output_format=Model)` | Schema-valid by construction | Native, Anthropic API |
| `model_json_schema()` in prompt | Prompt and validator never drift | Free |

### [05 · Tool / Function Calling](level-1/05-tool-calling.md)

| Concept | Key fact |
|---------|----------|
| Tool definition | `{name, description, input_schema}` — schema is JSON Schema |
| Descriptions | Say *when* to use the tool; the model chooses based on them |
| Detection | `response.stop_reason == "tool_use"` |
| Request block | `block.type == "tool_use"` with `.id`, `.name`, `.input` (already parsed) |
| Result message | `{"type": "tool_result", "tool_use_id": ..., "content": ...}` in a **user** turn |
| Parallel calls | Multiple `tool_use` blocks in one response → all results in one message |
| Errors | Return as result text with `is_error: True`; never raise |
| Control | `tool_choice`: `auto` / `any` / `tool` / `none` |
| Safety | The model only *requests*; your code decides what runs. Never `eval()` model output |

### [06 · Conversation State &amp; Memory](level-1/06-conversation-state-memory.md)

| Concept | Key fact |
|---------|----------|
| Statelessness | The model sees only what's in this request's `messages` |
| Memory cost | Full history is re-billed as input every turn |
| Sliding window | Keep last N turns; must still start with `user`; don't split tool exchanges |
| Summarization | Replace old turns with a model-written summary; keep recent turns verbatim |
| Cheap summarizer | Use the fast model (`claude-haiku-4-5`) for summaries |
| System-prompt state | Stable facts/rules; survives truncation; keep it stable for caching |
| Message state | The dialogue itself; what gets trimmed/summarized |
| Trigger | Summarize on a token budget via `count_tokens`, not message count |

### [07 · Streaming](level-1/07-streaming.md)

| Concept | Key fact |
|---------|----------|
| Why stream | First token in ~0.5s instead of full-response latency; long outputs require it |
| Easy path | `with client.messages.stream(...) as stream:` |
| Text chunks | `for text in stream.text_stream: print(text, end="", flush=True)` |
| Full message | `stream.get_final_message()` after the loop — usage, stop_reason, content |
| History | Append `final.content` exactly as in non-streaming code |
| Events | `content_block_delta` carries `text_delta`; `message_delta` carries stop_reason/usage |
| Wire format | Server-Sent Events (SSE) — proxyable to browsers |
| Partial output | On cancel/disconnect, repair your history; never leave a dangling turn |
| `flush=True` | Without it, terminal buffering hides the streaming |

### [08 · Errors, Rate Limits &amp; Cost](level-1/08-errors-rate-limits-cost.md)

| Concern | Practice |
|---------|----------|
| Typed errors | Catch `RateLimitError`, `APIStatusError`, `APIConnectionError` — most specific first |
| Retryable | 429, ≥500, connection errors — with exponential backoff + jitter |
| Not retryable | 400/401/403/404 — fix the request, not the retry count |
| `retry-after` | Honor the header on 429s |
| Built-in retries | `anthropic.Anthropic(max_retries=N)` — SDK backs off automatically |
| Timeouts | `timeout=30.0` on the client; stream long responses |
| Rate limit visibility | `x-ratelimit-*` response headers via `with_raw_response` |
| Cost formula | `(in_tokens × in_price + out_tokens × out_price) / 1e6` |
| Cost data | `response.usage` after; `count_tokens` (free) before |
| Cheap tier | Route easy tasks to `claude-haiku-4-5` |
| Caching | App-level response cache + provider prompt caching |

### [09 · A Simple Agent Loop](level-1/09-simple-agent-loop.md)

| Concept | Key fact |
|---------|----------|
| Agent | Model plans → your code executes tools → model observes → repeat |
| Loop exit | `stop_reason != "tool_use"` → final answer |
| Iteration cap | Always bound the loop (`MAX_ITERATIONS`) — runaway loops burn money |
| Dangerous actions | Human confirmation before anything that writes/sends/spends |
| Path safety | Resolve + containment-check every model-supplied filename |
| Errors to the model | Return tool errors as results; the agent adapts |
| Cost guardrail | Track usage per iteration; abort past a budget |
| Overkill test | If steps are known in advance, use a pipeline, not an agent |

### [10 · Capstone — CLI Personal Assistant](level-1/10-capstone-project.md)

| Component | Modules it applies |
|-----------|--------------------|
| `tools.py` — schemas, safe implementations, dispatcher | 05 |
| `memory.py` — token-budgeted summarization | 02, 06 |
| `assistant.py` — streaming agent loop + iteration cap | 05, 07, 09 |
| `CostTracker` — usage → dollars | 08 |
| Retries/timeout via SDK config | 08 |
| `evals.py` — golden prompts with checkable expectations | 03 |
| System prompt with tool rules | 02, 03 |

## Level 2 · Intermediate

### [01 · Advanced Prompting Patterns](level-2/01-advanced-prompting-patterns.md)

| Pattern | Use it for |
|---|---|
| Meta-prompting | Fast first draft of a new system prompt |
| Prompt chaining | Complex tasks with clear intermediate outputs |
| Self-critique | Catching errors before returning output to users |
| Prompt library | Versioning, reviewing, and rolling back prompt changes |

### [02 · Prompt Caching &amp; Cost Optimization](level-2/02-prompt-caching.md)

| Concept | Key fact |
|---|---|
| `cache_control` | Marks the end of a cacheable prefix segment |
| Prefix rule | Everything before the breakpoint must be byte-identical across calls |
| Ordering | Stable content (system, docs) first; volatile content (question) last |
| First call | Pays a small write premium (`cache_creation_input_tokens`) |
| Later calls | Steep discount on `cache_read_input_tokens` |
| Verification | Always log `usage` — never assume the cache is hitting |
| Multi-turn | Cache the growing history by breakpointing the latest message each turn |

### [03 · Vision &amp; Multimodal Inputs](level-2/03-vision-multimodal.md)

| Task | Content block |
|---|---|
| Photo, screenshot, chart | `{"type": "image", "source": {...}}` |
| PDF (text + layout) | `{"type": "document", "source": {...}}` |
| Multiple images | Multiple blocks in one message, in reading order |
| Precision on small text | Ask for transcription first, then the answer |
| Cost | Images/PDFs consume real context tokens — budget for them |

### [04 · Advanced Tool Use Patterns](level-2/04-advanced-tool-use.md)

| Concept | Key fact |
|---|---|
| Parallel execution | Model batches requests; your code decides serial vs. concurrent |
| Strict schema | `enum`/`pattern`/`additionalProperties: False` constrain generation directly |
| Forced tool | `tool_choice: {"type": "tool", "name": ...}` — reliable structured extraction |
| Disable parallel | `disable_parallel_tool_use: True` when ordering matters |
| Server-side tools | Provider executes; you only enable and read results |
| Trust boundary | Validate tool arguments on receipt even with a strict schema |

### [05 · Model Context Protocol (MCP)](level-2/05-mcp.md)

| Concept | Key fact |
|---|---|
| MCP | Protocol standardizing tool discovery + invocation across apps |
| Tools / Resources / Prompts | The three things an MCP server can expose |
| `list_tools()` | Client fetches current schemas at connection time |
| `call_tool()` | Client-side equivalent of Level 1's `execute_tool` |
| Transport | stdio for local subprocess servers, HTTP for remote ones |
| Reuse payoff | One server backs many client applications |

### [06 · Evaluation Frameworks](level-2/06-evaluation-frameworks.md)

| Concept | Key fact |
|---|---|
| Golden dataset | Fixed inputs + expected outputs, versioned like test fixtures |
| Exact-match scoring | Use for classification/extraction with one correct answer |
| LLM-as-judge | Use for open-ended output; grade against an explicit rubric |
| Judge consistency | Pin the judge model; spot-check against human judgment |
| Regression suite | Threshold-gated script run in CI on every prompt/model change |
| Eval-driven dev | Write the failing eval case first when you find a bug |

### [07 · Advanced Agent Patterns](level-2/07-advanced-agent-patterns.md)

| Pattern | Purpose |
|---|---|
| Planner/executor split | Auditable, checkpointable multi-step plans |
| Reflection | Catch drift/dead-ends before they compound |
| Task list as external state | Compact, always-current progress view |
| Context editing | Stay within the context window on long-running agents |

### [08 · Multi-Agent Systems](level-2/08-multi-agent-systems.md)

| Concept | Key fact |
|---|---|
| Orchestrator | Delegates via tool calls; never does the work itself |
| Subagent | A `messages.create` call with a narrow system prompt, invoked as a tool |
| Specialization payoff | Narrower prompt + isolated context = higher reliability |
| Shared workspace | Files/DB rows for artifacts too large to pass through messages |
| Inter-agent messaging | An explicit message log both sides read/append to |
| No hidden coordination | Agents only share what your code explicitly routes between them |

### [09 · Building LLM Web Apps](level-2/09-llm-web-apps.md)

| Concern | Approach |
|---|---|
| Basic endpoint | FastAPI route wrapping `messages.create` |
| Perceived latency | `StreamingResponse` + SSE, token-by-token |
| Session state | External store (Redis/DB) keyed by session id, with TTL |
| Concurrency limits | `asyncio.Semaphore` capping in-flight model calls |
| Provider errors | Catch and translate to clean HTTP status + retry guidance |

### [10 · Project — Production-Ready Agent Service](level-2/10-project-agent-service.md)

| Layer | Module it came from |
|---|---|
| Cached system prompt | Module 2 |
| Strict tool schemas | Module 4 |
| Agent loop + iteration cap | Level 1, module 9 |
| SSE streaming | Module 9 |
| Redis session persistence | Module 9 |
| CI-gated eval suite | Module 6 |

## Level 3 · Advanced

### [01 · Transformer Internals](level-3/01-transformer-internals.md)

| Component | Role |
|---|---|
| Embedding table | Maps token ids to learned vectors |
| Positional encoding / RoPE | Injects order into an otherwise order-agnostic mechanism |
| Self-attention | Weighted average of all (visible) tokens' values |
| Causal mask | Blocks attending to future tokens — enables next-token prediction |
| Multi-head | Several parallel attention computations, concatenated |
| Feed-forward + residual + norm | Nonlinear per-position processing, stable deep stacking |
| KV cache | Reuses prior tokens' keys/values instead of recomputing every step |

### [02 · Tokenizers Deep Dive](level-3/02-tokenizers.md)

| Concept | Key fact |
|---|---|
| BPE | Iteratively merges the most frequent adjacent pair into a new token |
| Vocabulary | The ordered list of learned merges, fixed after training |
| Unseen words | Decompose into smaller known pieces — never a hard failure |
| Token ≠ word ≠ character | None of these counts are interchangeable |
| Cross-model variance | Different training corpora → different vocabularies → different counts |
| Cost estimation | Always use the real tokenizer/count API, never a word-count heuristic |

### [03 · Embeddings &amp; Semantic Search](level-3/03-embeddings-semantic-search.md)

| Concept | Key fact |
|---|---|
| Embedding | Fixed-length vector; semantically similar text → nearby vectors |
| `input_type` | Query vs. document embeddings are often different — don't mix them up |
| Cosine similarity | Measures angle, not magnitude — matches the training objective |
| Vector store | Approximate nearest-neighbor index for scale beyond brute force |
| Chunking | Balance too-small (loses context) vs. too-large (dilutes meaning) |
| RAG | Retrieval + generation, for synthesized rather than raw-document answers |

### [04 · Running Local Models with Ollama](level-3/04-local-models-ollama.md)

| Concept | Key fact |
|---|---|
| `ollama pull` | Downloads a model's weights, quantized by tag |
| `/api/generate` | Single-prompt completion |
| `/api/chat` | Multi-turn, same `messages` shape as the hosted API |
| Streaming | `"stream": true` → newline-delimited JSON chunks |
| Tool calling | Supported by newer models; validate reliability per model |
| OpenAI-compat endpoint | `base_url="http://localhost:11434/v1"` |
| Hardware rule of thumb | ~parameter count in GB of RAM at 4-bit quantization |

### [05 · Serving Open Models with vLLM](level-3/05-serving-vllm.md)

| Concept | Key fact |
|---|---|
| vLLM | OpenAI-compatible inference server for open-weight models |
| PagedAttention | Block-based, on-demand KV cache allocation — less wasted GPU memory |
| Continuous batching | Requests join/leave the running batch per-step, not per-batch |
| `--gpu-memory-utilization` | KV cache pool size — biggest lever on concurrent capacity |
| `--enable-prefix-caching` | Self-hosted equivalent of hosted prompt caching |
| Load testing | Always measure real throughput/latency before tuning further |

### [06 · Fine-Tuning Fundamentals (LoRA/QLoRA)](level-3/06-fine-tuning.md)

| Concept | Key fact |
|---|---|
| Fine-tune vs. prompt | Prompting first; fine-tune for consistent style/format at volume |
| Not for facts | Use retrieval (module 3) for facts; fine-tuning bakes in behavior, not knowledge |
| Dataset | Consistent `messages`-shaped examples; quality over volume |
| LoRA | Freezes base weights; trains small low-rank adapter matrices |
| `r` (rank) | Key size/quality knob — too small under-fits, too large loses the LoRA benefit |
| QLoRA | LoRA on a 4-bit quantized frozen base — lowest memory footprint |
| Always re-eval | Compare fine-tuned vs. base+prompting on a held-out set, not just training loss |

### [07 · Quantization &amp; Inference Optimization](level-3/07-quantization-inference.md)

| Concept | Key fact |
|---|---|
| Quantization | Fewer bits per weight → smaller model, faster memory reads |
| GGUF | llama.cpp/Ollama format; per-block scaling; CPU/consumer-GPU friendly |
| GPTQ | Calibrated quantization minimizing output error, layer by layer |
| AWQ | Preserves high-precision for activation-correlated "important" weights |
| Bottleneck | Single-request generation is memory-bandwidth-bound, not compute-bound |
| Speculative decoding | Draft model proposes, target model verifies in one batched pass |
| Always measure | Quality loss is task-dependent — eval, don't assume |

### [08 · Guardrails &amp; Content Moderation](level-3/08-guardrails-moderation.md)

| Layer | Purpose |
|---|---|
| Input filtering | Block or flag before the request reaches the main model |
| Output filtering | Catch problematic completions the input check couldn't predict |
| Dedicated moderation API | Cheaper, purpose-trained for common abuse categories |
| Custom LLM classifier | Domain-specific policy rules a generic API doesn't know |
| Policy table | Rules as data — auditable, versionable, separate from app code |
| Audit logging | Record every enforcement decision, independent of the chat transcript |

### [09 · Observability &amp; Tracing](level-3/09-observability-tracing.md)

| Concept | Key fact |
|---|---|
| Trace | All spans (LLM calls, tool executions) for one user request, tied by trace ID |
| Span | One unit of work with timing, metadata, and error status |
| Token/cost dashboard | Aggregated `usage` data over time, alertable on spend spikes |
| Debugging via trace | Answer "what did it see / decide / get back" without live reproduction |
| Observability platforms | Same data model as hand-rolled tracing, with less boilerplate |

### [10 · Project — Self-Hosted LLM Stack](level-3/10-project-self-hosted-stack.md)

| Layer | Module it came from |
|---|---|
| Open model, quantized | Modules 4, 7 |
| vLLM serving | Module 5 |
| Input/output guardrails | Module 8 |
| Tracing + usage logging | Module 9 |
| Gateway orchestration | Plain FastAPI, no new mechanism |

## Level 4 · Master

### [01 · Pretraining &amp; Scaling Laws Overview](level-4/01-pretraining-scaling-laws.md)

| Concept | Key fact |
|---|---|
| Pretraining objective | Predict the next token — no task-specific labels |
| Data pipeline | Dedup → quality filter → PII/safety filter → deliberate source mixing |
| Scaling laws | Empirical power-law relationships between loss, size, data, compute |
| Compute-optimal | For fixed compute, there's a size/data split that minimizes loss |
| Diminishing returns | Power-law scaling means each doubling helps less than the last |
| Practical takeaway | Data quality/mix now rivals raw scale as the main quality lever |

### [02 · Alignment &amp; RLHF Overview](level-4/02-alignment-rlhf.md)

| Stage | Purpose |
|---|---|
| SFT | Teach assistant format/behavior from curated examples |
| Reward model | Predict human preference ranking as a scalar score |
| PPO-RLHF | RL against the reward model, KL-penalized to a reference policy |
| DPO | Same objective as RLHF, solved as a direct supervised loss — no RL loop |
| Constitutional AI / RLAIF | Model-generated preference labels guided by written principles |
| KL penalty | Prevents reward hacking and capability collapse during RL |
| Why over-refusal happens | Alignment generalizes patterns across training distribution, not per-case rules |

### [03 · Production LLM Platform Architecture](level-4/03-llm-platform-architecture.md)

| Component | Job |
|---|---|
| Provider adapter | Translates internal request/response shape to one provider's API |
| Routing table | Maps task type to an ordered list of providers/models to try |
| Fallback chain | Tries the next provider on failure instead of failing the whole request |
| Token bucket | Per-team/key rate limiting with smooth, continuous refill |
| Normalized errors | One retryable/non-retryable contract across all providers |
| Centralized logging | One place to see cross-team cost, latency, and error rate |

### [04 · Safety &amp; Red-Teaming](level-4/04-safety-red-teaming.md)

| Concept | Key fact |
|---|---|
| Jailbreak categories | Role-play, instruction override, obfuscation, context stuffing, escalation |
| Control cases | Benign prompts that must NOT be refused — measures over-refusal |
| Automated red-teaming | Use an attacker LLM to generate/mutate adversarial variants at scale |
| LLM-as-judge for refusal | Classify compliance vs. refusal on the target's actual response |
| Layered defense | Alignment + system prompt + input filter + output filter + monitoring |
| Why layers compound | Independent failure modes multiply, not just add, reducing overall risk |

### [05 · Advanced Evaluation at Scale](level-4/05-evaluation-at-scale.md)

| Concept | Key fact |
|---|---|
| Continuous eval | Sample live traffic continuously, not just a static golden set |
| LLM-as-judge at scale | Automates scoring of sampled production traffic |
| A/B testing | Deterministic hash-based assignment; measure with a significance test |
| Effect size | Whether a measured difference is meaningful, not just non-zero |
| Rolling baseline | Alert on relative drift from a moving baseline, not a fixed threshold |
| Why sampling matters | Live traffic covers failure modes no static golden set anticipated |

### [06 · Cost Optimization at Scale](level-4/06-cost-optimization-at-scale.md)

| Lever | Mechanism | Best for |
|---|---|---|
| Model tiering | Route to cheapest sufficient model, escalate on low confidence | Mixed-difficulty traffic |
| Batch API | Trade latency (hours) for a substantial per-token discount | Non-interactive, bulk workloads |
| Exact-match caching | Skip inference entirely on identical repeated requests | FAQ-style, repeated queries |
| Prompt caching | Reuse processed prefix across calls with shared context | Long, stable system prompts/docs |
| Cost attribution | Tag every request with team/caller at the gateway | Accountability, budget alerts |

### [07 · Latency Engineering &amp; Batching](level-4/07-latency-engineering.md)

| Concept | Key fact |
|---|---|
| p99 over mean | Tail latency reflects real worst-case user experience; mean hides it |
| Time-to-first-token | Matters more than total time for perceived responsiveness |
| Streaming-first | Default every user-facing feature to stream, not wait-then-return |
| Request batching | Trade a small wait for much better throughput/GPU utilization |
| Speculative decoding | Draft model guesses tokens; target model verifies in one pass |
| Latency budget routing | Fall back to a faster model when the primary is trending slow |

### [08 · Security — Prompt Injection &amp; Beyond](level-4/08-security-prompt-injection.md)

| Concept | Key fact |
|---|---|
| Direct injection | Attacker is the user, writing malicious text into their own prompt |
| Indirect injection | Attacker plants instructions in third-party data the model processes |
| Delimiter defense | Mark untrusted content structurally; reduces but doesn't eliminate risk |
| Least-privilege tools | Scope tool access tightly per task — the strongest structural defense |
| Human-in-the-loop | Require explicit confirmation for irreversible/sensitive tool calls |
| Exfiltration risk | Any outbound network call with attacker-influenced content/URL |

### [09 · Compliance, Privacy &amp; Governance](level-4/09-compliance-privacy-governance.md)

| Concept | Key fact |
|---|---|
| Retention policy | Define per data category, not one global rule |
| PII redaction | Apply before data leaves your boundary; regex + model for full coverage |
| Audit trail | Hash-chained, tamper-evident log of every consequential action |
| Purpose limitation | Don't silently repurpose data collected for one use into another |
| Data minimization | Send models only what the task needs, not everything available |
| Deletion by design | Every store needs a `user_id`-indexed deletion path from day one |

---

Go deeper with the full [LLM Development Mastery Path](index.md) curriculum or test yourself with the [interview questions](interview-questions.md).
