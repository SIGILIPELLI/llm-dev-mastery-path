---
title: "LLM Development Interview Questions and Answers (179 Questions)"
description: "179 LLM Development interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the LLM Development Mastery Path."
---

# LLM Development Interview Questions and Answers

**179 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free LLM Development Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · The LLM Landscape &amp; Setup](level-1/01-llm-landscape-setup.md)

**Q1.** What does an LLM do at its core?

**Answer:** Predicts the next token in a sequence

At scale this simple objective yields summarizing, translating, coding and instruction following.

**Q2.** When is a local open-weight model the better fit than a hosted API?

**Answer:** Privacy-critical or offline use, or high fixed volume

Hosted APIs suit most products; local models run on your own hardware.

**Q3.** Why does the course use the Anthropic API for examples?

**Answer:** The concepts (messages, system prompts, tools, streaming) are identical across providers

Everything transfers between providers.

**Q4.** What is rule one of LLM development regarding the API key?

**Answer:** Never hardcode it; use environment variables

Anyone with the key can spend your credit.

### [02 · API Fundamentals](level-1/02-api-fundamentals.md)

**Q5.** Why must you resend the whole conversation in each call?

**Answer:** The API is stateless: the model only 'remembers' what is in the request

Alternate user/assistant turns, always ending on a user turn.

**Q6.** Where does the system prompt go in the Anthropic API?

**Answer:** A top-level system parameter, not a message in the list

In OpenAI's API it's a message with role system; the concept is the same.

**Q7.** What is max_tokens?

**Answer:** A hard cap on output; generation stops at the limit even mid-sentence

It is not a request for a length.

**Q8.** What does the system prompt control?

**Answer:** Persona, rules, format and boundaries, treated as more authoritative than user text

It is the highest-leverage line in most LLM apps.

### [03 · Prompt Engineering](level-1/03-prompt-engineering.md)

**Q9.** What is a good mental test for a prompt?

**Answer:** Would a smart new intern produce the right output from it alone, with no chance to ask questions?

State the task, audience, format and constraints.

**Q10.** Why wrap data in XML-style tags?

**Answer:** The model can't confuse data with instructions, and you can refer to sections by name

It also helps defang injection-like inputs.

**Q11.** Why use few-shot examples?

**Answer:** Showing input→output pairs pins down format and edge cases better than prose

Showing beats telling.

**Q12.** Which prompt is stronger?

**Answer:** 'Summarize for busy software engineers in exactly 3 bullets under 20 words, no marketing language'

Vague prompts get vague answers.

### [04 · Structured Output](level-1/04-structured-output.md)

**Q13.** Why isn't 'ask for JSON and json.loads it' enough in production?

**Answer:** It works most of the time, but you may get fences, commentary or wrong types

Production code needs guarantees.

**Q14.** What does tolerant extraction do?

**Answer:** Strips fences and takes the outermost JSON object before parsing

It's a cheap robustness layer.

**Q15.** What does pydantic add beyond parsing?

**Answer:** Validation: types, requiredness and constraints against your schema

Parsing proves it's JSON; validation proves it's your JSON.

**Q16.** What does the toolkit in this module include?

**Answer:** Schema-in-prompt, pydantic validation, retry-on-invalid and native structured-output mode

Each step adds more reliability.

### [05 · Tool / Function Calling](level-1/05-tool-calling.md)

**Q17.** What is the crucial mental model of tool calling?

**Answer:** The model never runs code; it only emits a structured request and your program executes it

Your program stays in control of what executes.

**Q18.** Why does a tool description matter so much?

**Answer:** The model reads it to decide when to use the tool, so say when to call it, not just what it does

enum pins down fixed choices in parameters.

**Q19.** Why does the lesson implement safe_eval instead of calling eval()?

**Answer:** eval() on model-produced text would execute arbitrary code

Parse arithmetic with an allowed set of operators.

**Q20.** What is the tool loop?

**Answer:** Request → execute → respond: describe tools, model asks, your code runs it, model uses the result

It is the foundation for agents.

### [06 · Conversation State &amp; Memory](level-1/06-conversation-state-memory.md)

**Q21.** Why do naive conversation costs balloon?

**Answer:** You re-pay for the whole history on every turn; turn 1 is resent in a 50-turn chat 50 times

input_tokens climbs every turn.

**Q22.** What is sliding-window truncation?

**Answer:** Keeping only the last N turns

It's simple and predictable.

**Q23.** What are the two gotchas when truncating?

**Answer:** History must still start with a user message, and never cut between a tool_use and its tool_result

Trim at whole-exchange boundaries.

**Q24.** What does appending the assistant reply to messages represent?

**Answer:** The memory itself

You are the memory manager because the API is stateless.

### [07 · Streaming](level-1/07-streaming.md)

**Q25.** Why stream?

**Answer:** Perceived latency: first words in a few hundred milliseconds instead of waiting 10–30 seconds for the full answer

It also avoids HTTP timeouts on very large outputs.

**Q26.** Why print with flush=True when streaming?

**Answer:** Without it Python buffers output and the stream appears in bursts

flush defeats buffering.

**Q27.** What does get_final_message() return?

**Answer:** The complete message with usage and stop_reason after the stream ends

Stream for the user, collect the final message for your program.

**Q28.** What does text_stream give you?

**Answer:** A convenience over the typed event sequence, yielding plain text chunks

Iterate events when you need thinking indicators or tool-call progress.

### [08 · Errors, Rate Limits &amp; Cost](level-1/08-errors-rate-limits-cost.md)

**Q29.** Which errors should you retry with backoff?

**Answer:** RateLimitError (429), server errors (≥500) and connection errors

A 400 will fail forever, so fix the code.

**Q30.** Why add random jitter to retries?

**Answer:** So a fleet of clients doesn't retry in lockstep

Also honor the server's retry-after header when present.

**Q31.** What should you catch instead of string-matching error messages?

**Answer:** Typed SDK exceptions

For example RateLimitError and APIStatusError.

**Q32.** What does a 401 AuthenticationError mean?

**Answer:** Bad or missing API key: don't retry, fix the key

PermissionDeniedError (403) means the key lacks access.

### [09 · A Simple Agent Loop](level-1/09-simple-agent-loop.md)

**Q33.** What is the agent loop?

**Answer:** Plan → act (call tools) → observe results → repeat until done

The model drives control flow; your code supplies tools, limits and the transcript.

**Q34.** What guardrail keeps loops from running away?

**Answer:** A maximum number of iterations

The capstone uses MAX_TOOL_ITERATIONS = 8 per user turn.

**Q35.** Why does safe_path confine file access to the workspace?

**Answer:** Never trust model-supplied paths

A model could otherwise request files outside the intended directory.

**Q36.** When is an agent the wrong tool?

**Answer:** When a simple fixed workflow or a single call would do the job more predictably

The module discusses this explicitly.

### [10 · Capstone — CLI Personal Assistant](level-1/10-capstone-project.md)

**Q37.** What does the CLI assistant combine?

**Answer:** Tools, streaming, conversation memory with summarization, cost tracking and a golden-prompt eval suite

It's a production-style architecture at miniature scale.

**Q38.** Why does config.py use a cheaper SUMMARIZER_MODEL?

**Answer:** Summarizing history is a simple task, so a cheaper model saves cost

The main model is claude-sonnet-5; the summarizer is claude-haiku-4-5.

**Q39.** What does the system prompt tell the assistant about arithmetic?

**Answer:** Use the calculator tool for all arithmetic, never compute in its head

It also says to use save_note for durable facts.

**Q40.** What are golden prompts in evals.py?

**Answer:** Fixed test prompts with expected behavior, run to catch regressions

They form a small evaluation suite.

## Level 2 · Intermediate

### [01 · Advanced Prompting Patterns](level-2/01-advanced-prompting-patterns.md)

**Q41.** What is meta-prompting?

**Answer:** Asking a strong model to draft a prompt, then refining it against real examples

Treat the output as a first draft and validate it against a labeled eval set.

**Q42.** What is prompt chaining?

**Answer:** Breaking a complex task into a pipeline of focused calls with narrow jobs and clean outputs

For example summarize, then extract actions, then draft a reply.

**Q43.** Why validate a meta-generated prompt before shipping?

**Answer:** It is only a reasonable starting point, not gospel

Use a labeled eval set.

**Q44.** What do the advanced patterns help manage?

**Answer:** Prompt libraries that grow into the hundreds, with self-critique loops and versioning

Versioned prompt libraries keep them maintainable.

### [02 · Prompt Caching &amp; Cost Optimization](level-2/02-prompt-caching.md)

**Q45.** What does prompt caching let the server do?

**Answer:** Reuse an already-processed prefix instead of recomputing it, cutting cost and latency

It helps when a large prefix repeats across calls.

**Q46.** How do you mark a cacheable prefix?

**Answer:** With cache_control on a block; everything up to and including it is a candidate

It's an ephemeral breakpoint.

**Q47.** What is the most important caching rule?

**Answer:** Tokens before the breakpoint must be byte-for-byte identical across calls; per-request changes go after it

A date or random request ID before the breakpoint breaks the cache.

**Q48.** What do the first and second calls show in usage?

**Answer:** First writes the cache (high cache_creation tokens); the second reads it (high cache_read, tiny input_tokens)

Reads are charged at a steep discount.

### [03 · Vision &amp; Multimodal Inputs](level-2/03-vision-multimodal.md)

**Q49.** How are images sent to the model?

**Answer:** As base64 (or URL) image content blocks alongside text in the messages structure

This avoids a separate OCR pipeline for many tasks.

**Q50.** Where should the image block go relative to the text instruction?

**Answer:** Before the text, since the model reads blocks in order

Grounding the instruction after the image reduces ambiguity.

**Q51.** How should you handle comparing two images?

**Answer:** Label them explicitly in text, e.g. 'Image 1 (before):', since the model sees blocks in sequence

There are no named variables.

**Q52.** How are PDFs sent?

**Answer:** As a document content block, with the model reading both extracted text and page layout

Useful for tables, forms and scanned pages.

### [04 · Advanced Tool Use Patterns](level-2/04-advanced-tool-use.md)

**Q53.** When should tools run in parallel?

**Answer:** Only when they are independent and safe out-of-order, such as read-only lookups

Conflicting side effects should stay serial.

**Q54.** Why tighten the tool's JSON Schema?

**Answer:** A loose schema lets the model emit valid-but-useless arguments; enums and patterns enforce the constraint

Don't rely on the model to infer constraints from descriptions.

**Q55.** What does a pattern like ^\d{4}-\d{2}-\d{2}$ enforce?

**Answer:** An ISO date format

It's the due_date constraint in the create_ticket example.

**Q56.** What is a server-side tool?

**Answer:** One the model can invoke without a round trip through your code

The module covers them alongside parallelism and control over when tools fire.

### [05 · Model Context Protocol (MCP)](level-2/05-mcp.md)

**Q57.** What problem does MCP solve?

**Answer:** It standardizes how apps discover and call tools exposed by independent servers, so a tool is written once and reused

Hand-written TOOLS lists don't scale across apps.

**Q58.** What are the three kinds of things an MCP server exposes?

**Answer:** Tools, resources and prompts

Resources are readable data addressed by URI; prompts are parameterized templates.

**Q59.** How does the Python SDK's @mcp.tool() build the schema?

**Answer:** It inspects the function's type hints and docstring

You write a normal Python function.

**Q60.** How does your app act with MCP?

**Answer:** As a client that launches or connects to the server and forwards its tools into messages.create

Servers use stdio for local processes or HTTP for remote ones.

### [06 · Evaluation Frameworks](level-2/06-evaluation-frameworks.md)

**Q61.** What is a golden dataset?

**Answer:** A fixed set of representative inputs with known-good outputs, kept in version control

Include real examples and deliberate edge cases.

**Q62.** Why must a golden set include hard cases?

**Answer:** A set of only easy cases passes forever and tells you nothing

Use ambiguous inputs, adversarial phrasing and known past failures.

**Q63.** When do you score with plain code instead of an LLM?

**Answer:** Tasks with a single correct answer, such as classification or extraction

Exact match is meaningless for open-ended output.

**Q64.** What is LLM-as-judge?

**Answer:** A second model call grades output against a rubric, returning a structured verdict you can aggregate

It's used for open-ended tasks.

### [07 · Advanced Agent Patterns](level-2/07-advanced-agent-patterns.md)

**Q65.** Why split a planner from an executor?

**Answer:** A forced-schema planning call yields a plan you can log, edit or approve before any tool runs, making the agent auditable

The executor only consults the planner again if the plan needs revising.

**Q66.** What problem do long-horizon agents hit with a single undifferentiated loop?

**Answer:** They drift, lose the original goal or overflow their context window

The patterns here keep them on track.

**Q67.** What does tool_choice forcing set_plan guarantee?

**Answer:** The model returns a structured plan via that tool

Structured output through a forced tool call.

**Q68.** Which patterns does the module cover?

**Answer:** Planner/executor splits, reflection, task management and context editing

All aim at long-running tasks.

### [08 · Multi-Agent Systems](level-2/08-multi-agent-systems.md)

**Q69.** What does the orchestrator do?

**Answer:** Decides which subagent handles each piece, delegates via tool calls and assembles the results, without doing the work itself

Subagents are specialized with different roles.

**Q70.** Why give subagents distinct abilities and descriptions?

**Answer:** The orchestrator reads the tool descriptions to choose the right specialist

For example the researcher can't write code and the coder can't browse.

**Q71.** What is a researcher subagent's system prompt told to do?

**Answer:** Use only verifiable facts and say clearly when uncertain

Focused instructions per role.

**Q72.** When does the orchestrator loop end?

**Answer:** When stop_reason is no longer tool_use and the final text is returned

Otherwise it keeps delegating.

### [09 · Building LLM Web Apps](level-2/09-llm-web-apps.md)

**Q73.** What is the web framework's job in an LLM web app?

**Answer:** Routing and (de)serialization; the LLM logic underneath is unchanged

The /chat endpoint mirrors the terminal conversation loop.

**Q74.** Why use Server-Sent Events for streaming?

**Answer:** Users see tokens as they're generated instead of staring at a spinner for the full response

The same mechanism as terminal streaming, piped to an HTTP response.

**Q75.** Why store history by session_id?

**Answer:** A user's conversation persists across requests instead of resetting each time

It's backed by a database.

**Q76.** How do you run the FastAPI app in development?

**Answer:** uvicorn app:app --reload

It's then callable from any HTTP client.

### [10 · Project — Production-Ready Agent Service](level-2/10-project-agent-service.md)

**Q77.** Which pieces does the agent service combine?

**Answer:** FastAPI with SSE, an agent loop with tools and a planner, Redis sessions, a cached system prompt and a CI-gated eval suite

It's the shape of a real production agent.

**Q78.** Why cache the system prompt and docs?

**Answer:** They are large and stable, so caching cuts cost and latency

Prefix stability is the rule that matters.

**Q79.** Why do the tool schemas set additionalProperties: false and minLength?

**Answer:** Strict schemas reduce malformed calls

Module 4's strict schemas.

**Q80.** What does the eval.yml workflow do?

**Answer:** Gates changes with a regression suite run in CI

Golden prompts catch regressions before shipping.

## Level 3 · Advanced

### [01 · Transformer Internals](level-3/01-transformer-internals.md)

**Q81.** What is a token embedding?

**Answer:** A learned vector looked up from an embedding table, not derived from the token's spelling

It's a lookup, shaped by training.

**Q82.** Why does a transformer need positional information?

**Answer:** Attention has no inherent notion of order, so 'the cat sat' and 'sat cat the' would look identical

A position-dependent vector is added to each embedding.

**Q83.** How do rotary position embeddings (RoPE) differ from sinusoidal ones?

**Answer:** They rotate query/key vectors by a position-dependent angle, generalizing better to unseen lengths

Modern production models more commonly use RoPE.

**Q84.** What does self-attention compute for each token?

**Answer:** A weighted average of all tokens' values, with weights from query–key matches

Weights come from how well each query matches every key.

### [02 · Tokenizers Deep Dive](level-3/02-tokenizers.md)

**Q85.** Why not tokenize by characters or whole words?

**Answer:** Characters make sequences very long; words make the vocabulary explode and can't represent unseen words

BPE sits between them.

**Q86.** How does BPE build its vocabulary?

**Answer:** Start from characters/bytes and repeatedly merge the most frequent adjacent pair

It builds up substrings that actually recur in the corpus.

**Q87.** Why can BPE represent any string?

**Answer:** Worst case it falls back to individual bytes

Gibberish and new words are still representable.

**Q88.** Why do token counts differ between models and strings?

**Answer:** Different tokenizers merge different pieces, so strings you expect to cost the same can differ

Token counts drive cost and context use.

### [03 · Embeddings &amp; Semantic Search](level-3/03-embeddings-semantic-search.md)

**Q89.** What problem do embeddings solve over keyword search?

**Answer:** Query and document describing the same thing in different words ('car' vs 'automobile')

Similar text lands near each other in vector space.

**Q90.** Why use cosine similarity rather than raw distance?

**Answer:** It measures the angle, dividing out magnitude, so same-direction vectors are similar regardless of length

cos = a·b / (‖a‖‖b‖).

**Q91.** What does input_type (document vs query) do?

**Answer:** Models produce slightly different vectors optimized for each role; mixing them degrades search quality without errors

Both calls succeed, which hides the mistake.

**Q92.** A query 'How do I get my money back?' matches a doc about refunds. What does that show?

**Answer:** Semantic matching works with almost no shared words

That's the value proposition of embeddings.

### [04 · Running Local Models with Ollama](level-3/04-local-models-ollama.md)

**Q93.** When do you run a model locally with Ollama?

**Answer:** Privacy-sensitive data that can't leave your network, offline development, or avoiding per-token cost during heavy experimentation

Hosted APIs remain the faster path to frontier quality.

**Q94.** What does the tag in llama3.1:8b select?

**Answer:** The parameter count/quantization variant; larger tags need more RAM and are slower but generally more capable

ollama list shows local models.

**Q95.** Where does Ollama serve its HTTP API?

**Answer:** localhost:11434, with native and OpenAI-compatible endpoints

Switching from a hosted API is mostly a base-URL change.

**Q96.** What does "stream": True return?

**Answer:** Newline-delimited JSON chunks

It mirrors the hosted streaming module.

### [05 · Serving Open Models with vLLM](level-3/05-serving-vllm.md)

**Q97.** When do you choose vLLM over Ollama?

**Answer:** Serving an open-weight model to many concurrent users with production-grade throughput

vLLM uses PagedAttention and continuous batching.

**Q98.** Why can client code usually switch to vLLM with only a base_url change?

**Answer:** It exposes an OpenAI-compatible HTTP server

The request/response shape matches the hosted OpenAI API.

**Q99.** What does PagedAttention do?

**Answer:** Splits the KV cache into fixed-size blocks allocated on demand, so memory is used only as tokens are generated and can be shared or freed per block

It borrows OS virtual-memory paging.

**Q100.** What was wasteful about naive KV cache allocation?

**Answer:** Contiguous memory sized for each request's maximum possible length, unusable by others

Most requests finish well short of the max.

### [06 · Fine-Tuning Fundamentals (LoRA/QLoRA)](level-3/06-fine-tuning.md)

**Q101.** When should you try fine-tuning?

**Answer:** After prompting (and more few-shot examples) fails to pin down a narrow, high-volume style or format, or to shrink prompt cost/latency

Prompting is cheaper and faster to iterate.

**Q102.** Why not use fine-tuning to inject facts?

**Answer:** Weights encode patterns, not a queryable knowledge base, and facts go stale once baked in; use retrieval instead

Reserve fine-tuning for behavior and style.

**Q103.** What does fine-tuning data look like?

**Answer:** (input, ideal-output) pairs in the messages shape; quality and consistency matter far more than volume

Aim for at least a few hundred consistent examples.

**Q104.** What does fine-tuning change?

**Answer:** The model's weights

LoRA/QLoRA do this with far fewer trainable parameters.

### [07 · Quantization &amp; Inference Optimization](level-3/07-quantization-inference.md)

**Q105.** What does quantization do?

**Answer:** Stores each weight with fewer bits (8, 4 or fewer), trading precision for a smaller file and faster memory-bound computation

Weights are normally 16- or 32-bit floats.

**Q106.** How does symmetric int8 quantization compute its scale?

**Answer:** scale = max(|weights|) / 127

Quantized = round(w / scale).

**Q107.** What is GGUF?

**Answer:** The format used by llama.cpp and Ollama, with per-block scaling and several bit-widths, optimized for CPU and consumer GPUs

GGUF, AWQ and GPTQ are schemes/formats, not different models.

**Q108.** What does AWQ observe?

**Answer:** A small fraction of weights matters disproportionately, so it keeps those at higher precision

GPTQ instead calibrates error on a sample dataset layer by layer.

### [08 · Guardrails &amp; Content Moderation](level-3/08-guardrails-moderation.md)

**Q109.** Why filter input before it reaches the main model?

**Answer:** A filter call is cheaper than a full generation you'd discard, and bad input never enters context

Check the user input first.

**Q110.** Why also filter the model's output?

**Answer:** A model can produce a problematic response from benign input, or a jailbreak may partially succeed

Moderate what the model produces.

**Q111.** Why use a forced-tool-schema classification call rather than asking free text 'is this okay?'

**Answer:** The policy decision stays structured, loggable and separable from the conversational response

The schema has violates_policy and categories.

**Q112.** What does the policy-enforcement layer add?

**Answer:** Auditability independent of either model's behavior

Defenses go beyond trusting the model's judgment.

### [09 · Observability &amp; Tracing](level-3/09-observability-tracing.md)

**Q113.** What is a trace?

**Answer:** A record tying every LLM call, tool execution and retrieval step in one request under a shared trace ID, with timing and metadata per span

Without tracing, debugging means reading scattered print statements.

**Q114.** What does each span in the Tracer record?

**Answer:** Name, metadata, duration in ms and any error

It's built with a context manager.

**Q115.** What should you wrap in spans?

**Answer:** Every model call and tool execution

Add token usage and result previews as metadata.

**Q116.** Why does a single user request need tracing in modern systems?

**Answer:** Chained calls, subagents, tools and retrieval can trigger a dozen model calls across several services

It lets you debug why the agent did something.

### [10 · Project — Self-Hosted LLM Stack](level-3/10-project-self-hosted-stack.md)

**Q117.** Which components make up the self-hosted stack?

**Answer:** A FastAPI gateway, input/output guardrails, a vLLM server with a quantized open model, and tracing/usage logs

It's a complete alternative to calling a hosted API.

**Q118.** What do --quantization awq and --enable-prefix-caching provide?

**Answer:** A good throughput/quality balance on modest hardware

Quantization shrinks the model; prefix caching reuses shared prefixes.

**Q119.** What does the gateway do for every request?

**Answer:** Runs the input guardrail, calls the model, runs the output guardrail and logs a trace, behind one clean API

Blocked inputs return a refusal with a trace_id.

**Q120.** What does the policy check ask the model to reply?

**Answer:** Exactly one word, BLOCK or ALLOW

The gateway then checks for BLOCK in the reply.

## Level 4 · Master

### [01 · Pretraining &amp; Scaling Laws Overview](level-4/01-pretraining-scaling-laws.md)

**Q121.** What is the pretraining objective?

**Answer:** Predict the next token given all previous tokens over a huge corpus, with no labels or task-specific supervision

The 'label' is just the token that actually came next.

**Q122.** What is the per-position loss?

**Answer:** Cross-entropy, −log P(correct token)

It's averaged over all positions in a batch.

**Q123.** Why does the pipeline deduplicate documents?

**Answer:** Near-identical documents waste compute and can cause memorization of repeated content

It's followed by quality and safety/PII filtering.

**Q124.** Where do reasoning and instruction following come from, per the lesson?

**Answer:** Downstream effects of next-token prediction at scale, refined by the alignment stage

The objective never mentions them.

### [02 · Alignment &amp; RLHF Overview](level-4/02-alignment-rlhf.md)

**Q125.** What is a base model fresh out of pretraining?

**Answer:** A raw next-token predictor with no notion of helpfulness

Alignment turns it into an assistant.

**Q126.** What are the three post-pretraining stages?

**Answer:** Supervised fine-tuning (SFT), reward modeling, and RLHF/policy optimization

SFT teaches the format; the reward model proxies human preference; RL optimizes against it.

**Q127.** Why is RLHF penalized for straying from the SFT model?

**Answer:** To stop reward-hacking gibberish that scores well but reads badly

It keeps outputs fluent.

**Q128.** What does the reward model loss -logsigmoid(r_chosen - r_rejected) train?

**Answer:** The model to score human-preferred responses higher (Bradley–Terry pairwise ranking)

It predicts human rankings.

### [03 · Production LLM Platform Architecture](level-4/03-llm-platform-architecture.md)

**Q129.** Why does a production platform need an LLM gateway?

**Answer:** Otherwise every team duplicates provider clients, retries, fallbacks and cost tracking inconsistently, with no central policy or spend view

It centralizes abstraction, routing/fallback, quotas and observability.

**Q130.** What does a ProviderAdapter do?

**Answer:** Translates the internal request shape to one provider's exact API and normalizes the response

Application code doesn't need to know which provider is answering.

**Q131.** What do per-team quotas prevent?

**Answer:** One caller exhausting shared capacity or budget

They're enforced by the gateway.

**Q132.** What does routing and fallback decide?

**Answer:** Which model handles a request and what happens when the primary fails or is too slow

For example when a provider is down or rate-limited.

### [04 · Safety &amp; Red-Teaming](level-4/04-safety-red-teaming.md)

**Q133.** How does red-teaming differ from ordinary testing?

**Answer:** It assumes an adversarial user trying to make the system misbehave, instead of checking expected inputs

A model can pass every functional test and still fail red-teaming.

**Q134.** Which technique hides a harmful request by encoding it or splitting it across turns?

**Answer:** Obfuscation

Pattern-matching filters miss it while the model still decodes it.

**Q135.** What is gradual escalation?

**Answer:** Starting benign and shifting incrementally toward the harmful target so no single turn looks alarming

The taxonomy is a checklist, not exhaustive.

**Q136.** What is context stuffing?

**Answer:** Burying the harmful request inside a large benign-looking document or long conversation

Instructions late in a long context may get less scrutiny.

### [05 · Advanced Evaluation at Scale](level-4/05-evaluation-at-scale.md)

**Q137.** What can a static golden set not catch?

**Answer:** Distribution drift, silent model updates and rare-pattern interaction effects at scale

Continuous evaluation adds a live online layer.

**Q138.** Why sample only a small percentage of live traffic for scoring?

**Answer:** Scoring every request is expensive at scale

The lesson uses a 1% default.

**Q139.** Why periodically add human review to LLM-as-judge scoring?

**Answer:** Automated judges themselves need calibration checks against human judgment

Mix automated and human scoring.

**Q140.** What is a silent model update?

**Answer:** A provider changes a model version behind a stable API name, shifting behavior on inputs your eval set doesn't cover

It's a reason to evaluate continuously.

### [06 · Cost Optimization at Scale](level-4/06-cost-optimization-at-scale.md)

**Q141.** What is model tiering?

**Answer:** Routing each request to the cheapest model that reliably handles it, escalating only when needed

Small for classification/extraction, large for complex reasoning.

**Q142.** What does try-cheap-first escalation do?

**Answer:** Calls a small model and escalates to a large one only when confidence is below a threshold

Most real traffic is simple and a small model handles it just as well.

**Q143.** Which workloads are candidates for batch APIs?

**Answer:** Non-interactive ones: nightly summarization, bulk classification, offline eval scoring

They don't need a response in the same interaction.

**Q144.** Why attribute cost to teams?

**Answer:** So the right teams see and own their spend

Small per-request savings compound at platform scale.

### [07 · Latency Engineering &amp; Batching](level-4/07-latency-engineering.md)

**Q145.** Why is average latency the wrong metric?

**Answer:** A 100ms mean can hide a 5-second p99, where 1% of users wait 50x longer

Set budgets and alerts on p99, not the mean.

**Q146.** What does time-to-first-token (TTFT) matter more for?

**Answer:** Perceived speed: users start reading immediately

Stream by default; non-streaming is the exception.

**Q147.** What is p99 latency?

**Answer:** The latency of the slowest 1% of requests

It determines whether users find the product fast or frustrating.

**Q148.** What is the highest-leverage change for perceived responsiveness?

**Answer:** Streaming tokens as they're generated

Chat UIs feel fast even when total generation takes seconds.

### [08 · Security — Prompt Injection &amp; Beyond](level-4/08-security-prompt-injection.md)

**Q149.** How does prompt injection differ from jailbreaking?

**Answer:** It hijacks an LLM-powered application, and the attacker often controls content the app feeds to the model rather than being the user

Examples: a web page, an email or a retrieved file.

**Q150.** What is indirect injection?

**Answer:** The malicious instruction arrives via a third-party data source the model processes for the legitimate user

It's higher-risk for agents because the user never sees or approves it.

**Q151.** What is the most durable defense?

**Answer:** Privilege separation: treat processed content as data, never as instructions

Use structural signals such as delimiters and prompting.

**Q152.** Why are agentic systems at higher risk?

**Answer:** Model output triggers real actions, not just displayed text

An injected instruction could forward emails, for example.

### [09 · Compliance, Privacy &amp; Governance](level-4/09-compliance-privacy-governance.md)

**Q153.** How should data retention be defined?

**Answer:** Per data category, since raw PII prompts, anonymized aggregates and audit logs need different periods

A single setting is too short for audit or too long for PII minimization.

**Q154.** Why redact PII before storing, logging or sending to a third-party provider?

**Answer:** It reduces exposure at the source

Detect with patterns and redact.

**Q155.** What does the lesson say about legal advice?

**Answer:** It is not legal advice; it shows engineering patterns that make compliance possible

Obligations vary by jurisdiction and industry.

**Q156.** Why keep audit trails?

**Answer:** To make the system operable under scrutiny, with consequential actions logged

The lesson's example keeps audit logs for 7 years.

### [10 · Capstone Project](level-4/10-capstone-project.md)

**Q157.** What does the capstone build?

**Answer:** A production-shaped LLM platform serving simulated teams with routing, cost controls, safety layers, evaluation and compliance

Each piece depends on the others as in a real deployment.

**Q158.** What does tiering with escalation do in the platform?

**Answer:** Simple requests use a cheap model and escalate only when needed

It comes from module 6.

**Q159.** What does the audit trail require?

**Answer:** Consequential actions logged in a tamper-evident chain, with PII redacted before storage

It combines module 9's patterns.

**Q160.** What does per-team tracking cover?

**Answer:** Quotas, spend, and p99 latency with a documented fallback path

Latency budgets sit alongside cost attribution.

## Mixed questions across all levels

**Q161.** What does this print?

```
import json
print(json.loads('{"skills": ["go", "postgres"]}')['skills'][0])
```

**Answer:** go

json.loads parses the string into a dict; index 0 of the list is 'go'.

**Q162.** What does this print?

```
import re
print(re.findall(r'[\w.+-]+@[\w-]+\.\w+', 'mail a@b.com or c.d@e.org'))
```

**Answer:** ['a@b.com', 'c.d@e.org']

A simple email pattern finds both addresses (the style of PII redaction in the course).

**Q163.** What does this print?

```
a=[100.0]*95+[5000.0]*5
s=sorted(a)
n=len(s)
print(s[int(n*0.50)], s[int(n*0.99)], sum(a)/n)
```

**Answer:** 100.0 5000.0 345.0

p50 hides the tail; p99 shows it, and the mean (345) is dominated by the common case.

**Q164.** What does this print?

```
import math
logits=[2.1,0.3,-1.0,4.5,0.8]
p=math.exp(logits[3])/sum(math.exp(x) for x in logits)
print(round(-math.log(p),3))
```

**Answer:** 0.126

Next-token cross-entropy loss at one position: −log of the softmax probability of the correct token.

**Q165.** What does this print?

```
w=[0.02,-0.01,0.015,-0.02]
scale=max(abs(x) for x in w)/127
q=[round(x/scale) for x in w]
print(q)
```

**Answer:** [127, -64, 95, -127]

Symmetric int8 quantization: scale = max|w|/127, then round(w/scale).

**Q166.** Your LLM app's second API call forgets the user's name from the first. Why?

**Answer:** The API is stateless; you must resend the full message history

The model only 'remembers' what is in the messages list.

**Q167.** A tool result must be returned to the model. Who executes the tool?

**Answer:** Your program; the model only emits a structured request

The model never runs code.

**Q168.** You get a 429 RateLimitError. What is the right response?

**Answer:** Retry with exponential backoff and jitter, honoring retry-after

A 400 or 401 shouldn't be retried, but 429 and 5xx should.

**Q169.** A chatbot's static system prompt and a 40,000-token manual repeat on every call. How do you cut cost and latency?

**Answer:** Mark the stable prefix with cache_control and put per-request data after it

Prompt caching reuses the already-processed prefix.

**Q170.** Open-ended summaries can't be graded by exact match. What do you use?

**Answer:** LLM-as-judge with a rubric, calibrated against humans

Use a second model call returning a structured verdict.

**Q171.** A long-running agent keeps losing track of its goal. What pattern helps?

**Answer:** A planner/executor split with a logged, approvable plan

It makes the workflow auditable and checkpointable.

**Q172.** You want tools reusable across many apps without rewriting them. What do you use?

**Answer:** An MCP server exposing tools, resources and prompts

MCP standardizes discovery and calls.

**Q173.** Search for 'cheap flights' should find a doc about 'budget airfare'. What makes that possible?

**Answer:** Embeddings with cosine similarity

Embeddings place semantically similar text near each other.

**Q174.** Your team needs to serve an open model to many concurrent users at high throughput. What do you choose?

**Answer:** vLLM with PagedAttention and continuous batching

Ollama is for single-user local convenience.

**Q175.** A model needs a very consistent legal-summary format that prompting can't fully pin down. What is a reasonable next step?

**Answer:** Fine-tune (LoRA/QLoRA) on consistent (input, output) examples

Fine-tuning shapes behavior and style, not facts.

**Q176.** A web page your agent browses says 'IGNORE ALL PREVIOUS INSTRUCTIONS'. What is this and how do you defend?

**Answer:** Indirect prompt injection; treat content as data and use privilege separation and least-privilege tools

The attacker controls content, not the user.

**Q177.** Your mean latency is 100ms but users complain. What should you check?

**Answer:** p99 latency, since the mean hides the tail

1% of users may wait 50x longer.

**Q178.** Why add input AND output guardrails?

**Answer:** Bad input shouldn't enter context, and benign input can still produce a problematic output

Each layer covers a different failure.

**Q179.** You need to see why an agent took a path across a dozen model and tool calls. What do you add?

**Answer:** Tracing: one trace ID with spans for each call and tool

Spans carry timing, tokens and results.

## Practical coding questions

### Truncate chat history safely

Write `truncate(messages, max_turns)` that keeps the last `max_turns` user+assistant pairs (2 messages per turn) and makes sure the kept history starts with a `user` message. Print the roles of the result for the sample list.

Starter code:

```
def truncate(messages, max_turns):
    # your code here
    pass

msgs = [{"role": "user"}, {"role": "assistant"}] * 3 + [{"role": "user"}]
print([m["role"] for m in truncate(msgs, 2)])
```

Expected output:

```text
['user', 'assistant', 'user']
```

??? success "Show a solution"

    ```
    def truncate(messages, max_turns):
        kept = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
        while kept and kept[0]["role"] == "assistant":
            kept = kept[1:]
        return kept

    msgs = [{"role": "user"}, {"role": "assistant"}] * 3 + [{"role": "user"}]
    print([m["role"] for m in truncate(msgs, 2)])
    ```

    Slicing keeps the last 4 messages, which start with an assistant turn; dropping it leaves a history that starts with a user message.

### Backoff delays

Write `backoff(attempt, base=1.0, cap=30.0)` returning `min(cap, base * 2**attempt)` and print the delays for attempts 0 to 6.

Starter code:

```
def backoff(attempt, base=1.0, cap=30.0):
    # your code here
    pass

print([backoff(a) for a in range(7)])
```

Expected output:

```text
[1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
```

??? success "Show a solution"

    ```
    def backoff(attempt, base=1.0, cap=30.0):
        return min(cap, base * 2 ** attempt)

    print([backoff(a) for a in range(7)])
    ```

    Exponential backoff doubles the wait each attempt up to a cap; real code adds random jitter so clients don't retry in lockstep.

---

More free practice: the [LLM Development Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
