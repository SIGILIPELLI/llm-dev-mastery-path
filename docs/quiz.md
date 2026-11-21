---
title: "Final Quiz"
description: "Test what you have learned across all four levels of LLM Development Mastery Path with a 19-question final quiz."
---

# Final Quiz

19 questions drawn from all four levels of this course. You need **70% (14 out of 19)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "What does this print?\n```\nimport json\nprint(json.loads('{\"skills\": [\"go\", \"postgres\"]}')['skills'][0])\n```",
    "options": [
      "postgres",
      "['go', 'postgres']",
      "go",
      "skills"
    ],
    "correctIndex": 2,
    "explanation": "json.loads parses the string into a dict; index 0 of the list is 'go'."
  },
  {
    "question": "What does this print?\n```\nimport re\nprint(re.findall(r'[\\w.+-]+@[\\w-]+\\.\\w+', 'mail a@b.com or c.d@e.org'))\n```",
    "options": [
      "['a@b.com', 'c.d@e.org']",
      "[]",
      "['a', 'c.d']",
      "['a@b.com']"
    ],
    "correctIndex": 0,
    "explanation": "A simple email pattern finds both addresses (the style of PII redaction in the course)."
  },
  {
    "question": "What does this print?\n```\na=[100.0]*95+[5000.0]*5\ns=sorted(a)\nn=len(s)\nprint(s[int(n*0.50)], s[int(n*0.99)], sum(a)/n)\n```",
    "options": [
      "5000.0 5000.0 5000.0",
      "100.0 100.0 100.0",
      "100.0 5000.0 345.0",
      "345.0 345.0 345.0"
    ],
    "correctIndex": 2,
    "explanation": "p50 hides the tail; p99 shows it, and the mean (345) is dominated by the common case."
  },
  {
    "question": "What does this print?\n```\nimport math\nlogits=[2.1,0.3,-1.0,4.5,0.8]\np=math.exp(logits[3])/sum(math.exp(x) for x in logits)\nprint(round(-math.log(p),3))\n```",
    "options": [
      "4.5",
      "0.126",
      "0.0",
      "1.0"
    ],
    "correctIndex": 1,
    "explanation": "Next-token cross-entropy loss at one position: −log of the softmax probability of the correct token."
  },
  {
    "question": "What does this print?\n```\nw=[0.02,-0.01,0.015,-0.02]\nscale=max(abs(x) for x in w)/127\nq=[round(x/scale) for x in w]\nprint(q)\n```",
    "options": [
      "[2, -1, 1, -2]",
      "[127, -64, 95, -127]",
      "[127, 64, 95, 127]",
      "[0, 0, 0, 0]"
    ],
    "correctIndex": 1,
    "explanation": "Symmetric int8 quantization: scale = max|w|/127, then round(w/scale)."
  },
  {
    "question": "Your LLM app's second API call forgets the user's name from the first. Why?",
    "options": [
      "The system prompt is missing",
      "max_tokens is too small",
      "The API is stateless; you must resend the full message history",
      "The model has a bug"
    ],
    "correctIndex": 2,
    "explanation": "The model only 'remembers' what is in the messages list."
  },
  {
    "question": "A tool result must be returned to the model. Who executes the tool?",
    "options": [
      "Your program; the model only emits a structured request",
      "The API provider always",
      "The model",
      "The tokenizer"
    ],
    "correctIndex": 0,
    "explanation": "The model never runs code."
  },
  {
    "question": "You get a 429 RateLimitError. What is the right response?",
    "options": [
      "Change the model name",
      "Retry with exponential backoff and jitter, honoring retry-after",
      "Retry instantly in a loop",
      "Ignore it"
    ],
    "correctIndex": 1,
    "explanation": "A 400 or 401 shouldn't be retried, but 429 and 5xx should."
  },
  {
    "question": "A chatbot's static system prompt and a 40,000-token manual repeat on every call. How do you cut cost and latency?",
    "options": [
      "Increase max_tokens",
      "Remove the system prompt",
      "Mark the stable prefix with cache_control and put per-request data after it",
      "Shorten the manual"
    ],
    "correctIndex": 2,
    "explanation": "Prompt caching reuses the already-processed prefix."
  },
  {
    "question": "Open-ended summaries can't be graded by exact match. What do you use?",
    "options": [
      "String equality",
      "Regex only",
      "LLM-as-judge with a rubric, calibrated against humans",
      "Nothing"
    ],
    "correctIndex": 2,
    "explanation": "Use a second model call returning a structured verdict."
  },
  {
    "question": "A long-running agent keeps losing track of its goal. What pattern helps?",
    "options": [
      "A bigger max_tokens",
      "Removing tools",
      "More few-shot examples",
      "A planner/executor split with a logged, approvable plan"
    ],
    "correctIndex": 3,
    "explanation": "It makes the workflow auditable and checkpointable."
  },
  {
    "question": "You want tools reusable across many apps without rewriting them. What do you use?",
    "options": [
      "A different SDK",
      "A longer prompt",
      "An MCP server exposing tools, resources and prompts",
      "Copy-paste tool definitions"
    ],
    "correctIndex": 2,
    "explanation": "MCP standardizes discovery and calls."
  },
  {
    "question": "Search for 'cheap flights' should find a doc about 'budget airfare'. What makes that possible?",
    "options": [
      "Regex",
      "Keyword matching",
      "SQL LIKE",
      "Embeddings with cosine similarity"
    ],
    "correctIndex": 3,
    "explanation": "Embeddings place semantically similar text near each other."
  },
  {
    "question": "Your team needs to serve an open model to many concurrent users at high throughput. What do you choose?",
    "options": [
      "A cron job",
      "A notebook",
      "vLLM with PagedAttention and continuous batching",
      "Ollama on a laptop"
    ],
    "correctIndex": 2,
    "explanation": "Ollama is for single-user local convenience."
  },
  {
    "question": "A model needs a very consistent legal-summary format that prompting can't fully pin down. What is a reasonable next step?",
    "options": [
      "Lower max_tokens",
      "Inject facts via fine-tuning",
      "Fine-tune (LoRA/QLoRA) on consistent (input, output) examples",
      "Remove the system prompt"
    ],
    "correctIndex": 2,
    "explanation": "Fine-tuning shapes behavior and style, not facts."
  },
  {
    "question": "A web page your agent browses says 'IGNORE ALL PREVIOUS INSTRUCTIONS'. What is this and how do you defend?",
    "options": [
      "A syntax error",
      "A rate limit",
      "Indirect prompt injection; treat content as data and use privilege separation and least-privilege tools",
      "A jailbreak by the user; ignore it"
    ],
    "correctIndex": 2,
    "explanation": "The attacker controls content, not the user."
  },
  {
    "question": "Your mean latency is 100ms but users complain. What should you check?",
    "options": [
      "The mean again",
      "Token counts",
      "p99 latency, since the mean hides the tail",
      "The system prompt"
    ],
    "correctIndex": 2,
    "explanation": "1% of users may wait 50x longer."
  },
  {
    "question": "Why add input AND output guardrails?",
    "options": [
      "Guardrails are decorative",
      "To speed up generation",
      "Bad input shouldn't enter context, and benign input can still produce a problematic output",
      "To avoid logging"
    ],
    "correctIndex": 2,
    "explanation": "Each layer covers a different failure."
  },
  {
    "question": "You need to see why an agent took a path across a dozen model and tool calls. What do you add?",
    "options": [
      "A bigger model",
      "Tracing: one trace ID with spans for each call and tool",
      "A longer prompt",
      "More print statements"
    ],
    "correctIndex": 1,
    "explanation": "Spans carry timing, tokens and results."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### Truncate chat history safely

Write `truncate(messages, max_turns)` that keeps the last `max_turns` user+assistant pairs (2 messages per turn) and makes sure the kept history starts with a `user` message. Print the roles of the result for the sample list.

**Starter code**

```python
def truncate(messages, max_turns):
    # your code here
    pass

msgs = [{"role": "user"}, {"role": "assistant"}] * 3 + [{"role": "user"}]
print([m["role"] for m in truncate(msgs, 2)])
```

**Expected output**

```text
['user', 'assistant', 'user']
```

<details><summary>Show solution and explanation</summary>

<pre><code>def truncate(messages, max_turns):
    kept = messages[-max_turns * 2:] if len(messages) &gt; max_turns * 2 else messages
    while kept and kept[0][&quot;role&quot;] == &quot;assistant&quot;:
        kept = kept[1:]
    return kept

msgs = [{&quot;role&quot;: &quot;user&quot;}, {&quot;role&quot;: &quot;assistant&quot;}] * 3 + [{&quot;role&quot;: &quot;user&quot;}]
print([m[&quot;role&quot;] for m in truncate(msgs, 2)])</code></pre>

<p>Slicing keeps the last 4 messages, which start with an assistant turn; dropping it leaves a history that starts with a user message.</p>
</details>

### Backoff delays

Write `backoff(attempt, base=1.0, cap=30.0)` returning `min(cap, base * 2**attempt)` and print the delays for attempts 0 to 6.

**Starter code**

```python
def backoff(attempt, base=1.0, cap=30.0):
    # your code here
    pass

print([backoff(a) for a in range(7)])
```

**Expected output**

```text
[1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
```

<details><summary>Show solution and explanation</summary>

<pre><code>def backoff(attempt, base=1.0, cap=30.0):
    return min(cap, base * 2 ** attempt)

print([backoff(a) for a in range(7)])</code></pre>

<p>Exponential backoff doubles the wait each attempt up to a cap; real code adds random jitter so clients don&#x27;t retry in lockstep.</p>
</details>
