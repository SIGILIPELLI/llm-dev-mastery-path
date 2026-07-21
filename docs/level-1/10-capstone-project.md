# 10 · Capstone — CLI Personal Assistant

Time to combine everything from Level 1 into one real, multi-file project: a
terminal personal assistant with tools (file search, calculator, note
saving), streaming responses, conversation memory with summarization, cost
tracking, and a small evaluation suite of golden prompts. This is the same
architecture — at miniature scale — as production AI assistants.

Requires `ANTHROPIC_API_KEY` in `.env` as usual, plus
`pip install anthropic python-dotenv`.

## Project layout

```
assistant/
├── README.md
├── .env                # ANTHROPIC_API_KEY=... (gitignored)
├── config.py           # model ids, prices, limits
├── tools.py            # tool schemas + implementations
├── memory.py           # history management + summarization
├── assistant.py        # the agent loop with streaming
├── main.py             # CLI entry point
├── evals.py            # golden-prompt evaluation suite
└── notes.md            # created at runtime by the save_note tool
```

## config.py

```python
# config.py
MODEL = "claude-sonnet-5"
SUMMARIZER_MODEL = "claude-haiku-4-5"

PRICES = {  # $ per million tokens: (input, output)
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

MAX_TOOL_ITERATIONS = 8       # per user turn
SUMMARIZE_OVER_TOKENS = 6000  # compress history past this
KEEP_RECENT_MESSAGES = 6

SYSTEM_PROMPT = """You are Aide, a personal assistant running in a terminal.

Rules:
- Use the calculator tool for ALL arithmetic — never compute in your head.
- Use search_files to answer questions about the user's documents.
- When the user shares a durable fact or asks you to remember something,
  use save_note.
- Be concise; this is a terminal, not an essay contest."""
```

## tools.py

```python
# tools.py
from pathlib import Path
import ast, operator, json, datetime

DOCS_DIR = Path("./docs_workspace").resolve()   # the assistant's search corpus
NOTES_FILE = Path("./notes.md").resolve()

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for ANY math.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_files",
        "description": "Case-insensitive text search across the user's documents. "
                       "Returns matching lines with filenames.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "text to look for"}},
            "required": ["query"],
        },
    },
    {
        "name": "save_note",
        "description": "Append a note to the user's permanent notes file. Use when "
                       "the user shares a durable fact or says 'remember...'.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def _safe_eval(expr: str) -> float:
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")
    return walk(ast.parse(expr, mode="eval").body)

def _search_files(query: str) -> str:
    hits = []
    if DOCS_DIR.exists():
        for f in sorted(DOCS_DIR.glob("**/*.txt")) + sorted(DOCS_DIR.glob("**/*.md")):
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if query.lower() in line.lower():
                    hits.append(f"{f.name}:{i}: {line.strip()}")
    return json.dumps(hits[:20]) if hits else "No matches found."

def _save_note(note: str) -> str:
    stamp = datetime.date.today().isoformat()
    with NOTES_FILE.open("a") as f:
        f.write(f"- [{stamp}] {note}\n")
    return f"Saved note: {note}"

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "calculator":
            return str(_safe_eval(args["expression"]))
        if name == "search_files":
            return _search_files(args["query"])
        if name == "save_note":
            return _save_note(args["note"])
        return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Error: {e}"
```

## memory.py

```python
# memory.py
import anthropic
from config import SUMMARIZER_MODEL, SUMMARIZE_OVER_TOKENS, KEEP_RECENT_MESSAGES, MODEL

def maybe_summarize(client: anthropic.Anthropic, system: str, messages: list) -> list:
    """Compress old history when it grows past the token budget."""
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages
    tokens = client.messages.count_tokens(
        model=MODEL, system=system, messages=messages
    ).input_tokens
    if tokens <= SUMMARIZE_OVER_TOKENS:
        return messages

    old, recent = messages[:-KEEP_RECENT_MESSAGES], messages[-KEEP_RECENT_MESSAGES:]
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in old if isinstance(m["content"], str)
    )
    summary = client.messages.create(
        model=SUMMARIZER_MODEL, max_tokens=400,
        messages=[{"role": "user", "content":
            "Summarize this conversation in <150 words, preserving all facts, "
            "numbers, names, and decisions:\n\n" + transcript}],
    ).content[0].text

    while recent and recent[0]["role"] != "user":
        recent = recent[1:]        # history must resume on a user turn
    return [
        {"role": "user", "content": f"<conversation_summary>{summary}</conversation_summary>"},
        {"role": "assistant", "content": "Understood — I have the context."},
        *recent,
    ]
```

## assistant.py — the streaming agent loop

```python
# assistant.py
import anthropic
from config import MODEL, PRICES, MAX_TOOL_ITERATIONS, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, execute_tool
from memory import maybe_summarize

class CostTracker:
    def __init__(self):
        self.dollars = 0.0
        self.input_tokens = self.output_tokens = 0

    def record(self, model: str, usage) -> None:
        inp, outp = PRICES[model]
        self.dollars += (usage.input_tokens * inp + usage.output_tokens * outp) / 1e6
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

class Assistant:
    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=4, timeout=60.0)
        self.messages: list = []
        self.cost = CostTracker()

    def ask(self, user_text: str, quiet: bool = False) -> str:
        """One user turn: stream text, run tools, loop until done."""
        self.messages = maybe_summarize(self.client, SYSTEM_PROMPT, self.messages)
        self.messages.append({"role": "user", "content": user_text})
        final_text = []

        for _ in range(MAX_TOOL_ITERATIONS):
            with self.client.messages.stream(
                model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS, messages=self.messages,
            ) as stream:
                for text in stream.text_stream:
                    if not quiet:
                        print(text, end="", flush=True)
                    final_text.append(text)
                response = stream.get_final_message()

            self.cost.record(MODEL, response.usage)
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                if not quiet:
                    print()
                return "".join(final_text)

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    if not quiet:
                        print(f"\n  ⚙ {block.name}({block.input})")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": execute_tool(block.name, block.input),
                    })
            self.messages.append({"role": "user", "content": results})

        return "".join(final_text) or "(stopped: too many tool steps)"
```

## main.py — the CLI

```python
# main.py
from dotenv import load_dotenv
from assistant import Assistant

def main() -> None:
    load_dotenv()
    bot = Assistant()
    print("Aide ready. Commands: /cost, /quit")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/cost":
            c = bot.cost
            print(f"session: {c.input_tokens} in / {c.output_tokens} out "
                  f"tokens = ${c.dollars:.4f}")
            continue
        print("aide> ", end="", flush=True)
        bot.ask(user)

if __name__ == "__main__":
    main()
```

A sample session (put a few `.txt`/`.md` files in `docs_workspace/` first):

```
you> remember that my landlord's email is lena@example.com
aide>   ⚙ save_note({'note': "Landlord's email is lena@example.com"})
Noted — I've saved your landlord's email.

you> search my docs for the lease end date, and how many days from July 21 2026 is it?
aide>   ⚙ search_files({'query': 'lease'})
  ⚙ calculator({'expression': '...'})
Your lease ends 2026-09-30 (lease.txt:4) — that's 71 days away.

you> /cost
session: 6412 in / 388 out tokens = $0.0250
```

## evals.py — golden prompts

Even a toy assistant deserves regression tests. Each golden case pairs a
prompt with a *checkable expectation* — a substring, a tool invocation, or a
file side-effect:

```python
# evals.py — run: python evals.py
from dotenv import load_dotenv
from pathlib import Path
from assistant import Assistant

load_dotenv()

def expect_contains(needle: str):
    return lambda reply, bot: needle.lower() in reply.lower()

def expect_note_saved(fragment: str):
    def check(reply, bot):
        notes = Path("notes.md")
        return notes.exists() and fragment.lower() in notes.read_text().lower()
    return check

def expect_used_tool(tool_name: str):
    def check(reply, bot):
        for m in bot.messages:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    if getattr(b, "type", None) == "tool_use" and b.name == tool_name:
                        return True
        return False
    return check

GOLDEN = [
    ("What is 1847 * 293? Use exact arithmetic.",
     [expect_contains("541171"), expect_used_tool("calculator")]),
    ("Remember that my wifi password is tulip-9942.",
     [expect_note_saved("tulip-9942")]),
    ("Please just say the word 'ready' and nothing else.",
     [expect_contains("ready")]),
]

def run() -> None:
    passed = 0
    for i, (prompt, checks) in enumerate(GOLDEN, 1):
        bot = Assistant()                      # fresh state per case
        reply = bot.ask(prompt, quiet=True)
        ok = all(check(reply, bot) for check in checks)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  [{i}] {prompt[:60]}")
        if not ok:
            print(f"      reply was: {reply[:200]!r}")
    print(f"\n{passed}/{len(GOLDEN)} passed — total eval cost ${bot.cost.dollars:.4f}")

if __name__ == "__main__":
    run()
```

Run `python evals.py` after every prompt or tool change — that habit, scaled
up, is how real LLM teams ship without regressions (Level 2 goes deep on
evals).

## README.md

Write one — future-you counts as a user:

```markdown
# Aide — CLI personal assistant

Level 1 capstone of the LLM Development Mastery Path.

## Setup
1. python3 -m venv .venv && source .venv/bin/activate
2. pip install anthropic python-dotenv
3. echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
4. mkdir docs_workspace   # drop .txt/.md files here for search

## Run
python main.py            # chat; /cost shows spend, /quit exits
python evals.py           # golden-prompt regression suite

## Architecture
CLI (main.py) → Assistant loop (assistant.py: streaming + tools + budget)
→ tools.py (calculator, search_files, save_note)
→ memory.py (token-budgeted summarization)
```

## Where you are now

You can: call an LLM API safely, engineer and *evaluate* prompts, force
valid structured output, wire in tools, manage conversation memory, stream
responses, handle failures, track cost, and assemble it all into an agent
with guardrails. That is the core toolkit of applied LLM engineering —
Level 2 builds production patterns (caching, evals at scale, MCP,
multi-agent systems) on top of it.

## Cheat sheet

| Component | Modules it applies |
|-----------|--------------------|
| `tools.py` — schemas, safe implementations, dispatcher | 05 |
| `memory.py` — token-budgeted summarization | 02, 06 |
| `assistant.py` — streaming agent loop + iteration cap | 05, 07, 09 |
| `CostTracker` — usage → dollars | 08 |
| Retries/timeout via SDK config | 08 |
| `evals.py` — golden prompts with checkable expectations | 03 |
| System prompt with tool rules | 02, 03 |

## Exercise

Ship three upgrades: (1) add a `list_notes` tool so the assistant can read
back saved notes, plus a golden case proving a fact saved in one session is
retrievable in a new `Assistant` instance; (2) add a `/model` CLI command
that switches between `claude-sonnet-5` and `claude-haiku-4-5` mid-session,
and compare `/cost` for the same three questions on each; (3) add a hard
budget guardrail — if session cost exceeds $0.50, the assistant refuses
further calls with a clear message. Re-run `evals.py` after each change and
keep it green.
