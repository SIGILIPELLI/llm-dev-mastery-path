# 06 · Conversation State & Memory

LLM APIs are stateless: every request stands alone, and the model "remembers"
only what you put in the `messages` list. That makes *you* the memory
manager. Do it naively and costs balloon (you re-pay for the whole history on
every turn) until you overflow the context window; do it well and long
conversations stay cheap, fast, and coherent. This module builds a
conversation manager with truncation and summarization, and shows what state
belongs in the system prompt versus the message list.

## A minimal conversation manager

```python
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

class Conversation:
    def __init__(self, system: str):
        self.system = system
        self.messages: list[dict] = []

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        response = client.messages.create(
            model=MODEL, max_tokens=1000,
            system=self.system, messages=self.messages,
        )
        reply = response.content[0].text
        # Append the assistant turn — this IS the memory
        self.messages.append({"role": "assistant", "content": reply})
        self.last_usage = response.usage
        return reply

convo = Conversation("You are a concise cooking assistant.")
print(convo.send("I have eggs, spinach, and feta. Dinner idea?"))
print(convo.send("Make it lower-carb."))          # "it" resolves because history is sent
print(convo.last_usage.input_tokens)              # grows every turn
```

Watch `input_tokens` climb turn after turn — that's the cost of naive memory.
A 50-turn conversation resends turn 1 fifty times.

## Strategy 1: Sliding-window truncation

Keep only the last N turns. Simple, predictable, and fine for chat where old
turns stop mattering:

```python
MAX_TURNS = 10  # keep the last 10 user+assistant pairs

def truncated(messages: list[dict]) -> list[dict]:
    if len(messages) <= MAX_TURNS * 2:
        return messages
    kept = messages[-MAX_TURNS * 2:]
    # History must start with a "user" message — drop a leading assistant turn
    while kept and kept[0]["role"] == "assistant":
        kept = kept[1:]
    return kept
```

Two gotchas: the trimmed history must still start with a `user` message, and
if you're using tools (module 5), never cut between a `tool_use` and its
`tool_result` — trim at whole-exchange boundaries.

The failure mode is obvious: the model abruptly forgets your name, the
budget you stated, the decision from turn 3. Which leads to…

## Strategy 2: Summarize old turns

Instead of dropping old history, compress it — use the model itself to write
a summary, then replace those turns with it:

```python
def summarize_turns(messages: list[dict]) -> str:
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
        if isinstance(m["content"], str)
    )
    response = client.messages.create(
        model="claude-haiku-4-5",   # summarization is easy — use the cheap model
        max_tokens=400,
        messages=[{"role": "user", "content":
            "Summarize this conversation in under 150 words. Preserve all "
            "facts, names, numbers, preferences, and decisions:\n\n" + transcript}],
    )
    return response.content[0].text


class SummarizingConversation(Conversation):
    KEEP_RECENT = 6          # last 6 messages stay verbatim
    TRIGGER = 20             # summarize when history exceeds 20 messages

    def send(self, user_text: str) -> str:
        if len(self.messages) > self.TRIGGER:
            old, recent = self.messages[:-self.KEEP_RECENT], self.messages[-self.KEEP_RECENT:]
            summary = summarize_turns(old)
            self.messages = [
                {"role": "user", "content": f"<conversation_summary>{summary}</conversation_summary>"},
                {"role": "assistant", "content": "Understood — I have the context."},
                *recent,
            ]
        return super().send(user_text)
```

This is exactly what production assistants (including Claude-based coding
agents) do, and hosted APIs increasingly offer it server-side as automatic
"compaction." Rule of thumb: recent turns verbatim, older turns summarized,
critical facts never dropped.

## System-prompt state vs. message state

Two different homes for two different kinds of state:

| | System prompt | Message list |
|---|---|---|
| Holds | Stable facts & rules: persona, user profile, preferences, today's date | The flowing dialogue: questions, answers, tool results |
| Changes | Rarely (per session) | Every turn |
| Survives truncation | Always — it's outside the history | Only if you keep/summarize it |

```python
def build_system(profile: dict) -> str:
    return f"""You are a personal assistant.

<user_profile>
Name: {profile['name']}
Diet: {profile['diet']}
Timezone: {profile['tz']}
</user_profile>

Follow the profile silently; don't recite it."""

convo = Conversation(build_system({"name": "Priya", "diet": "vegetarian", "tz": "CET"}))
```

Durable facts you learn mid-conversation ("I'm allergic to peanuts") deserve
promotion out of the transient history — into the profile that feeds the
system prompt, or into a small database/file. That's the seed of long-term
memory across sessions, which the capstone implements with a notes file.

!!! warning "Caching caveat"
    Providers cache long stable prompt prefixes to cut costs (Level 2 covers
    this). Editing the system prompt every turn breaks that caching — keep
    the system prompt stable within a session and put per-turn facts in
    messages.

## Knowing when to act: count tokens

Trigger summarization by *token* budget, not message count, using the free
counter from module 2:

```python
def history_tokens(system: str, messages: list[dict]) -> int:
    return client.messages.count_tokens(
        model=MODEL, system=system, messages=messages,
    ).input_tokens

if history_tokens(convo.system, convo.messages) > 8_000:
    ...  # summarize now
```

## Cheat sheet

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

## Exercise

Extend `SummarizingConversation` with a `remember(fact: str)` method that
appends durable facts to a `facts` list which is injected into the system
prompt inside `<known_facts>` tags. Then script a 15-turn conversation where
you state your name and favorite cuisine in turn 1, call
`remember()` for both, chat about unrelated topics for 12 turns (enough to
trigger summarization), and finally ask "what's my name and what cuisine do
I like?" Verify the answer survives. Re-run without `remember()` and a
brutal `KEEP_RECENT = 2` — does the summary alone preserve the facts?
