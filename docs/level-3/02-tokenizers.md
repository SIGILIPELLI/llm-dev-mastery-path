# 02 · Tokenizers Deep Dive

Every module up to now treated "tokens" as an abstract cost/length unit.
This module builds a byte-pair encoding (BPE) tokenizer from scratch to
show exactly how text becomes the integer ids the transformer in module 1
actually consumes — and why token counts differ between models, and even
between different strings you'd expect to cost the same.

## Why not just split on characters or words?

Character-level tokenization keeps vocabulary tiny but makes sequences
very long (every character is a step of expensive attention computation).
Word-level tokenization keeps sequences short but the vocabulary explodes
(every inflection, typo, and rare word needs its own slot, and unseen words
have no representation at all). BPE sits between the two: common words stay
single tokens, rare words decompose into meaningful sub-word pieces, and
literally any string — including gibberish or new words — can be
represented, because worst-case it falls back to individual bytes.

## Building a BPE tokenizer

BPE starts from individual bytes/characters and iteratively merges the
most frequent adjacent pair into a new token, building up a vocabulary of
whatever substrings actually recur in the training corpus:

```python
from collections import Counter

def get_pair_counts(corpus: list[list[str]]) -> Counter:
    counts = Counter()
    for word in corpus:
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += 1
    return counts

def merge_pair(corpus: list[list[str]], pair: tuple) -> list[list[str]]:
    merged = "".join(pair)
    new_corpus = []
    for word in corpus:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus

def train_bpe(words: list[str], num_merges: int) -> list[tuple]:
    corpus = [list(w) + ["</w>"] for w in words]     # start at character level
    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        corpus = merge_pair(corpus, best_pair)
        merges.append(best_pair)
    return merges

training_words = ["lower", "lowest", "newer", "wider", "newest"] * 20
merges = train_bpe(training_words, num_merges=10)
print(merges[:5])
# e.g. [('e', 'r'), ('er', '</w>'), ('n', 'e'), ('ne', 'w'), ('new', 'er</w>')]
```

The learned merges are the tokenizer's vocabulary, in the order they were
discovered — `("e", "r")` merging first means "er" was the single most
common adjacent character pair across the training corpus, and it becomes
one token going forward.

## Encoding with a trained tokenizer

Applying the merges in the order they were learned to a new string is how
encoding works — the same merge rules that built the vocabulary now
segment new text into that vocabulary's pieces:

```python
def encode(word: str, merges: list[tuple]) -> list[str]:
    tokens = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append("".join(pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

print(encode("newer", merges))    # likely ['new', 'er</w>'] if trained as above
print(encode("newestly", merges)) # unseen word — falls back to smaller known pieces
```

An unseen word never fails outright; it just decomposes into whatever
pieces the trained merges cover, down to individual characters/bytes in
the worst case — this is why a tokenizer never returns an "unknown token"
error the way a fixed word-level vocabulary would.

## Why token counts differ across models and strings

Real tokenizers (used by production LLMs) are trained on huge, differently
composed corpora, so the same string produces a different token count on
different models — a string efficient in one model's vocabulary (because
similar text was common in its training data) may split into many more
pieces in another's:

- **Non-English text** often tokenizes less efficiently than English on
  vocabularies trained mostly on English corpora — expect noticeably more
  tokens per character.
- **Code and rare identifiers** (`snake_case_variable_123`) frequently
  split into several sub-word tokens, since exact identifier strings are
  far less likely to recur verbatim in training data than common English
  words.
- **Whitespace and casing changes the tokenization** — `"Hello"`,
  `" Hello"`, and `"hello"` are commonly three distinct tokens or token
  sequences, not variants of the same one, because BPE merges operate on
  the literal byte sequence, whitespace included.

```python
# Illustrative: don't assume word count ≈ token count
for s in ["hello world", "  hello world", "HELLO WORLD", "hello_world_var"]:
    print(s, "->", encode(s.replace(" ", "_"), merges))  # toy tokenizer, illustrative only
```

For real cost estimation, always use the provider's actual tokenizer or a
`count_tokens` API rather than a word-count heuristic — the module 8 cost
tracking from Level 1 should call the real counting endpoint, not
`len(text.split())`.

## Tokenization gotchas that bite in practice

- **Numbers split unpredictably.** `"12345"` might be one token, two, or
  five depending on the vocabulary — arithmetic reliability partly
  depends on how consistently a model's tokenizer represents digit
  sequences, which is one reason Level 1's tool-calling module told you to
  offload math to a calculator tool rather than trust generated digits.
- **Token boundaries don't align with word boundaries.** A regex or
  string-length-based prompt truncation strategy can cut a token in half
  from the model's perspective (though not from yours, since you're
  slicing the string) — truncate by *token count*, not character count,
  when you're near a context-window limit.
- **A single "character" can be several tokens.** Emoji and many non-Latin
  scripts are represented as multiple bytes, which can become multiple
  tokens — never assume `len(string) ≈ len(tokens)`.

## How It Actually Works

Tokenization is not part of the transformer's learned computation from
module 1 — it's a fixed, separately-trained preprocessing step that runs
*before* the embedding lookup, converting raw text into the sequence of
integer ids the model was trained against. Once a tokenizer's vocabulary
is fixed (typically frozen before pretraining begins), it never changes
for that model — this is precisely why token counts for identical text are
consistent within one model but vary between models with different
vocabularies: each vocabulary encodes a different set of frequent
substrings, learned from a different training corpus via the same
frequency-driven merge process demonstrated above at a much larger scale
(commonly ten-thousand to several-hundred-thousand merges, not ten).

This also explains the numeric and multilingual gotchas directly: BPE
merges are chosen purely by byte/character-pair frequency in the training
corpus, with no built-in concept of "this is a number" or "this is a word
boundary" — a digit sequence's tokenization depends entirely on how often
that exact digit substring appeared during training, and a
low-resource language's tokenization is worse simply because its
substrings were rarer in the corpus the merges were learned from, so fewer
of its common patterns earned a place in the vocabulary as a single token.

## Cheat sheet

| Concept | Key fact |
|---|---|
| BPE | Iteratively merges the most frequent adjacent pair into a new token |
| Vocabulary | The ordered list of learned merges, fixed after training |
| Unseen words | Decompose into smaller known pieces — never a hard failure |
| Token ≠ word ≠ character | None of these counts are interchangeable |
| Cross-model variance | Different training corpora → different vocabularies → different counts |
| Cost estimation | Always use the real tokenizer/count API, never a word-count heuristic |

## Exercise

Train the toy BPE tokenizer above on a corpus of at least 50 short English
sentences for 60 merges, then encode 5 test strings: a common English
sentence, a snake_case code identifier, a string of digits, a sentence
with unusual capitalization, and a word absent from training. Print the
token count for each and explain in a comment, for each case, *why* it
tokenized the way it did based on which merges fired.
