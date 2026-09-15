---
description: "Transformer Internals — Every module so far treated the model as a black box: text in, text out. This module opens the box just enough to explain why the…"
---

# 01 · Transformer Internals

Every module so far treated the model as a black box: text in, text out.
This module opens the box just enough to explain *why* the API behaves the
way it does — attention, KV caches, positional encoding, and what a
forward pass actually computes. You won't train a transformer here; you'll
build a minimal, readable one in plain NumPy to see the mechanism directly.

## Tokens as vectors

Everything a transformer does happens on vectors, not characters. Each
input token id is looked up in an embedding table to get its initial
vector representation:

```python
import numpy as np

VOCAB_SIZE, D_MODEL = 1000, 64
rng = np.random.default_rng(0)
embedding_table = rng.normal(0, 0.02, size=(VOCAB_SIZE, D_MODEL))

def embed(token_ids: list[int]) -> np.ndarray:
    return embedding_table[token_ids]        # shape: (seq_len, d_model)

tokens = [42, 7, 891, 15]
x = embed(tokens)
print(x.shape)   # (4, 64)
```

This is a lookup, not a computation — a token's embedding is a learned
vector shaped by training, not derived from the token's spelling.

## Positional encoding

A transformer's core attention mechanism has no inherent notion of order —
without extra information, "the cat sat" and "sat cat the" would look
identical to it. Positional information is injected by adding a
position-dependent vector to each token's embedding:

```python
def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    position = np.arange(seq_len)[:, None]
    dim = np.arange(d_model)[None, :]
    angle = position / np.power(10000, (2 * (dim // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

x = x + sinusoidal_positions(len(tokens), D_MODEL)
```

Modern production models more commonly use rotary position embeddings
(RoPE), which rotate query/key vectors by a position-dependent angle
instead of adding a fixed vector — better at generalizing to sequence
lengths not seen during training — but the sinusoidal version above makes
the *purpose* (inject order into an otherwise order-agnostic mechanism)
easiest to see directly in code.

## Self-attention

For every token, attention computes a weighted average of *all* tokens'
values, where the weights come from how well each token's "query" matches
every other token's "key":

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(x: np.ndarray, Wq, Wk, Wv, causal_mask: bool = True) -> np.ndarray:
    Q, K, V = x @ Wq, x @ Wk, x @ Wv                 # each: (seq_len, d_k)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])          # (seq_len, seq_len)

    if causal_mask:
        seq_len = x.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(mask, -np.inf, scores)     # token i can't see token j > i

    weights = softmax(scores)                        # each row sums to 1
    return weights @ V                                # weighted average of values

D_K = 64
Wq, Wk, Wv = (rng.normal(0, 0.02, (D_MODEL, D_K)) for _ in range(3))
attn_out = self_attention(x, Wq, Wk, Wv)
print(attn_out.shape)   # (4, 64)
```

The `causal_mask` is what makes this a *language-model* attention: token 2
can attend to tokens 0-2 but never token 3, because at generation time
token 3 doesn't exist yet — the model must be trainable to predict the
next token from only what came before it.

## Multi-head attention

Real models run several attention computations ("heads") in parallel, each
with its own learned Q/K/V projections, letting different heads specialize
in different relationships (one head might track subject-verb agreement,
another local word order):

```python
def multi_head_attention(x: np.ndarray, num_heads: int, params: list[tuple]) -> np.ndarray:
    head_outputs = [self_attention(x, Wq, Wk, Wv) for (Wq, Wk, Wv) in params]
    return np.concatenate(head_outputs, axis=-1)    # (seq_len, num_heads * d_k)
```

## A minimal transformer block

Attention alone can only mix information between positions linearly; a
feed-forward network afterward gives each position's representation
nonlinear processing capacity, and residual connections plus normalization
keep gradients well-behaved across many stacked layers:

```python
def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def feed_forward(x: np.ndarray, W1, W2) -> np.ndarray:
    hidden = np.maximum(0, x @ W1)     # ReLU
    return hidden @ W2

def transformer_block(x: np.ndarray, attn_fn, ff_W1, ff_W2) -> np.ndarray:
    x = x + attn_fn(layer_norm(x))          # residual around attention
    x = x + feed_forward(layer_norm(x), ff_W1, ff_W2)   # residual around FFN
    return x
```

Stack this block N times (32-100+ in production-scale models) and add a
final projection back to vocabulary size to get logits over the next
token — the same architecture underlying every model you've called through
the API in earlier levels, at a scale many orders of magnitude larger.

## KV caching, conceptually

During generation, computing attention for a new token needs the keys and
values of *every prior* token — recomputing all of them from scratch for
every new token would make generating a 500-token response roughly
quadratic in cost. The KV cache stores each token's key/value vectors the
first time they're computed, so generating token N+1 only computes Q/K/V
for the new token and reuses cached K/V for tokens 0..N:

```python
class KVCache:
    def __init__(self):
        self.keys, self.values = [], []

    def append_and_get(self, new_k: np.ndarray, new_v: np.ndarray):
        self.keys.append(new_k)
        self.values.append(new_v)
        return np.vstack(self.keys), np.vstack(self.values)
```

This is exactly the mechanism prompt caching (Level 2, module 2) exposes
at the API level — persisting KV state across a cache boundary instead of
only within one generation run.

## How It Actually Works

Everything downstream of this module — tool calling, structured output,
streaming, agent loops — is client-side orchestration around what you just
built: a stack of attention + feed-forward blocks that turns a sequence of
input token vectors into a probability distribution over the next token.
Attention's weighted-average-of-values mechanism is why context actually
matters to output quality: every token's representation at every layer is
literally an average of other tokens' representations, weighted by learned
relevance — a fact buried under wrong or irrelevant information genuinely
has less influence over the output the more other content competes for
attention weight, which is the mechanistic reason vague or bloated prompts
produce vaguer answers.

The causal mask is why a model can only condition on what precedes a
position, never what follows — the entire notion of "the model doesn't
know what it will say next until it says it" (relevant to why forcing a
model to "show its reasoning" changes its answer) follows directly from
this masking rule, not from any higher-level property of "reasoning."

## Cheat sheet

| Component | Role |
|---|---|
| Embedding table | Maps token ids to learned vectors |
| Positional encoding / RoPE | Injects order into an otherwise order-agnostic mechanism |
| Self-attention | Weighted average of all (visible) tokens' values |
| Causal mask | Blocks attending to future tokens — enables next-token prediction |
| Multi-head | Several parallel attention computations, concatenated |
| Feed-forward + residual + norm | Nonlinear per-position processing, stable deep stacking |
| KV cache | Reuses prior tokens' keys/values instead of recomputing every step |

## Exercise

Extend the NumPy transformer block above into a 2-layer stack with 4 attention
heads each, run it on a toy 6-token sequence, and print the attention weight
matrix for one head at the last layer. Verify the causal mask by confirming
every row's weights are zero above the diagonal. Then implement a tiny KV
cache and show that generating a 5th token reuses the previously computed
keys/values instead of recomputing attention for tokens 0-3 from scratch.
