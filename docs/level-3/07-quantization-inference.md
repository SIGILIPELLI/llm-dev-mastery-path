---
description: "Quantization & Inference Optimization — Module 4 and 5 both referenced quantization in passing — this module explains it directly: what GGUF, AWQ, and…"
---

# 07 · Quantization & Inference Optimization

Module 4 and 5 both referenced quantization in passing — this module
explains it directly: what GGUF, AWQ, and GPTQ actually do to a model's
weights, the real quality-vs-size tradeoff, and other levers for faster
inference.

## What quantization actually does

A model's weights are normally stored as 16-bit floating point numbers
(sometimes 32-bit). Quantization stores each weight with fewer bits — 8,
4, or even fewer — trading numeric precision for a smaller file and faster
memory-bound computation:

```python
import numpy as np

def quantize_to_int8(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quantization — the simplest scheme."""
    scale = np.abs(weights).max() / 127.0
    quantized = np.round(weights / scale).astype(np.int8)
    return quantized, scale

def dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale

weights = np.random.default_rng(0).normal(0, 0.02, size=(4, 4)).astype(np.float32)
q, scale = quantize_to_int8(weights)
reconstructed = dequantize(q, scale)

print("original:     ", weights[0])
print("reconstructed:", reconstructed[0])
print("max error:    ", np.abs(weights - reconstructed).max())
```

Every weight now takes 1 byte instead of 2 or 4, cutting model size (and
memory bandwidth needed to read it during inference) by half to a quarter
— at the cost of the small reconstruction error printed above, repeated
across billions of weights.

## GGUF, AWQ, GPTQ — what differs

These are different *quantization schemes and file formats*, not
different models:

- **GGUF** — the format used by `llama.cpp` and Ollama (module 4);
  supports a range of bit-widths (`Q4_K_M`, `Q5_K_M`, `Q8_0`, etc.) with
  per-block scaling factors (rather than one scale for the whole tensor)
  to reduce error, and is optimized for CPU and consumer-GPU inference.
- **GPTQ** — a post-training quantization method that calibrates
  quantization error using a small sample dataset, minimizing the
  *output* difference (not just weight-value difference) between the
  quantized and original model, layer by layer.
- **AWQ** (Activation-aware Weight Quantization) — observes that a small
  fraction of weights matter disproportionately for output quality
  (correlated with which activations tend to be large) and preserves
  those at higher precision while quantizing the rest more aggressively.

```python
# Loading a GPTQ-quantized model via transformers + auto-gptq
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-3-8B-GPTQ", device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-3-8B-GPTQ")
```

```bash
# Loading a GGUF model via llama.cpp / Ollama — quantization choice is in the filename
ollama pull llama3.1:8b-instruct-q4_K_M
```

## The quality-vs-size tradeoff, measured

Never take "4-bit is fine" on faith — measure it against your own eval set
(Level 2, module 6), since quality loss varies by task and by how
aggressively you quantize:

```python
def evaluate_at_precision(model_variant: str, golden: list[dict]) -> float:
    correct = 0
    for case in golden:
        output = generate(model_variant, case["input"])
        correct += output.strip() == case["expected"]
    return correct / len(golden)

results = {
    "fp16 (baseline)": evaluate_at_precision("llama3.1:8b-instruct-fp16", golden),
    "8-bit":           evaluate_at_precision("llama3.1:8b-instruct-q8_0", golden),
    "4-bit":           evaluate_at_precision("llama3.1:8b-instruct-q4_K_M", golden),
}
for label, score in results.items():
    print(f"{label:20s} {score:.1%}")
```

A common real-world pattern: 8-bit quantization is often nearly
indistinguishable from full precision on most tasks, while aggressive
4-bit (or lower) quantization starts showing measurable degradation on
tasks needing precise numeric or structured reasoning — but the exact
threshold is task- and model-dependent, which is why you measure rather
than assume.

## Speeding up inference beyond quantization

Quantization reduces memory bandwidth needs, which is usually the
bottleneck for single-request generation (memory-bound, not
compute-bound, because generating one token needs the *entire* weight
matrix read from memory even though it's used for one small computation).
Other levers:

- **Speculative decoding** — a small, fast "draft" model proposes several
  tokens ahead; the large model verifies them in one batched forward pass
  instead of one sequential pass per token, accepting correct guesses and
  only falling back to normal generation where the draft was wrong.
- **Flash Attention** — a fused, memory-efficient attention
  implementation that avoids materializing the full attention score
  matrix, reducing memory traffic without changing the mathematical
  result.
- **Batching** (module 5's continuous batching) — amortizes the fixed
  cost of reading weight matrices from memory across many concurrent
  requests' worth of computation.

```python
# Conceptual speculative decoding sketch
def speculative_generate(prompt, draft_model, target_model, num_draft_tokens=4):
    draft_tokens = draft_model.generate(prompt, max_new_tokens=num_draft_tokens)
    # Target model verifies all draft tokens in one forward pass
    accepted = target_model.verify(prompt, draft_tokens)
    return accepted   # tokens the target model agrees with, up to the first disagreement
```

## How It Actually Works

Every weight in a transformer participates in a matrix multiplication
during the forward pass from module 1 — generating each token means
reading every relevant weight matrix from memory at least once. For a
single request generating one token at a time, the GPU spends most of its
time *waiting* for weights to arrive from memory, not computing — this is
why it's called memory-bandwidth-bound rather than compute-bound, and why
quantization (moving fewer bytes for the same weights) speeds things up
even though it does no less arithmetic once weights are dequantized on the
fly.

GPTQ and AWQ differ from naive per-tensor quantization (the int8 sketch
above) by choosing quantization parameters *based on their effect on
model output*, not just on minimizing raw weight-value error. GPTQ
calibrates layer-by-layer using a small dataset, adjusting remaining
weights to compensate for the error already introduced by quantizing
earlier ones in the same layer. AWQ instead identifies which weights
correlate with the largest activation magnitudes for that specific model
and keeps those at higher precision — because rounding error in a weight
that gets multiplied by a large activation contributes proportionally
more to the final output error. Both are more expensive to run once
(calibration is a real computation over sample data) but produce
noticeably better output quality per bit than uniform rounding.

Speculative decoding's speedup comes from the same memory-bandwidth-bound
fact stated above, exploited differently: verifying several draft tokens
in one batched forward pass reads the target model's weights from memory
*once* for multiple tokens' worth of computation, instead of once per
token sequentially — it doesn't reduce total arithmetic, it amortizes the
same fixed memory-read cost that dominates latency across more useful
work per read.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Quantization | Fewer bits per weight → smaller model, faster memory reads |
| GGUF | llama.cpp/Ollama format; per-block scaling; CPU/consumer-GPU friendly |
| GPTQ | Calibrated quantization minimizing output error, layer by layer |
| AWQ | Preserves high-precision for activation-correlated "important" weights |
| Bottleneck | Single-request generation is memory-bandwidth-bound, not compute-bound |
| Speculative decoding | Draft model proposes, target model verifies in one batched pass |
| Always measure | Quality loss is task-dependent — eval, don't assume |

## Exercise

Using Ollama, pull the same model at `q4_K_M`, `q8_0`, and (if your
hardware allows) `fp16` quantization levels. Run a 20-question eval
covering both factual recall and arithmetic/structured-output tasks
against all three, and plot accuracy vs. quantization level per task
category. Also measure tokens/sec for each. Write up whether the
quality/speed tradeoff differs between the two task categories, and why
that's consistent with the "output-aware vs. naive quantization" mechanism
above.
