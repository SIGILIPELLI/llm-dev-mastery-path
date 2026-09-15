---
description: "Fine-Tuning Fundamentals (LoRA/QLoRA) — Every module so far shaped model behavior with prompting — instructions, examples, tool schemas. Fine-tuning…"
---

# 06 · Fine-Tuning Fundamentals (LoRA/QLoRA)

Every module so far shaped model behavior with prompting — instructions,
examples, tool schemas. Fine-tuning changes the model's *weights* instead,
useful when a task needs a narrow style or format so consistently that no
amount of prompting reliably gets there, or when you need to shrink a
large general-purpose model's prompt (fewer few-shot examples needed) to
cut latency and cost at scale.

## When fine-tuning beats prompting

Try prompting (and, if output is inconsistent, more few-shot examples)
first — it's cheaper, faster to iterate, and doesn't require training
infrastructure. Reach for fine-tuning when:

- **A narrow, high-volume task** needs consistent formatting a prompt
  can't fully pin down after real effort (e.g., matching a specific legal
  document style exactly, every time).
- **You need to shrink cost/latency** by baking a long few-shot prompt's
  behavior into weights, so a smaller model or shorter prompt achieves
  the same accuracy in production.
- **The task requires knowledge best taught by examples**, not
  instructions — style transfer, tone matching, structured output in an
  unusual schema the model has never seen described in a prompt before.

Don't reach for fine-tuning to inject *facts* — a model's weights encode
patterns from training, not a queryable knowledge base, and new facts go
stale the moment they're baked in. Use retrieval (module 3) for facts;
reserve fine-tuning for behavior and style.

## Dataset preparation

Fine-tuning data is a set of (input, ideal-output) pairs in the same
`messages` shape you already use for API calls — quality and consistency
matter far more than volume:

```python
import json

def build_example(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a legal-summary assistant. Always structure output as: Parties, Terms, Risks."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }

examples = [
    build_example(contract_text_1, "Parties: ...\nTerms: ...\nRisks: ..."),
    build_example(contract_text_2, "Parties: ...\nTerms: ...\nRisks: ..."),
    # aim for at least a few hundred high-quality, consistent examples
]

with open("finetune_data.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
```

A hundred carefully-curated, consistently-formatted examples reliably
outperforms a thousand noisy ones — every inconsistency in your training
set teaches the model that inconsistency is acceptable. Hold out 10-20% of
examples as an eval set (Level 2, module 6) so you can measure whether
fine-tuning actually improved behavior rather than just changed it.

## LoRA: fine-tuning without touching most of the weights

Full fine-tuning updates every parameter in the model — expensive in
memory and compute, and risks catastrophically forgetting general
capability. **LoRA** (Low-Rank Adaptation) instead freezes the original
weights and trains a small pair of low-rank matrices added alongside each
targeted weight matrix, updating a tiny fraction of total parameters:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,                    # rank of the low-rank matrices — the key size/quality knob
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "v_proj"],   # which weight matrices get an adapter
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 8,034,000,000 || trainable%: 0.05%
```

That 0.05% is the entire point: training updates a tiny adapter instead of
8 billion parameters, which is why LoRA fine-tuning is feasible on a
single consumer GPU where full fine-tuning of the same model would need a
multi-GPU cluster.

## QLoRA: LoRA on a quantized base model

QLoRA goes further by loading the frozen base model in 4-bit quantization
(module 7) and only training the LoRA adapters in higher precision on top
— cutting the memory needed to even *load* the base model, which is often
the binding constraint on consumer hardware:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
```

The base model's 4-bit weights never change during training — only the
LoRA adapter's higher-precision matrices are updated, which is why
accuracy loss from quantizing the frozen base is much smaller in practice
than quantizing a model you then can't adapt at all.

## Training loop (sketch)

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("json", data_files="finetune_data.jsonl")["train"]

def tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=2048)

tokenized = dataset.map(tokenize)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./lora-adapter",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
    ),
    train_dataset=tokenized,
)
trainer.train()
model.save_pretrained("./lora-adapter")   # only the small adapter is saved
```

Watch training loss, but don't trust it alone — run your held-out eval set
through the fine-tuned model after training and compare against the base
model's score on the same set (Level 2, module 6's exact regression-suite
pattern), because a lower training loss doesn't guarantee better real-world
behavior; it can just as easily mean overfitting to quirks of the training
examples.

## Evaluation after fine-tuning

```python
def evaluate_finetuned(golden: list[dict], model, tokenizer) -> float:
    correct = 0
    for case in golden:
        output = generate(model, tokenizer, case["input"])
        correct += matches_format(output, case["expected_format"])   # your own check
    return correct / len(golden)

base_score = evaluate_finetuned(golden_eval, base_model, tokenizer)
finetuned_score = evaluate_finetuned(golden_eval, model, tokenizer)
print(f"base: {base_score:.1%}  finetuned: {finetuned_score:.1%}")
```

If the fine-tuned model doesn't clearly beat the base model plus good
prompting on your eval set, the fine-tune isn't earning its added
complexity (a training pipeline, an adapter to version and deploy, a new
thing that can silently regress) — go back to prompting.

## How It Actually Works

Fine-tuning updates the same weight matrices module 1 introduced
(attention projections, feed-forward layers) using ordinary gradient
descent: run training examples through the model, compute how far its
predicted next-token distribution is from the target output at each
position, and adjust weights to reduce that error, backpropagated through
every layer — the identical process that produced the base model's
weights in the first place, just starting from an already-trained
checkpoint on a much smaller, task-specific dataset instead of from
random initialization on the full pretraining corpus.

LoRA's insight is that the *update* needed to adapt a large weight matrix
to a new narrow task is often well-approximated by a low-rank matrix (the
product of two small matrices), even though the original weight matrix
itself is full-rank — so instead of learning a full-size update, LoRA
learns two small matrices whose product approximates it, added to the
frozen original weight at inference time. This is why `target_modules` and
`r` are the decisions that matter most: they determine how much
adaptation capacity you're giving the model and where, and why a
too-small `r` under-fits (the adapter can't express the needed change)
while a too-large `r` starts to approach the cost of full fine-tuning
without the low-rank benefit.

QLoRA's memory savings come from a separate, complementary fact: loading
weights in 4-bit format only affects the *storage and matrix-multiply
precision* of the frozen base model, and gradients only ever need to flow
into the small LoRA matrices (kept in higher precision) — the frozen base
never needs gradient storage at all, since it never updates, which is the
specific reason QLoRA's memory footprint is so much smaller than
full-precision full fine-tuning even though the *model being adapted* is
identical.

## Cheat sheet

| Concept | Key fact |
|---|---|
| Fine-tune vs. prompt | Prompting first; fine-tune for consistent style/format at volume |
| Not for facts | Use retrieval (module 3) for facts; fine-tuning bakes in behavior, not knowledge |
| Dataset | Consistent `messages`-shaped examples; quality over volume |
| LoRA | Freezes base weights; trains small low-rank adapter matrices |
| `r` (rank) | Key size/quality knob — too small under-fits, too large loses the LoRA benefit |
| QLoRA | LoRA on a 4-bit quantized frozen base — lowest memory footprint |
| Always re-eval | Compare fine-tuned vs. base+prompting on a held-out set, not just training loss |

## Exercise

Prepare a 100-example dataset (even synthetically generated, clearly
labeled as such) for a narrow reformatting task — e.g., converting free-text
meeting notes into a fixed `Decisions / Action Items / Open Questions`
structure. Fine-tune a small open model with QLoRA at `r=8` and `r=32`,
and compare both against zero-shot prompting on a 20-example held-out set
using an exact-format-match eval. Report which approach wins and by how
much, and note the training time and adapter file size for each rank.
