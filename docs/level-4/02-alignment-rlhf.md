# 02 · Alignment & RLHF Overview

A base model, fresh out of pretraining, is a raw next-token predictor: it
completes text plausibly, but it has no notion of "answer the user's
question helpfully" versus "continue this text in whatever direction is
statistically likely." Alignment is the set of training steps that turn
that raw completion engine into an assistant that follows instructions,
refuses harmful requests, and prefers helpful, honest, harmless responses.
Understanding how alignment works — even if you never train a model
yourself — explains why deployed models behave the way they do: why they
refuse certain requests, why they sometimes over-refuse benign ones, and
why "just prompt it differently" only gets you so far against training
that pushes the opposite direction.

## The three-stage pipeline

Most production chat models go through three broad stages after
pretraining:

1. **Supervised fine-tuning (SFT)** — the model is trained on curated
   (prompt, ideal-response) pairs written or selected by humans. This
   teaches the model the *format* of being an assistant: answering
   directly, following instructions, using a consistent tone. It's
   ordinary fine-tuning (Level 3, module 6) on a dataset specifically
   built to demonstrate assistant behavior.
2. **Reward modeling (RM)** — humans (or, in RLAIF, another model) rank
   multiple candidate responses to the same prompt from best to worst. A
   separate model is trained to predict that ranking, producing a scalar
   "reward" for any given response. This reward model becomes a proxy for
   "how much would a human like this response."
3. **RLHF / policy optimization** — the SFT model is further trained with
   reinforcement learning, using the reward model's score as the signal to
   maximize, subject to a penalty for straying too far from the SFT model
   (so it doesn't degenerate into reward-hacking gibberish that scores
   well but reads badly).

```python
# Conceptual reward model training — pairwise ranking loss
import torch
import torch.nn.functional as F

def reward_model_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    # Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    # Training pushes the model to score human-preferred responses higher.
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()

# During training: for each (prompt, response_a, response_b, human_preferred) example,
# run both responses through the reward model, compute this loss, backpropagate.
```

## RLHF with PPO (the classical approach)

Proximal Policy Optimization (PPO) is the reinforcement learning
algorithm most associated with RLHF. At a high level, for each training
prompt:

```python
# Conceptual RLHF training step (heavily simplified)
def rlhf_step(prompt, policy_model, ref_model, reward_model, kl_coef=0.1):
    response = policy_model.generate(prompt)                 # sample a response
    reward = reward_model.score(prompt, response)             # how good is it?

    # KL penalty: don't let the policy drift too far from the reference
    # (the original SFT model) — this prevents reward hacking and
    # catastrophic forgetting of general capability.
    policy_logprobs = policy_model.logprobs(prompt, response)
    ref_logprobs = ref_model.logprobs(prompt, response)
    kl_penalty = (policy_logprobs - ref_logprobs).sum()

    total_reward = reward - kl_coef * kl_penalty
    # PPO then uses total_reward as the RL signal, clipping policy updates
    # to stay within a "trust region" of the previous policy for stability.
    return total_reward
```

The KL penalty term is not a minor implementation detail — it's the thing
that keeps RLHF from producing a model that games the reward model with
degenerate, high-scoring-but-useless text (a well-documented failure mode
called reward hacking). Too small a `kl_coef` and the policy drifts into
exploiting quirks of the reward model; too large and the policy barely
moves from the SFT baseline.

## DPO: alignment without a separate RL loop

Direct Preference Optimization (DPO) reformulates the same underlying
objective as a single supervised loss, skipping the separate reward model
and RL training loop entirely — it directly optimizes the policy on
preference pairs:

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # beta controls how far the policy is allowed to move from the reference model
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()
```

DPO has become popular in practice because it removes an entire moving
part (the reward model and its own training instability) while optimizing
a mathematically equivalent objective under reasonable assumptions — a
meaningful engineering simplification, not just a shortcut, which is why
many open-weight aligned models today are DPO-tuned rather than
full-PPO-RLHF-tuned.

## Constitutional AI and RLAIF

Human preference labeling is expensive and slow to scale. Constitutional
AI and RLAIF (RL from AI Feedback) replace or supplement human raters with
model-generated feedback, guided by a written set of principles (a
"constitution"):

```python
constitution_principles = [
    "Choose the response that is more helpful and directly answers the question.",
    "Choose the response that avoids providing instructions for causing harm.",
    "Choose the response that acknowledges uncertainty rather than fabricating facts.",
]

def ai_feedback_rank(prompt, response_a, response_b, critique_model):
    # The critique model is prompted with the constitution and asked to
    # judge which response better satisfies the principles.
    verdict = critique_model.judge(prompt, response_a, response_b, constitution_principles)
    return verdict  # "a" or "b" — used exactly like a human preference label
```

This doesn't remove humans from the loop entirely — the constitution
itself is human-written, and human-labeled data typically anchors the
critique model's judgment — but it lets preference data scale far beyond
what human labeling budgets alone would allow.

## Why aligned models still refuse things you didn't expect

Because alignment optimizes for a reward model's aggregate preferences
across a huge, diverse training distribution, individual edge cases don't
get individually tuned — the model generalizes a *pattern* of
"cautious around topic X" from many training examples, which is why it
can over-refuse benign requests that superficially resemble a harmful
pattern (a chemistry question that resembles a weapons question) or
under-refuse a genuinely harmful request phrased unusually. This is a
distribution-generalization problem, not a bug in any single training
example, and it's why red-teaming (module 4) and layered runtime
guardrails (Level 3, module 8) remain necessary even on well-aligned
models — alignment shifts the *default* behavior; it doesn't guarantee
correct behavior on every input.

## How It Actually Works

The core mechanical insight across PPO-RLHF and DPO is that both are
solving the same constrained optimization problem — maximize expected
human preference while staying close to a reference policy — just via
different mathematical paths. PPO does it explicitly: sample responses,
score them with a learned reward model, and use policy-gradient updates
(clipped to prevent destructively large steps) to shift the model's
output distribution toward higher-reward regions, with the KL term
enforced as an explicit penalty added to the reward signal at every step.

DPO exploits an algebraic identity: under the same reward-maximization
formulation, the optimal policy's log-probability ratio versus the
reference model has a closed-form relationship to the reward. Substituting
that relationship back into the preference-modeling loss (the same
Bradley-Terry pairwise comparison used to train a reward model) yields a
loss expressed purely in terms of the *policy's own* log-probabilities on
chosen versus rejected responses — no separate reward model, no sampling,
no RL update rule, just a classification-style loss you can optimize with
standard supervised gradient descent. This is why DPO is dramatically
simpler to implement and more stable to train: it turns an RL problem
into a supervised-learning problem by algebraic substitution, not by
approximation.

The reason any of this changes model *behavior* rather than just model
*scores* comes back to the same weight-update mechanism as fine-tuning:
gradients computed from these losses flow through the same attention and
feed-forward layers, nudging the probability the model assigns to
generating preferred continuations upward and dispreferred continuations
downward, token by token, until the aggregate effect across millions of
preference pairs reshapes the model's default response style, tone, and
refusal patterns.

## Cheat sheet

| Stage | Purpose |
|---|---|
| SFT | Teach assistant format/behavior from curated examples |
| Reward model | Predict human preference ranking as a scalar score |
| PPO-RLHF | RL against the reward model, KL-penalized to a reference policy |
| DPO | Same objective as RLHF, solved as a direct supervised loss — no RL loop |
| Constitutional AI / RLAIF | Model-generated preference labels guided by written principles |
| KL penalty | Prevents reward hacking and capability collapse during RL |
| Why over-refusal happens | Alignment generalizes patterns across training distribution, not per-case rules |

## Exercise

Take a small open preference dataset (e.g., a subset of Anthropic's HH-RLHF
or a similar public dataset) and implement the `dpo_loss` function above
against a small model (1-3B parameters) using a library like TRL. Train
for one epoch, then compare the fine-tuned model's responses against the
base model's on 10 held-out prompts, scoring each pair with a simple
LLM-as-judge rubric (Level 3, module 9's tracing patterns are useful for
recording every judged comparison). Report how often the DPO-tuned model's
response was preferred, and note any responses where the tuned model
became measurably more cautious or verbose as a side effect.
