# Teaching a Transformer to Think

### DeepSeek-R1 Through the Lens of *Formal Algorithms for Transformers*

A first-pass reading of [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, January 2025).

Presented for a class that has internalized Phuong & Hutter (2022).

---

## Roadmap

| Section | Topic |
|---------|-------|
| 1 | Our Starting Point — The Algorithms We Know |
| 2 | The Question R1 Asks |
| 3 | What Changed — Shown as Diffs |
| 4 | The Full Training Pipeline |
| 5 | Discussion Questions |
| 6 | Summary |

---

## 1 · Our Starting Point

We've spent the semester with three algorithms that define how a decoder-only transformer works end-to-end. Let's put them side by side as a refresher, because everything DeepSeek-R1 does is a *modification* to this stack.

### The Stack We Know

```
┌─────────────────────────────────────────────────────────┐
│  Algorithm 13: DTransformer(x | θ)  →  P               │
│    The forward pass. Tokens in, probability matrix out. │
│    Unchanged by R1. Not one line is different.          │
├─────────────────────────────────────────────────────────┤
│  Algorithm 14: DTraining(x_{1:N}, θ)  →  θ̂             │
│    Minimize cross-entropy on next-token prediction.     │
│    ← THIS IS WHAT R1 REPLACES.                         │
├─────────────────────────────────────────────────────────┤
│  Algorithm 15: DInference(x, θ̂)  →  y                  │
│    Prompt → sample tokens autoregressively.             │
│    ← THIS IS WHAT R1 EXTENDS.                          │
└─────────────────────────────────────────────────────────┘
```

The key thing to notice: the *architecture* (the forward pass, the attention mechanism, MHAttention, layer\_norm — all of it) is left completely alone. DeepSeek-R1 changes **how the model is trained** and **how its outputs are structured**. The transformer itself is the same object we've been studying all semester.

---

## 2 · The Question R1 Asks

Phuong & Hutter's `DTraining` optimizes a single objective:

> Maximize the log-likelihood of the next token given all preceding tokens.

$$\text{loss}(\theta) = -\sum_{t=1}^{\ell-1} \log P_\theta\!\left(x[t{+}1] \mid x[1{:}t]\right)$$

This is *imitation*. The model learns to mimic the statistical patterns in its training corpus. If the corpus contains reasoning, the model learns to produce text that *looks like* reasoning.

**DeepSeek-R1 asks:** What if, instead of imitating reasoning, we *reward* correct answers and let the model figure out how to reason on its own?

This is the shift from **supervised learning** to **reinforcement learning**. The model no longer gets told *what tokens to produce* — it gets told *whether its final answer was right*, and must discover the intermediate reasoning steps itself.

Let's see exactly what changes in the formal algorithms.

---

## 3 · What Changed — Shown as Diffs

### 3.1 · Diff #1: The Training Objective

This is the central change. Let's compare Phuong & Hutter's `DTraining` with DeepSeek-R1-Zero's training loop side by side.

**Phuong & Hutter — Algorithm 14: `DTraining`**

```
DTraining(x_{1:N_data}, θ) → θ̂
    for epoch = 1, 2, ..., N_epochs:
        for n = 1, 2, ..., N_data:
            ℓ ← length(x_n)
            P(θ) ← DTransformer(x_n | θ)
            loss(θ) = −Σ_{t=1}^{ℓ−1} log P(θ)[x_n[t+1], t]     ← cross-entropy
            θ ← θ − η · ∇loss(θ)                                 ← gradient descent
    return θ̂ = θ
```

**DeepSeek-R1-Zero — Algorithm 5: `TrainR1Zero`**

```
TrainR1Zero(Q, θ_0) → θ̂
    θ ← θ_0;  θ_ref ← θ_0
    for s = 1, 2, ..., N_steps:
        sample mini-batch {q_b} from Q
        θ_old ← θ
        for each q_b:
            for i = 1, ..., G:                                    ← sample G outputs per question
                o_i ~ π_{θ_old}(· | Template(q_b))
            for i = 1, ..., G:
                r_i ← RuleReward(q_b, o_i, a_b*)                 ← did it get the right answer?
            {A_i} ← GRPOAdvantage({r_i})                         ← normalize within group
        J ← (1/B) Σ_b GRPOObjective(q_b, {o_i}, {A_i} | θ, θ_old, θ_ref)
        θ ← θ + η · ∇_θ J                                       ← gradient ASCENT (maximize)
        if s mod N_ref = 0:  θ_ref ← θ
    return θ̂ = θ
```

Here is the diff:

```diff
  TRAINING A DECODER-ONLY TRANSFORMER
  ────────────────────────────────────

- Input:  {x_n}, a dataset of token sequences
+ Input:  Q, a set of questions with ground-truth answers

- for epoch = 1, ..., N_epochs:
-     for n = 1, ..., N_data:
+ θ_ref ← θ_0
+ for s = 1, ..., N_steps:
+     sample mini-batch {q_b} from Q
+     θ_old ← θ

-         P(θ) ← DTransformer(x_n | θ)                    // one forward pass
+         for i = 1, ..., G:                               // G forward passes per question
+             o_i ~ π_{θ_old}(· | Template(q_b))           // full generation, not just one step

-         loss(θ) = −Σ_t log P(θ)[x_n[t+1], t]            // cross-entropy on every token
+         r_i ← RuleReward(q_b, o_i, a_b*)                // binary: was the final answer right?
+         A_i ← (r_i − mean) / std                        // normalize within the group

-         θ ← θ − η · ∇ loss(θ)                           // minimize loss
+         J ← Σ min(ρ_i · A_i, clip(ρ_i, 1−ε, 1+ε) · A_i) − β·D_KL
+                                                          // clipped surrogate + KL penalty
+         θ ← θ + η · ∇ J                                 // maximize objective

+     if s mod N_ref = 0:  θ_ref ← θ                      // refresh reference policy

  return θ̂ = θ
```

#### What to notice in this diff

**Five things changed. Nothing else.**

1. **The data** went from token sequences to question-answer pairs. The model no longer sees the "right" token sequence — it has to generate one.

2. **The forward pass** went from a single call to `DTransformer` to sampling $G = 16$ complete output sequences per question. Each output is a full autoregressive rollout (the model runs `DInference` internally, many tokens long).

3. **The loss signal** went from per-token cross-entropy to a scalar reward on the *final answer*. The model gets no signal about which intermediate tokens were good — only whether the end result was correct.

4. **The gradient** went from descent (minimize loss) to ascent (maximize objective), using a clipped surrogate from the PPO family.

5. **A reference policy** was added. This is a frozen copy of $\theta$, updated every 400 steps, used to compute a KL divergence penalty that prevents the policy from drifting too far too fast.

---

### 3.2 · Diff #2: The Advantage Estimation (GRPO vs. PPO)

DeepSeek-R1 doesn't use PPO exactly. It uses **GRPO** (Group Relative Policy Optimization), which simplifies advantage estimation dramatically.

In standard PPO, you need a *value function* $V_\phi(s)$ — a separate neural network, often as large as the policy itself, trained to estimate expected future reward at each token position. The advantage is:

$$A_t^{\text{PPO}} = \sum_{k=0}^{T-t} (\gamma\lambda)^k \big(r_{t+k} + \gamma V_\phi(s_{t+k+1}) - V_\phi(s_{t+k})\big)$$

GRPO replaces all of this with a z-score:

```diff
  ADVANTAGE ESTIMATION
  ────────────────────

- // PPO: Generalized Advantage Estimation
- Train value network V_ϕ(s) ≈ E[Σ γ^k r_{t+k}]         // ~671B parameters
- for each token position t:
-     A_t = Σ_k (γλ)^k (r_{t+k} + γ V_ϕ(s_{t+k+1}) − V_ϕ(s_{t+k}))

+ // GRPO: Group Relative Advantage
+ Sample G outputs for question q
+ Compute reward r_i for each output
+ μ ← (1/G) Σ r_i                                        // group mean
+ σ ← sqrt((1/G) Σ (r_i − μ)²)                           // group std
+ A_i ← (r_i − μ) / σ                                    // that's it
```

That's it. No value network. No temporal structure. No per-token credits. Just: *compared to the other outputs in your group, how good was your reward?*

For a 671B-parameter model, this eliminates the need to train and store a second 671B-parameter value network — a practical necessity, not just a simplification.

---

### 3.3 · Diff #3: The Reward Signal

In `DTraining`, the "reward" is implicit — it's the cross-entropy loss, which provides a dense gradient signal at every token position. R1's reward is sparse and comes only at the end:

```diff
  TRAINING SIGNAL
  ───────────────

- // Phuong & Hutter: Dense, per-token
- for t = 1, ..., ℓ−1:
-     signal_t = −log P(θ)[x[t+1], t]                     // every token gets a gradient

+ // DeepSeek-R1: Sparse, outcome-only
+ r_acc ← 𝟙[answer is correct]                            // 0 or 1
+ r_fmt ← 𝟙[output has valid <think>...</think><answer>...</answer> tags]
+ r_lang ← fraction of words in target language
+ r = r_acc + r_fmt + r_lang                               // single scalar for entire sequence
```

This is a *dramatically* weaker signal. The model generates thousands of tokens of reasoning, and all it hears back is "correct" or "incorrect." It has to figure out *on its own* which tokens in its chain of thought mattered.

The paper's key finding: **this is enough**. With $G = 16$ samples per question, the group-relative advantage provides enough contrast (some outputs right, some wrong) for the model to learn which reasoning patterns lead to correct answers.

---

### 3.4 · Diff #4: Inference — Structured Output

The last diff is at inference time. Phuong & Hutter's `DInference` produces an unstructured token sequence:

```diff
  INFERENCE
  ─────────

- DInference(x, θ̂) → y
-     ℓ ← length(x)
-     for i = 1, ..., ℓ_gen:
-         P ← DTransformer(x | θ̂)
-         p ← P[:, ℓ+i−1]
-         sample y from p^{1/τ}
-         x ← [x, y]
-     return y = x[ℓ+1 : ℓ+ℓ_gen]                         // flat token sequence

+ R1Inference(x, θ̂) → (c, a)
+     z ← Template(x)                                      // wrap in conversation template
+     y ← <think>                                          // force-start with thinking tag
+     for t = 1, ..., ℓ_max:
+         p ← π_{θ̂}(· | [z, y])
+         sample y_t from p^{1/τ} with nucleus sampling
+         y ← [y, y_t]
+         if y_t = EOS or y ends with </answer>:  break
+     c ← tokens between <think> and </think>              // chain-of-thought
+     a ← tokens between <answer> and </answer>            // final answer
+     return (c, a)                                        // structured pair, not flat sequence
```

#### What to notice

The output goes from $y \in V^*$ (a flat sequence) to $(c, a) \in V^* \times V^*$ (a structured pair). But this structure is **not architecturally enforced**. There is no change to the transformer's forward pass. The `<think>` and `<answer>` tags are just tokens in the vocabulary.

The structure is maintained by:
1. The **template** (which starts generation with `<think>`), and
2. The **format reward** $r_{\text{fmt}}$ (which penalizes outputs that don't close their tags properly).

The model learned to produce structured output *because it was rewarded for it*, not because it was constrained to.

---

## 4 · The Full Training Pipeline

DeepSeek-R1-Zero (pure RL, no supervised data) works surprisingly well — reasoning emerges spontaneously, including self-correction behaviors the authors call "aha moments." But it has problems: poor readability, language mixing, and no non-reasoning capabilities.

The full DeepSeek-R1 uses a four-stage pipeline. Here it is mapped against the Phuong & Hutter primitives:

```
 Stage   What Happens                   P&H Primitive Used
 ─────   ──────────────────────────────  ─────────────────────────
   1     Cold-Start SFT                  DTraining (on ~thousands
         Fine-tune base on curated          of long-CoT examples)
         long chain-of-thought data.

   2     Reasoning-Focused RL            TrainR1Zero-style GRPO
         GRPO on math, code, STEM,       (replaces DTraining)
         logic. Rule-based rewards
         only. ε = 10, τ = 1.0.

   3     Rejection Sampling + SFT        DInference → filter →
         Generate from Stage 2 model,       DTraining
         keep correct + readable
         outputs, mix with 200k
         non-reasoning data. SFT on
         800k total from scratch.

   4     Comprehensive RL                GRPO again, but now with
         GRPO on reasoning + general        both rule-based AND
         tasks. Model-based rewards         model-based rewards.
         added for helpfulness/safety.
         General data in last 400/1700
         steps only.
```

### Why four stages?

Each stage addresses a failure mode of the previous one:

- **Stage 1** (Cold-Start SFT): Gives the model a readable reasoning format to start from, so RL doesn't have to discover formatting from scratch.
- **Stage 2** (Reasoning RL): Improves actual reasoning ability beyond what imitation can provide.
- **Stage 3** (Rejection Sampling + SFT): Harvests the best reasoning traces from Stage 2 and combines them with general-purpose data. Resets to the base model and trains from scratch on this curated mix — a form of *distillation*.
- **Stage 4** (Comprehensive RL): Final polish. Adds helpfulness and safety rewards. General instruction data is introduced only in the last 400 steps to prevent reward hacking.

Notice the alternating pattern: **SFT → RL → SFT → RL**. Each SFT stage stabilizes what RL discovered; each RL stage pushes beyond what SFT could teach.

---

## 5 · Discussion Questions

### Question 1: The Degenerate Group Problem

Look at Algorithm 1 (`GRPOAdvantage`):

```
μ ← (1/G) Σ r_i
σ ← sqrt((1/G) Σ (r_i − μ)²)
A_i ← (r_i − μ) / σ
```

When all $G$ outputs for a question receive the **same reward** (all correct or all wrong), $\sigma = 0$ and the advantage $A_i$ is undefined (division by zero). The pseudocode does not handle this case.

**Part (a):** How often would you expect this to happen during training, and how does the frequency change as the model improves? Think about what the reward distribution within a group looks like at initialization (base model on math problems) versus after thousands of steps.

**Part (b):** GRPO implicitly assumes that *within-group variance* is a meaningful signal — that some outputs will be right and some wrong, providing contrast. When this assumption fails (all right or all wrong), the model receives zero gradient for that question. Propose a modification to the advantage estimator that degrades gracefully in these cases. Consider: should the model still learn something when all its outputs are correct?

**Part (c):** Compare this to PPO's learned value function. In PPO, even when all outputs are correct, the value function can still provide a gradient by distinguishing *better* correct solutions from *worse* correct ones (e.g., shorter reasoning, fewer errors, more elegant proofs). What is GRPO sacrificing by discarding the value function? Is there a middle ground?

---

### Question 2: The Ghost in the Template

In Phuong & Hutter's framework, `DInference` produces a flat sequence $y \in V^*$. There is no formal notion of *structured output*. R1 produces a structured pair $(c, a)$ — but this structure lives in a strange place.

It is **not** in the architecture (the forward pass is unchanged).
It is **not** in a grammar or constraint decoder.
It is in the **reward function** ($r_{\text{fmt}}$ rewards valid tags) and the **template** (generation starts with `<think>`).

**Part (a):** If you were adding a `StructuredDInference` algorithm to Phuong & Hutter's paper, how would you formalize this? Where does the structural constraint formally live — in the input specification, the output specification, the hyperparameters, or somewhere else? Write the algorithm signature.

**Part (b):** The structure is *soft* — nothing prevents the model from generating malformed output during inference (and it sometimes does). Compare this to *hard* structural constraints like constrained decoding, where the logits are masked at each step to enforce a grammar. What are the tradeoffs? Consider: in R1-Zero (pure RL, no SFT), the model learned the `<think>/<answer>` format entirely from the format reward. Could a constrained decoder have achieved the same thing without $r_{\text{fmt}}$?

**Part (c):** This question generalizes beyond R1. More and more LLM systems use tool calls, function signatures, JSON schemas, and other structured outputs. In the Phuong & Hutter formalism, all of these are just tokens in $V^*$ — the framework has no way to express that some outputs are "valid" and some aren't. Is this a gap in the formalism, or is it the right level of abstraction?

---

## 6 · Summary: What's Actually New

Viewed through the lens of Phuong & Hutter, DeepSeek-R1's contribution is surprisingly *narrow* in terms of what it changes, and surprisingly *deep* in terms of what it discovers.

### What Didn't Change
- `DTransformer` (the forward pass) — identical
- `Attention`, `MHAttention`, `layer_norm` — identical
- The embedding, unembedding, positional encoding — identical
- The architecture is DeepSeek-V3, a 671B MoE model, but the R1 paper's contribution is orthogonal to the architecture

### What Changed

| Phuong & Hutter | DeepSeek-R1 | Why |
|---|---|---|
| `DTraining`: minimize cross-entropy on next-token prediction | `TrainR1Zero`/`TrainR1`: maximize GRPO objective on outcome rewards | Shift from imitation to incentivization |
| Loss is **dense** (per-token) | Reward is **sparse** (per-sequence) | Forces the model to discover its own intermediate steps |
| Single training phase | Four-stage pipeline (SFT → RL → SFT → RL) | Each stage addresses failure modes of the previous one |
| `DInference`: flat $y \in V^*$ | `R1Inference`: structured $(c, a) \in V^* \times V^*$ | Separates reasoning from final answer |
| Advantage via value network (PPO) | Advantage via group statistics (GRPO) | Eliminates the need for a second 671B model |

### The Big Takeaway

The transformer architecture — the thing we spent the semester formalizing — is a *fixed point* of this paper. What moves is everything *around* it: the training objective, the reward signal, the data pipeline, and the inference structure.

R1's claim is that if you hold the architecture fixed and change only the training signal from "predict the next token" to "get the right answer," a sufficiently large transformer will *spontaneously develop* chain-of-thought reasoning, self-verification, error correction, and dynamic compute allocation.

Whether you find this claim exciting or alarming probably depends on your priors about what "reasoning" means — and that's a conversation worth having.

---

## Appendix: Key Notation Crosswalk

For reference, how R1's notation maps to Phuong & Hutter:

| R1 Symbol | P&H Equivalent | Meaning |
|---|---|---|
| $\pi_\theta(o \mid q)$ | $P_\theta(x[t+1] \mid x[1:t])$ | Model's distribution, but over *full sequences* vs. *next tokens* |
| $q$ | $x$ (input) | The prompt / question |
| $o_i$ | $y$ (output) | A sampled output; R1 samples $G$ of them |
| $r_i$ | $-\text{loss}$ (loosely) | Scalar reward; replaces the dense per-token loss |
| $A_i$ | no equivalent | Group-relative advantage (new concept) |
| $\theta_{\text{old}}$ | no equivalent | Frozen copy for importance sampling |
| $\theta_{\text{ref}}$ | no equivalent | Slowly-updating reference for KL penalty |
| $\varepsilon$ (clip ratio) | no equivalent | GRPO hyperparameter |
| $\beta$ (KL coeff) | no equivalent | Regularization strength |
| $G$ (group size) | no equivalent | Number of rollouts per question |
| $\text{Template}(q)$ | no equivalent | Wraps prompt with `<think>`/`<answer>` instructions |

---

*End of presentation.*
