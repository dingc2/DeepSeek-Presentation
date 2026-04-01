# Teaching a Transformer to Think

### DeepSeek-R1 Through the Lens of *Formal Algorithms for Transformers*

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
│  Algorithm 13: DTransformer(x | θ)  →  P                │
│    The forward pass. Tokens in, probability matrix out. │
│    Unchanged by R1. Not one line is different.          │
├─────────────────────────────────────────────────────────┤
│  Algorithm 14: DTraining(x_{1:N}, θ)  →  θ̂              │
│    Minimize cross-entropy on next-token prediction.     │
│    ← THIS IS WHAT R1 REPLACES.                          │
├─────────────────────────────────────────────────────────┤
│  Algorithm 15: DInference(x, θ̂)  →  y                   │
│    Prompt → sample tokens autoregressively.             │
│    ← THIS IS WHAT R1 EXTENDS.                           │
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

$$
\begin{aligned}
& \textbf{DTraining}(x_{1:N_{\text{data}}},\; \theta) \to \hat{\theta} \\[6pt]
& \textbf{for } \text{epoch} = 1, 2, \ldots, N_{\text{epochs}}\textbf{:} \\
& \quad \textbf{for } n = 1, 2, \ldots, N_{\text{data}}\textbf{:} \\
& \qquad \ell \leftarrow \text{length}(x_n) \\
& \qquad P(\theta) \leftarrow \text{DTransformer}(x_n \mid \theta) \\
& \qquad \text{loss}(\theta) = -\sum_{t=1}^{\ell-1} \log P(\theta)\big[x_n[t{+}1],\; t\big] \quad \triangleright \text{ cross-entropy} \\
& \qquad \theta \leftarrow \theta - \eta \cdot \nabla\,\text{loss}(\theta) \quad \triangleright \text{ gradient descent} \\[6pt]
& \textbf{return } \hat{\theta} = \theta
\end{aligned}
$$

**DeepSeek-R1-Zero — Algorithm 5: `TrainR1Zero`**

$$
\begin{aligned}
& \textbf{TrainR1Zero}(\mathcal{Q},\; \theta_0) \to \hat{\theta} \\[6pt]
& \theta \leftarrow \theta_0 \;;\quad \theta_{\text{ref}} \leftarrow \theta_0 \\
& \textbf{for } s = 1, 2, \ldots, N_{\text{steps}}\textbf{:} \\
& \quad \text{sample mini-batch } \{q_b\} \text{ from } \mathcal{Q} \\
& \quad \theta_{\text{old}} \leftarrow \theta \\
& \quad \textbf{for each } q_b\textbf{:} \\
& \qquad \textbf{for } i = 1, \ldots, G\textbf{:} \quad \triangleright \text{ sample } G \text{ outputs per question} \\
& \qquad\quad o_i \sim \pi_{\theta_{\text{old}}}(\;\cdot \mid \text{Template}(q_b)\;) \\
& \qquad \textbf{for } i = 1, \ldots, G\textbf{:} \\
& \qquad\quad r_i \leftarrow \text{RuleReward}(q_b,\; o_i,\; a_b^{\ast}) \quad \triangleright \text{ did it get the right answer?} \\
& \qquad \{A_i\} \leftarrow \text{GRPOAdvantage}(\{r_i\}) \quad \triangleright \text{ normalize within group} \\
& \quad \mathcal{J} \leftarrow \tfrac{1}{B} \textstyle\sum_b \text{GRPOObjective}(q_b,\; \{o_i\},\; \{A_i\} \mid \theta,\; \theta_{\text{old}},\; \theta_{\text{ref}}) \\
& \quad \theta \leftarrow \theta + \eta \cdot \nabla_\theta \mathcal{J} \quad \triangleright \text{ gradient ascent (maximize)} \\
& \quad \textbf{if } s \bmod N_{\text{ref}} = 0\textbf{: } \theta_{\text{ref}} \leftarrow \theta \\[6pt]
& \textbf{return } \hat{\theta} = \theta
\end{aligned}
$$

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

#### Five things changed

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

$$
\begin{aligned}
& \textbf{GRPOAdvantage}(\{r_i\}_{i=1}^{G}) \to \{A_i\}_{i=1}^{G} \\[6pt]
& \mu \leftarrow \frac{1}{G} \sum_{i=1}^{G} r_i \quad \triangleright \text{ group mean} \\[4pt]
& \sigma \leftarrow \sqrt{\frac{1}{G} \sum_{i=1}^{G} (r_i - \mu)^2} \quad \triangleright \text{ group std} \\[4pt]
& \textbf{for } i = 1, 2, \ldots, G\textbf{:} \\
& \quad A_i \leftarrow \frac{r_i - \mu}{\sigma} \\[6pt]
& \textbf{return } \{A_i\}_{i=1}^{G}
\end{aligned}
$$

And the GRPO objective each output contributes to:

$$\mathcal{J}_i^{\text{GRPO}} = \min\!\Big(\rho_i \, A_i,\; \text{clip}(\rho_i,\; 1{-}\varepsilon,\; 1{+}\varepsilon)\, A_i\Big) - \beta \, \hat{D}_{\text{KL}}$$

where $\rho_i = \pi_\theta(o_i \mid q) \;/\; \pi_{\theta_{\text{old}}}(o_i \mid q)$ is the importance sampling ratio.

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

The output goes from $y \in V^{\ast}$ (a flat sequence) to $(c, a) \in V^{\ast} \times V^{\ast}$ (a structured pair). But this structure is **not architecturally enforced**. There is no change to the transformer's forward pass. The `<think>` and `<answer>` tags are just tokens in the vocabulary.

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

Look at `GRPOAdvantage`: when all $G$ outputs for a question get the **same** reward (all correct or all wrong), $\sigma = 0$ and $A_i$ is undefined — division by zero. The model learns nothing from that question.

**As the model improves and starts getting most questions right, this happens more and more often — what does that imply about GRPO's ability to keep improving, and how would you fix it?**

---

### Question 2: The Ghost in the Template

R1's `<think>/<answer>` structure is enforced by neither the architecture nor a grammar — only by a scalar reward $r_{\text{fmt}}$ and a prompt template. The formal framework (Phuong & Hutter) has no way to express that some output sequences are "structurally valid" and others aren't.

**If structured outputs (tool calls, JSON, chain-of-thought tags) are becoming central to how LLMs are used, is this a gap in the formalism that needs a new primitive, or is "just tokens in $V^{\ast}$" the right abstraction?**

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
| `DInference`: flat $y \in V^{\ast}$ | `R1Inference`: structured $(c, a) \in V^{\ast} \times V^{\ast}$ | Separates reasoning from final answer |
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
