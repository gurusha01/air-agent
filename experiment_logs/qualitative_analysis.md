# Qualitative Analysis: AIRA → LLM-Guided → SFT → GRPO

## Results Table

All scores are **mean over 5 runs at n=5** unless noted. Higher is better for all metrics. Baseline = MLGym default submission (no search).

| Task | Metric | Baseline | AIRA (MCTS n5) | LLM-Guided (n5) | SFT (v3, n5) | GRPO (projected) |
|------|--------|----------|----------------|------------------|--------------|-------------------|
| **Titanic** | Accuracy | 0.766 | 0.855 | 0.860 | **0.873** | *pending* |
| **Battle of Sexes** | Payoff | 1.023 | 1.138 | **1.296** | 1.371 (sq_battle_e3) | *pending* |
| **Regression (House Price)** | R² | 0.880 | 0.901 | 0.897 | 0.892 (sq_e1) | *pending* |
| **Mountain Car** | Reward | 33.79 | 59.31 | 50.70 | *not yet run* | *pending* |

### N=20 Scaling Results (best achieved)

| Task | Baseline (n20) | SFT sq_e1 (n20) | SFT sq_battle_e3 (n20) |
|------|----------------|------------------|------------------------|
| **Titanic** | 0.945 (mean 0.922) | **0.964** (mean 0.914) | -- |
| **Battle of Sexes** | 1.441 (1 run) | 1.433 (mean 1.362) | **1.444** (mean 1.414) |
| **Regression** | 0.888 (mean 0.886) | **0.909** (mean 0.892) | -- |

---

## 1. Where LLM-Guided Outperforms AIRA

### 1.1 The Core Problem: AIRA Cannot Diagnose Failures

AIRA uses UCT (Upper Confidence bounds for Trees) to select which node to expand:

```
UCT = Q_normalized + c * sqrt(ln(N_parent) / N_child)
```

This treats every experiment outcome as a scalar score. When an experiment **fails** — crashes, produces a bad score, or times out — MCTS reduces the value of that branch. But in ML experimentation, failure has two very different causes:

- **Hypothesis failure**: The approach is genuinely wrong (e.g., linear regression on a non-linear problem).
- **Implementation failure**: The approach is right but the code had a bug (e.g., KeyError from a typo in a column name).

AIRA cannot distinguish these. A promising branch (e.g., "try gradient boosting with feature engineering") gets abandoned after a single import error, while a less promising but bug-free branch (e.g., "logistic regression with default parameters") gets over-exploited because it produces valid scores.

**Concrete Example — Titanic:**
In AIRA MCTS runs, we observed that PyTorch-based nodes had ~70% crash rates and averaged 12+ debugging actions. sklearn-based nodes had ~10% crash rates and averaged 5 actions. MCTS naturally gravitates toward sklearn nodes because they produce valid scores, even when PyTorch approaches (once debugged) could achieve higher accuracy.

### 1.2 LLM-Guided Reads Logs and Reasons About "Why"

The LLM-Guided scientist replaces formula-based selection with a two-turn reasoning process:

1. **Inspection turn**: The scientist examines the full tree (all node scores, strategies, error messages) and selects nodes to analyze in detail.
2. **Decision turn**: The scientist diagnoses *why* each node succeeded or failed, builds hypotheses, and proposes the next experiment direction.

This enables several capabilities AIRA lacks:

**Failure diagnosis:** When a Random Forest node crashes with `KeyError: 'FamilySize'`, the scientist reads the error log and determines this is a missing feature engineering step (implementation failure), not a bad approach. It proposes the same strategy but with the feature engineering step included.

**Success generalization:** When log-transforming `GrLivArea` improves R² from 0.88 to 0.90, the scientist generalizes: "other right-skewed features like `LotArea` and `TotalBsmtSF` might also benefit from log transforms." AIRA just backpropagates the score without extracting the lesson.

**Counterfactual reasoning:** The scientist can reason: "Node 3 used one-hot encoding and hit a cardinality explosion. If I use target encoding instead, I keep the information without the dimensionality problem."

**Budget awareness:** The scientist adapts its strategy based on remaining budget: "With 10 nodes left, I should explore broadly. With 2 nodes left, I should refine the best approach found so far."

### 1.3 Where the Difference Is Largest: Battle of Sexes

Battle of Sexes is a repeated game theory task where the agent must reason about opponent modeling and adaptive strategy. This is where LLM-Guided most clearly outperforms AIRA:

- **AIRA MCTS (n5)**: mean 1.138 — often gets stuck on simple fixed strategies because MCTS cannot reason about opponent behavior.
- **LLM-Guided (n5)**: mean 1.296 — the scientist identifies "the opponent copies with ~80% probability" and designs adaptive strategies that exploit this pattern.

The qualitative difference: AIRA treats game theory as "try random strategies, score them." The LLM-Guided scientist treats it as "what is the opponent's policy? what strategy maximizes payoff against that policy?" This reasoning capability is exactly what formula-based selection cannot provide.

### 1.4 Where AIRA Still Competes: Regression

On House Price Regression, AIRA MCTS (n5) scores 0.901 vs LLM-Guided (n5) at 0.897. Regression is near its practical ceiling (~0.91 R²), so the search space is small and formula-based selection is adequate. There is little need for sophisticated failure diagnosis when most approaches produce reasonable scores.

---

## 2. Where SFT Outperforms LLM-Guided

### 2.1 The LLM-Guided Bottleneck: Cost and Scientist-Executor Disagreement

LLM-Guided requires an extra LLM call per node (the scientist reasoning step). Beyond cost, this creates a **scientist-executor disagreement problem**: when the scientist gives code-level instructions ("use sed to modify line 42"), the executor tries to follow literally and often fails. When the scientist gives vague instructions ("try something better"), the executor has no direction.

The optimal instruction level is **high-level strategy**: "add feature engineering for skewed features" — specific enough to be actionable, abstract enough to let the executor implement it its own way. But finding this level consistently is hard for a general-purpose LLM.

### 2.2 SFT Bakes Reasoning Into Weights

SFT (Supervised Fine-Tuning) distills the scientist's reasoning patterns directly into the executor's weights. Instead of two separate models negotiating at runtime, a single fine-tuned model makes better initial proposals without needing external guidance.

**The v3 breakthrough — task grounding:**

| SFT Version | Data | Result |
|-------------|------|--------|
| v1 (template QA) | 8,170 template-generated pairs | Loss 2.16→0.07 but **zero** downstream improvement. Memorized format, not reasoning. |
| v2 (Claude reasoning) | 704 rich traces from Claude | Marginal improvement (+0.009 R² on regression). Model applied Titanic-specific patterns to game theory tasks. |
| v3 (grounded) | 2,378 focused QA + 788 deep traces, **each including the MLGym task description** | **First statistically significant improvement: +0.046 accuracy on Titanic (p=0.032)** |

The critical insight: including the task YAML description in every training sample forces the model to **condition on actual task properties** rather than memorizing task-specific patterns.

### 2.3 Concrete Example: Titanic

**LLM-Guided approach (at runtime):**
1. Scientist analyzes tree → proposes "try ensemble with feature engineering" (1 expensive LLM call)
2. Executor attempts to implement → may misinterpret or over-complicate
3. Result: mean accuracy 0.860

**SFT approach (v3 sq_e1):**
1. Fine-tuned model directly proposes better initial approaches informed by training
2. No runtime scientist overhead
3. Produces structured feature engineering + model selection from the start
4. Result: mean accuracy **0.873** (p=0.032 vs baseline)

The +0.013 accuracy gain over LLM-Guided comes not from better reasoning per se, but from **eliminating the communication bottleneck** between scientist and executor.

### 2.4 Where SFT Wins Most: In-Domain Tasks

Per-task SFT models significantly outperform multi-task ones:
- Per-task average delta: **+0.016** over baseline
- Multi-task average delta: **-0.018 to -0.028** (task interference)

The lesson: SFT's advantage is strongest when the model is trained on data from the **same task family**. It learns task-specific patterns (e.g., "for tabular data, try gradient boosting with feature engineering") that transfer within a domain but not across domains.

### 2.5 Where SFT Struggles: Novel Tasks

When SFT encounters a task outside its training distribution, it can **hallucinate domain-specific features**. In v2, the model applied Titanic-specific reasoning ("survival rate by Pclass") to a game theory problem. V3's task grounding mitigates this but doesn't fully solve it — the model still struggles on truly novel task types.

---

## 3. Where GRPO Could Outperform SFT

*Note: GRPO training is in progress. These are projections based on the reward design and early experiments.*

### 3.1 SFT's Fundamental Limitation: No Budget Optimization

SFT teaches the model **what** to reason about (feature engineering, model selection, opponent modeling) but not **how to allocate its experimental budget**. Given 12 nodes to explore, the SFT scientist does not learn:

- Which hypotheses are worth testing first?
- When to stop validating a hypothesis and move on?
- How to balance exploring new directions vs. deepening existing ones?
- When a hypothesis is already resolved and further testing is wasteful?

SFT imitates demonstrations. GRPO optimizes a **policy**.

### 3.2 The Three-Component Reward Design

GRPO uses three reward signals that address SFT's blind spots:

**R_explore (weight 0.3) — Value of Information:**
Rewards experiments that *resolve uncertainty*, not just score well. An experiment that definitively shows "RNNs don't work on this task" gets exploration reward because it prunes the search space, even though the score is bad. SFT would never learn this — demonstrations only contain successful strategies.

**R_exploit (weight 0.5) — Genuine Improvement:**
Rewards actual score improvement, but **gated by execution quality**. Crashed runs that happen to score well (via fallback baseline submissions) get zero reward. This prevents the policy from learning to exploit artifacts in the scoring system. SFT has no such gating — it treats all scores equally.

**R_memory (weight 0.2) — Belief Quality:**
Rewards the scientist for maintaining accurate, evolving beliefs. If the scientist predicts "this experiment will improve score by 0.02-0.04" and the result is +0.029, that's a strong R_memory signal. If predictions are always vague ("something will happen"), R_memory is low. SFT doesn't train for prediction calibration at all.

### 3.3 Projected Advantage: Titanic

Consider a 12-node budget on Titanic:

**SFT behavior (observed):** The model proposes good initial strategies but doesn't adapt well mid-search. If feature engineering works, it keeps doing more feature engineering even when returns diminish. If a node crashes, it doesn't learn from the crash pattern.

**GRPO behavior (projected):** The trained policy would:
1. **Allocate early nodes to information gathering** — try 3 diverse approaches (sklearn, gradient boosting, neural net) to map the landscape.
2. **Prune low-information branches** — if linear models consistently score 0.82, stop exploring them.
3. **Deepen promising branches strategically** — allocate remaining budget to the highest-ceiling approach.
4. **Predict outcomes before executing** — "I expect gradient boosting + target encoding to score 0.88-0.91" → validates the hypothesis and updates beliefs.

### 3.4 Where GRPO Should Help Most: Mountain Car

Mountain Car is a reinforcement learning task with high variance and large headroom (baseline 33.79, ceiling ~200). SFT cannot be trained on this task (no prior successful demonstrations). GRPO can learn from trial-and-error experience:

- RL environments have long feedback loops (train for 1000 epochs, then score)
- The reward landscape is highly non-convex — small changes to hyperparameters cause large score swings
- Budget allocation matters enormously: spending 8 nodes debugging a crashed PyTorch RL implementation wastes budget that could explore JAX-based approaches

GRPO's R_explore reward specifically addresses this: it values experiments that reveal **which framework/algorithm works for this environment**, not just which one scores highest.

### 3.5 Where GRPO Faces Challenges

**Format compliance:** Early multi-task GRPO experiments showed that the model struggles with output format. Of 14 training steps, only 3 produced reward > 0 — in most cases, the model's output couldn't be parsed as valid scientist reasoning. This is a bootstrap problem: GRPO needs correctly formatted outputs to compute rewards, but the model hasn't yet learned the format.

**Solution:** Per-task GRPO starting from SFT checkpoints (currently queued). The SFT model already knows the format, so GRPO can focus on optimizing *content* rather than learning *structure*.

---

## 4. Summary: The Progression of Capabilities

| Capability | AIRA | LLM-Guided | SFT | GRPO |
|-----------|------|-----------|-----|------|
| Failure diagnosis | Score-only (can't distinguish bad idea from bug) | Reads logs, reasons about *why* | Learned patterns from demonstrations | Learned from reward signals |
| Budget allocation | UCT formula (fixed explore/exploit ratio) | Adaptive based on tree state | None (imitates demonstrations) | **Optimized via reward** |
| Strategy quality | Operator templates (Draft/Debug/Improve) | Rich reasoning with counterfactuals | Task-grounded reasoning in weights | Reward-optimized reasoning |
| Cost at inference | O(1) formula per node | O(expensive LLM call) per node | **O(0)** — baked into weights | **O(0)** — baked into weights |
| Generalization | Works on any task (but poorly) | Works on any task (expensive) | In-domain only | In-domain + learned exploration |
| Memory / beliefs | Score backpropagation | Natural language memory (5 entries) | Implicit in weights | Explicit belief state with VoI |

### The Insight Chain

1. **AIRA → LLM-Guided:** Formulas can't reason about failure modes. **Add an LLM that reads logs and diagnoses why.**
2. **LLM-Guided → SFT:** Runtime reasoning is expensive and creates communication bottlenecks. **Distill knowledge into weights.**
3. **SFT → GRPO:** Imitation learning doesn't teach budget allocation or information-seeking behavior. **Optimize with reinforcement learning rewards that value exploration, exploitation, and belief quality.**

Each step addresses the fundamental limitation of the previous one, building a progressively more capable system for automated ML experimentation.
