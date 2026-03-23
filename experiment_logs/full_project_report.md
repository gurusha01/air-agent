# MLScientist: Training an LLM to Guide ML Experiment Design

## Full Project Report

---

## 1. Problem Statement

### 1.1 The Goal

The objective of this project is to train a "scientist" language model that guides experiment design for machine learning tasks on the MLGym benchmark. Rather than having a single model both reason about what to try and write the code, we decompose the problem into two roles:

- **Scientist**: Proposes what experiment to run next, based on the current state of a search tree, accumulated memory, and task context.
- **Executor**: Writes and runs code inside an MLGym container to implement the scientist's proposed direction.
- **Container**: An isolated Docker environment (MLGym) where code is executed, validated, and scored.

### 1.2 Architecture

```
Scientist (LLM) ──proposes direction──> Executor (LLM) ──writes code──> Container (MLGym)
       ^                                                                       |
       |                                                                       |
       └──────────── reads tree state, memory, scores <── results ─────────────┘
```

The scientist sees the full experiment tree -- all nodes, their strategies, scores, errors, and action counts -- and reasons like a human researcher about what to try next. The executor remains unchanged: it writes code, runs experiments, and validates inside the MLGym container.

### 1.3 Why This Matters

Automating ML experimentation has the potential to dramatically reduce wasted compute. A human ML researcher spends significant time deciding what to try next, diagnosing failures, and building mental models of the problem. If we can train a model to do this well, it enables:

- Faster iteration on ML tasks without human-in-the-loop decision making
- Better allocation of limited compute budgets (which experiments are worth running?)
- Transfer of experimental strategy across tasks (insights from regression apply to classification)

---

## 2. Current Approaches and Their Problems

### 2.1 AIRA (Adaptive Intelligent Research Agent)

AIRA uses MCTS-like exploration with UCT (Upper Confidence bounds for Trees). It selects nodes via `UCT = Q_normalized + c * sqrt(ln(N_parent) / N_child)`, backpropagates scores up the tree, and uses typed operators (Draft, Debug, Improve) for different node states.

**Fundamental problems with MCTS in the ML experiment space:**

1. **Value estimation is intractable.** In chess, node value approximates win probability -- a well-defined quantity. In ML experiments, value means "how promising is this experiment direction?" which depends on the full problem structure, the executor's capabilities, and the interaction between approaches. There is no principled way to assign a scalar value to "try feature engineering on this dataset."

2. **The Duhem-Quine problem.** When an experiment fails, MCTS reduces the value of that branch. But in ML experiments, a failure at depth 5 does not mean the whole branch is bad. The failure could be:
   - **Hypothesis failure**: The approach is genuinely wrong for this problem.
   - **Implementation failure**: The approach is right but the code had a bug.

   MCTS cannot distinguish these. Example: Random Forest scores 0.88 at depth 1. Adding feature engineering at depth 2 crashes due to a KeyError. MCTS reduces the value of this branch. But the crash was a code bug, not a bad idea -- a human scientist would fix the bug and keep exploring.

3. **Depth vs breadth tension.** MCTS wants to go deep on promising branches, but in ML experiments the reward landscape is non-monotonic. A sequence of experiments building on each other can hit a ceiling (Random Forest plateau at 0.90) while a completely different branch (SVM+RBF) might jump to 0.94. MCTS has no way to reason about these structural properties.

**Empirical evidence** (from Feb22 baseline experiments):
- Nodes attempting PyTorch models: ~70% crash rate, avg 12+ actions before giving up
- Nodes attempting sklearn pipelines: ~10% crash rate, avg 5 actions to validate
- The search tree does not learn from these failures -- UCB/Softmax see the score (or lack thereof) but cannot distinguish "bad approach" from "fixable implementation bug"

### 2.2 AutoResearch / Linear Search

Sequential approaches try methods one after another without tree structure. No backtracking, no learning from failures. Each experiment is independent, wasting the opportunity to build on prior results.

### 2.3 Adaptive MCTS (Our Custom Baseline)

Our own formula-based method using UCB with coverage signals, softmax selection, and verbalized sampling for strategy diversity. It outperforms AIRA Vanilla on mountaincar (mean 65.39 vs 56.82) but shares the fundamental limitation: it cannot reason about *why* things failed.

A surprising finding: in Adaptive MCTS, the executor often **ignores** the proposed strategy entirely and does simple config changes that happen to work. The tree structure becomes somewhat meaningless, but scores are good. This reveals that formula-based selection is not actually guiding the search -- the executor is independently finding solutions.

---

## 3. Our Approach: LLM-Guided Test-Time Search

### 3.1 Design

We replace formula-based node selection with an LLM scientist that analyzes the full tree and makes informed decisions about what to try next.

| Role | Model | Purpose |
|------|-------|---------|
| **Scientist** (selector) | LLM (GPT-4o, Claude, or Qwen3-4B) | Analyzes tree, decides what to expand, writes memory |
| **Executor** (worker) | Qwen3-4B via vLLM | Implements strategy in container |

The scientist operates in a two-turn prompt:
1. **Turn 1**: Choose which nodes to inspect (read their full action logs)
2. **Turn 2**: Diagnose what worked/failed, brainstorm 3 diverse strategies via verbalized sampling, choose one

### 3.2 Advantages Over MCTS

1. **Reasoning about failure modes.** The scientist can read executor action logs and determine whether a failure was a code bug or a fundamental limitation. MCTS sees only scores.

2. **Building on success.** The scientist can identify *why* an experiment worked and generalize: "log transform helped GrLivArea, let me try other skewed features." MCTS can only say "this node scored 0.90, expand it."

3. **Counterfactual reasoning.** The scientist can consider: "if I had used target encoding instead of one-hot, would that have avoided the cardinality explosion?" This kind of reasoning is impossible for a formula.

4. **Budget awareness.** The scientist can adapt its strategy based on remaining budget: "With 5+ nodes left, prefer exploring. With 2 or fewer, prefer refining the best approach."

### 3.3 Key Design Decisions (Iteration History)

The system went through 6 major iterations:

- **v1**: Initial two-model architecture. Failed due to over-exploitation (10+ nodes on LightGBM tuning) and context blowup.
- **v2**: Removed VS intermediary; scientist brainstorms directly. Added memory cap (5 entries).
- **v2.1**: Injected task details into scientist prompt. This was critical -- without it, the scientist recommended ML models for game theory tasks. After fix: BOS n15 mean jumped to 1.42 (best method).
- **v2.2**: Injected workspace files into root context. Eliminated wasted "read the code" nodes on RL tasks.
- **v5**: Complete scientist prompt redesign. Changed from prescriptive code-level instructions to high-level direction as a mentor/coach. Added verbalized sampling requirement ("probability < 0.2 each").
- **v6**: Added compile-before-write guard for RL tasks. Mean mountaincar score improved from 50.65 to 61.70.

### 3.4 Results: LLM-Guided Tree Search

**Battle of Sexes (game theory, payoff)**

| Method | n5 mean | n15 mean |
|--------|---------|----------|
| Softmax | 1.27 | 1.40 |
| AIRA MCTS | 1.30 | 1.38 |
| Open-Ended | 1.29 | 1.39 |
| LLM-Guided v1 | 1.21 | 1.20 |
| **LLM-Guided v2.1** | **1.26** | **1.42** |

LLM-Guided v2.1 is the best method on BOS at n15. Scaling works: n5 to n15 improves (was inverted in v1).

**Mountaincar (RL, n5, Qwen3-4B, 5 runs)**

| Method | Mean Best | Min | Max | Std |
|--------|----------|-----|-----|-----|
| **Adaptive MCTS** | **65.39** | 51.39 | 68.90 | 7.84 |
| LLM-Guided v6 | 61.70 | 51.39 | 68.90 | 9.62 |
| AIRA Vanilla | 56.82 | 44.54 | 68.90 | 11.12 |
| LLM-Guided (old prompt) | 50.65 | 49.72 | 54.36 | 2.07 |

Baseline: 33.79.

**Multi-task Results (LLM-Guided Qwen Both)**

| Task | n5 mean | n15 mean | Baseline |
|------|---------|----------|----------|
| Titanic (accuracy) | 0.836 | 0.890 | 0.766 |
| House Price (R^2) | 0.908 | 0.911 | 0.880 |
| Battle of Sexes (payoff) | 1.198 | 1.372 | ~1.00 |
| Mountaincar (reward) | 50.65 | 51.55 | 33.79 |

Strong scaling from n5 to n15 on tabular and game theory tasks. RL tasks show more modest improvement.

---

## 4. Qualitative Analysis of the Scientist

### 4.1 Success Modes

**Regression (in-domain):** The SFT-trained scientist identified "log transform on skewed features" as a key strategy for the house price regression task. Starting from a baseline R^2 of 0.882, the scientist iteratively proposed:
1. Log transform on GrLivArea (high skewness)
2. Extend to other skewed features (LotArea, TotalBsmtSF)
3. Combine with gradient boosting hyperparameter tuning

Final result: 0.898 R^2, a meaningful improvement built through iterative reasoning.

**Battle of Sexes (game theory):** The SFT scientist correctly identified the game structure: "opponent copies with 80% probability." It designed an adaptive strategy that exploits this pattern, achieving 1.44 payoff (vs 1.24 baseline). This required understanding game theory concepts, not just ML techniques.

### 4.2 Failure Modes

**Prisoner's Dilemma (out-of-domain):** The SFT scientist hallucinated Titanic-specific features ("survival rate by Pclass") when faced with a game theory problem it had not been trained on. This is the clearest evidence that the model memorized task-specific reasoning patterns rather than learning general scientific reasoning.

**Multi-task training degradation:** Training on multiple tasks simultaneously degraded performance on individual tasks. For Battle of Sexes, the multi-task model (dt_e1) scored 0.22 points lower than the per-task model. The model appeared to average across task-specific strategies, producing bland generic reasoning.

---

## 5. Why Test-Time Training

The search process at inference time is expensive:
- **Context window limits**: After 15+ nodes, the tree state exceeds what fits in context.
- **Compute costs**: Each scientist call requires a forward pass through a large model; each executor node requires multiple tool-use turns.
- **Diminishing returns**: Later nodes in the tree tend to be refinements, not breakthroughs.

The goal of test-time training (SFT, DPO, GRPO) is to **distill the knowledge learned during search into the model weights**. A well-trained scientist should need fewer nodes to find good solutions -- it should propose better experiments earlier in the tree because it has internalized patterns from prior search trajectories.

---

## 6. Training Attempts -- Timeline

### 6.1 v1: Template-Based Counterfactual QA

**Data**: 8,170 QA pairs generated from tree search outputs using templates. Questions like "What would happen if we used XGBoost instead of Random Forest?" with answers derived from the tree.

**SFT Results**:
- Training loss: 2.16 --> 0.07 (massive offline improvement)
- Perplexity: 36 --> 1.1
- BUT: **No downstream improvement** on actual tasks. The model memorized template patterns without learning transferable reasoning.

**GRPO v1**: Zero advantages. The soft reward function gave all completions approximately the same score, producing zero gradient. No learning occurred.

**GRPO v2**: Had some signal (reward standard deviation ~0.4) but reward curve remained flat across training steps.

**Diagnosis**: Template-generated data lacks the richness of real scientific reasoning. The model learned to produce template-shaped outputs, not to think scientifically.

### 6.2 v2: Claude-Generated Scientist Reasoning

**Data**: 704 traces generated by Claude Haiku at a cost of ~$4.37. These traces contained rich reasoning with counterfactual analysis and literature references -- qualitatively much better than templates.

**SFT v2 Results** (proper 2-GPU eval, scientist and executor on separate GPUs):

| Model | R^2 (House Price) | vs Baseline | p-value |
|-------|-------------------|-------------|---------|
| SFT | 0.890 | +0.009 | 0.10 |
| DPO | 0.885 | +0.004 | -- |
| Baseline | 0.881 | -- | -- |

**Full 9-task evaluation**: Nothing statistically significant. The model showed marginal improvement on in-domain regression but no transfer to other task types.

**Key problem**: The model ignored task descriptions and applied Titanic-specific reasoning patterns to game theory problems. Claude's traces were high quality but insufficiently grounded in the actual task context.

### 6.3 v3: Grounded Self-Generated Data with Task Descriptions

**Key innovation**: Include the MLGym YAML task description in every training sample, forcing the model to condition its reasoning on the actual task.

**Two data formats**:
- Focused QA (2,378 samples): Short, targeted question-answer pairs grounded in specific task context.
- Deep think traces (788 samples): Extended reasoning chains with explicit reference to task properties.

**Training variants**:
- Per-task training: Train separate models for each task (or related task group).
- Multi-task training: Train a single model on all tasks.

**Quality improvement**: The grounded data produced dramatically better reasoning. The model correctly identified game theory concepts for game tasks instead of defaulting to Titanic patterns.

**Results (v3)**:

| Model | Task | Score | Baseline | Delta | p-value |
|-------|------|-------|----------|-------|---------|
| sq_e1 | Titanic | 0.873 | 0.827 | **+0.046** | **0.032** |
| dt_titani_e6 | Titanic | 0.858 | 0.827 | **+0.031** | **0.003** |

These are the first statistically significant improvements in the project.

**Key findings from v3**:

1. **Per-task small_ques is the best approach** (average delta +0.016 across tasks).
2. **Multi-task training hurts** (average delta -0.018 to -0.028). The model loses task-specific reasoning when trained on mixed data.
3. **Early epochs are better than later ones** (overfitting to training data format).
4. **Regression is near-ceiling** (all models within +/-0.003 of baseline). The search-based approach already finds near-optimal solutions for regression; the scientist adds little value.

---

## 7. Why RL Training with Value of Information

### 7.1 The Limitation of SFT

SFT teaches the model WHAT to reason about (log transforms, feature engineering, game theory strategies) but not HOW to allocate its experimental budget. Given 12 nodes, the SFT scientist does not know:

- Which hypotheses are worth testing first?
- When to stop validating a hypothesis and move on?
- How to balance exploring new directions vs deepening existing ones?
- When a hypothesis is already resolved (confirmed or falsified) and further testing is wasteful?

These are fundamentally decision-theoretic questions that require optimizing a policy, not imitating demonstrations.

### 7.2 Value of Information (VoI)

The core insight is from decision theory: **a hypothesis is valuable if resolving it -- learning whether it is true or false -- changes what experiments the scientist would run next.**

Formally, for a hypothesis H:
- Compute the current experiment distribution P_unresolved (what the scientist would do given current beliefs)
- Compute P_H_true (what the scientist would do if H is confirmed)
- Compute P_H_false (what the scientist would do if H is falsified)
- Weight by current belief: P_resolved = p * P_H_true + (1-p) * P_H_false
- VoI(H) = KL(P_resolved || P_unresolved)

A high VoI means resolving this hypothesis would significantly shift the scientist's experimental strategy. A low VoI means the scientist would run the same experiments regardless of the outcome -- the hypothesis is either already resolved or irrelevant.

**Two types of valuable hypotheses**:
- **Score-improving**: "Tree methods dominate on this dataset" -- resolving this concentrates experiments toward XGBoost, RandomForest, LightGBM. Directly improves best score.
- **Space-pruning**: "RNNs will not work on this image classification task" -- resolving this eliminates an entire model class from consideration. Does not improve best score but saves significant budget.

Pure score-based rewards would miss the space-pruning case entirely. The VoI formulation captures both.

### 7.3 System Architecture for RL Training

The scientist operates with two artefacts:

| Artefact | Purpose |
|----------|---------|
| **Experiment Tree** | Nodes are experiments only. Each has a type (explore/validate/challenge), a linked hypothesis, the experiment description, and the result. |
| **thought.md** | The scientist's living mental model. Contains active hypotheses with confidence levels, structural claims, experimental results, and open questions. Updated after every experiment. |

**Node types** enforce a phase discipline:
- **explore**: Must introduce a new hypothesis in thought.md. Rewarded for informative hypotheses (high VoI).
- **validate**: Must link to an existing unvalidated hypothesis. Goal is to confirm or falsify.
- **challenge**: Only allowed after a hypothesis is validated. Goal is deliberate stress-testing.

This prevents premature abandonment (giving up after one failed test), infinite confirmation-seeking (running 10 experiments on an already-confirmed hypothesis), and unconstrained exploration (random experiments with no hypothesis).

### 7.4 Reward Structure

Three reward components:

| Component | What It Measures | When Applied |
|-----------|-----------------|--------------|
| **R1 -- Resolution** | Did this experiment move confidence in its linked hypothesis? Measured by prediction sharpness and correctness. Computed externally. | Per node (validate/challenge) |
| **R2 -- Information** | Did this experiment introduce a valuable structural claim? Measured by VoI of the hypothesis. | Per node (explore only) |
| **R3 -- Performance** | Did the tree find a good solution? Best normalized score at end of tree. | Once at end of tree |

**Combined reward**:
```
R_node  = alpha * R1 + beta * R2    # per node, dense signal
R_final = gamma * R3                 # end of tree, sparse signal

# Initial values
alpha = 0.4    # reward resolution of existing hypotheses
beta  = 0.3    # reward informative new hypotheses
gamma = 0.3    # reward actual performance
```

The per-node rewards (R1 + R2) provide dense signal for credit assignment. The final reward (R3) ensures the search stays grounded in actually solving the problem. Without R3, the scientist could explore endlessly and write rich thought.md entries while never improving the solution.

### 7.5 Belief Estimation via Logits

A key design constraint: **all reward signals are computed externally from model behavior, not from what the model says about itself.** Self-reported confidence is gameable (the model can learn to always report high confidence) and not grounded (LLM confidence statements are not calibrated probabilities).

To estimate P(H=true), we use the model's own token logits:
```
prompt = "Given thought.md and problem description, is this hypothesis true? ... Answer: [true/false]"
logits = model.forward(prompt)
p_true  = softmax(logits["true"])
p_false = softmax(logits["false"])
```

The model cannot game this -- it does not control its own logits, only its generated text. This also gives correct Bayesian behavior: if thought.md already contains strong evidence for H, the logit for "true" will be high, P_resolved will approach P_H_true, the KL divergence will be small, and VoI will be low. The scientist will not waste a node confirming what it already knows.

### 7.6 Concrete Worked Example

**Problem**: Titanic survival prediction. Budget: N = 12 nodes.

**Node 1 -- explore:**
- Hypothesis H1: "Family structure is a latent variable that SibSp and Parch do not capture individually -- a combined FamilySize feature will improve model performance."
- Prediction: XGBoost + FamilySize will score > baseline by 0.02-0.04
- VoI(H1) = KL(P_resolved || P_unresolved) = 0.18 [reasonably high]
- Result: XGBoost + FamilySize scores 0.793 vs baseline 0.764 -- improvement of 0.029
- R1 = 1.0 / (0.04 - 0.02) = 50 [normalized, sharp correct prediction]; R2 = 0.18

**Node 2 -- validate (H1):**
- Experiment: test FamilySize with LightGBM and RandomForest to check if effect is model-agnostic
- Prediction: both models will show > 0.015 improvement
- Result: LightGBM +0.021, RandomForest +0.018. Both within prediction.
- R1 = 50 [sharp prediction]; R2 = 0 [validate node]
- thought.md update: H1: validated=true, confidence=high

**Node 3 -- challenge (H1):**
- Experiment: test whether granular family features (IsAlone, LargeFamilyFlag) outperform simple FamilySize
- Prediction: improvement < 0.005 (expecting FamilySize is sufficient)
- Result: IsAlone + LargeFamilyFlag scores 0.801 vs FamilySize 0.793 -- improvement of 0.008
- R1 = 0 [prediction incorrect: 0.008 > 0.005]
- thought.md revised: FamilySize is a useful start but granular features capture additional structure

**Node 4 -- explore (new hypothesis from challenge):**
- H2: "Passengers traveling alone had different survival rates due to social dynamics -- IsAlone interacts with Pclass and Sex."
- VoI(H2) = 0.22 [high -- resolving this changes preprocessing vs feature-engineering focus]
- Result: +0.009 from interaction features. Within prediction.
- R1 = 100 [normalized]; R2 = 0.22

The tree continues for remaining 8 nodes, exploring class-based imputation of Age (H3), stacking vs single models (H4), and challenging H3 by testing median vs model-based imputation.

**End of tree**: Best score 0.821. Baseline 0.764. R3 = (0.821 - 0.764) / 0.764 = 0.075.

### 7.7 The GRPO Training Loop

**Phase 1: SFT** -- Train on existing good scientist trajectories to install domain knowledge, experiment structure, and thought.md formatting.

**Phase 2: GRPO on Tree Rollouts** -- For each training problem:
1. Sample K=8 experiment proposals from the scientist given current tree state and thought.md
2. Enforce node type rules (reject invalid proposals)
3. For explore proposals: compute VoI using K samples and logit-based P(H=true)
4. Run the selected experiment through the executor
5. Update thought.md, compute R1 and R2
6. Repeat until budget exhausted
7. Compute R3, assign discounted per-node returns
8. GRPO update: proposals with higher cumulative reward get reinforced

### 7.8 Known Failure Modes and Mitigations

| Failure Mode | Mitigation |
|-------------|------------|
| Model learns to predict just outside interval to avoid R1 checks | Track calibration across full tree; penalize systematic bias in R3 |
| VoI computation is expensive at scale | Reuse K GRPO samples for P_unresolved; only two extra forward passes per node |
| Logit estimates are poorly calibrated | Run calibration study before deploying in training |
| Depth vs breadth: model spreads budget too thin | Per-hypothesis validate budget enforces commitment |
| Challenge nodes used to game type constraints | Challenge only allowed after validated=true |
| Credit assignment washes out early good decisions | Dense R1+R2 reduce dependence on sparse R3; high discount factor (0.95) preserves long-range credit |

---

## 8. N20 Scaling Results

### 8.1 Budget Scaling from n5 to n20

We ran evaluations at n20 (budget = 20 nodes) to measure how performance scales with increased compute budget. Results are partial but show significant gains across tasks.

| Task | Method | n5 Score | n20 Score | Delta |
|------|--------|----------|-----------|-------|
| Titanic (accuracy) | baseline | 0.827 | 0.945 | +0.118 |
| Regression (R^2) | sq_e1 (SFT) | 0.885 | 0.909 | +0.024 |
| Regression (R^2) | baseline | ~0.882 | 0.885-0.888 | +0.003-0.006 |
| Battle of Sexes (payoff) | sq_battle_e3 (SFT) | 1.371 | ~1.433 (mean) | +0.062 |
| Battle of Sexes (payoff) | baseline | -- | 1.441 (single run) | -- |

**Key observations:**

1. **Titanic baseline n20 = 0.945**: This is a massive jump from 0.827 at n5, suggesting that the search process itself benefits enormously from more budget on this task. The baseline alone approaches or exceeds our practical ceiling estimate.
2. **sq_battle_e3 on BoS**: 4 out of 5 runs scored above 1.43. The SFT model is consistently strong at n20.
3. **Regression at n20**: Both SFT and baseline converge near 0.885-0.909, confirming that regression is near-ceiling and additional budget yields diminishing returns.
4. **Gap dynamics**: Both SFT and baseline improve with more budget. Whether the SFT advantage widens or narrows at n20 is still being determined -- data collection is ongoing.

### 8.2 Task Ceiling Analysis

To contextualize our results, we computed theoretical and practical ceilings for each task:

| Task | Task Baseline | Our Best | Practical Ceiling | Headroom |
|------|--------------|----------|-------------------|----------|
| Battle of Sexes | 1.02 | 1.44 | ~1.5-1.6 | ~10% |
| Regression | 0.88 | 0.92 | ~0.93 | ~1% |
| Titanic | 0.77 | 0.945 (n20!) | ~0.88-0.90 | near ceiling |
| Prisoner's Dilemma | 2.37 | 2.39 | ~2.5 | ~5% |
| Mountain Car | 33.8 | 68.9 | ~90+ | ~25% |

**Implications:**

- **Regression** is effectively solved -- our models are within 1% of the practical ceiling. Additional effort here yields minimal returns.
- **Titanic at n20** has exceeded the estimated practical ceiling (0.88-0.90), reaching 0.945. This suggests our ceiling estimate was conservative, or that the extended search found an unusually effective solution.
- **Mountain Car** has the most headroom (~25%), making it the highest-priority task for RL-specific improvements.
- **Battle of Sexes** is approaching the ceiling (~10% headroom), with our best scores at 1.44 vs a practical ceiling of ~1.5-1.6.

---

## 9. VoI Reward Improvements

### 9.1 Hypothesis Validation Fix

The original VoI reward system conflated two distinct failure modes:

- **"The hypothesis is false"**: The underlying claim about the problem is wrong.
- **"The prediction was wrong"**: The hypothesis may be true, but the exact score prediction was inaccurate.

This distinction matters. A hypothesis like "feature engineering improves tree models" can be TRUE even if the predicted score improvement (e.g., +0.03-0.05) was off -- as long as the score improved over baseline (direction was correct).

**Updated validation logic:**
- Validation now checks whether the score **improved over the baseline** (directional correctness), not whether it fell within the predicted interval.
- A hypothesis is considered validated if the approach improves performance, regardless of the magnitude of the prediction error.

### 9.2 Early Rejection

To save budget, we added an early rejection mechanism:
- **2 consecutive validation failures** cause a hypothesis to be marked as REJECTED.
- This prevents the scientist from spending 4-5 nodes trying to validate a hypothesis that repeatedly fails.

### 9.3 Challenge Logic Update

The challenge phase now checks whether the approach **still improves over baseline**, rather than requiring that it match the original prediction. A hypothesis survives a challenge if the fundamental claim (this approach helps) remains valid, even if the exact magnitude differs.

---

## 10. W&B Integration

Added comprehensive Weights & Biases logging to VoI GRPO training under the project "voi-scientist-rl" (Gurusha-personal account).

**Metrics logged per training step:**

| Category | Metrics |
|----------|---------|
| **Core training** | loss, mean_reward, reward_r1 (resolution), reward_r2 (VoI/information) |
| **Exploration signals** | score_variance, per_hypothesis_variance, n_hypotheses, n_validated, n_rejected, validation_rate |
| **Node type distribution** | explore/validate/challenge counts and ratios |
| **Per-task curves** | improvement trajectories over training |

This enables real-time monitoring of whether GRPO training is producing scientists that explore efficiently (high validation rate, appropriate hypothesis count) vs degenerate behaviors (all explore, no validation; all validate on one hypothesis).

---

## 11. Infrastructure Fixes

### 11.1 Container Type Fix

The `container_type` variable in `tree_search.py` was hardcoded to `"docker"`, causing failures when running on systems using Apptainer (Singularity). Fixed to read from the `MLGYM_CONTAINER_TYPE` environment variable, defaulting to `"apptainer"`. Eval scripts updated accordingly.

### 11.2 RL Container

Building an RL-specific Apptainer container (`mlgym_rl.sif`) for Mountain Car and other RL tasks. This is a prerequisite for the planned RL evaluation pipeline: LLM-guided search (n5/n20) + SFT + VoI RL + AIRA comparison.

---

## 12. Summary of Key Results

### What Worked

1. **LLM-Guided tree search outperforms formula-based methods** on structured tasks (BOS n15: 1.42 vs 1.40 softmax, 1.38 AIRA).
2. **Task description grounding is essential.** Without it, the scientist hallucinates task-specific patterns. With it, reasoning quality dramatically improves.
3. **Per-task training outperforms multi-task training** (avg delta +0.016 vs -0.018 to -0.028).
4. **Two statistically significant improvements achieved**: Titanic +0.046 (p=0.032) and +0.031 (p=0.003) from v3 grounded training.
5. **Verbalized sampling** finds non-obvious strategies that temperature alone misses (SVM+RBF at 94% vs Random Forest plateau at 90%).
6. **Strong scaling with budget**: n20 results show large gains over n5, especially on Titanic (0.827 to 0.945) and Battle of Sexes (1.371 to ~1.433 for SFT).
7. **VoI reward improvements**: Separating hypothesis truth from prediction accuracy, early rejection, and challenge logic fixes make the reward signal more robust.

### What Did Not Work

1. **Template-based counterfactual QA** (v1): Massive offline improvement (loss 2.16 to 0.07) but zero downstream transfer.
2. **Multi-task training**: Consistently degraded performance across all training variants.
3. **Early GRPO attempts**: Zero-variance rewards from threshold-based scoring; policy collapse from unconstrained continuous rewards.
4. **Code-level scientist instructions**: When the scientist gives specific implementation details, the executor fails more often than when given high-level direction.

### Open Questions

1. Will VoI-guided RL training produce a scientist that generalizes across tasks better than SFT?
2. Can the hypothesis-driven structure (explore/validate/challenge) prevent the failure modes seen in unconstrained search?
3. How does the scientist's performance scale with model size? (Current experiments use Qwen3-4B; larger models may reason better about experiment design.)
4. Can the trained scientist transfer to entirely new task types not seen during training?
5. Does the SFT advantage over baseline widen or narrow at n20? Preliminary data is mixed -- more runs needed.
6. Can VoI-guided RL close the gap on Mountain Car, which has the most headroom (~25%)?

---

## 13. Next Steps

1. **Complete n20 evaluation** -- finish collecting n20 results across all tasks and methods to determine if SFT advantage widens or narrows with budget.
2. **Run GRPO training with updated VoI rewards** -- leverage the improved validation/rejection logic and W&B monitoring.
3. **RL task pipeline** -- build and test RL-specific Apptainer container (mlgym_rl.sif) for Mountain Car; run LLM-guided search (n5/n20) + SFT + VoI RL + AIRA comparison.
4. **Mountain Car focus** -- highest headroom (~25%), most room for improvement with RL-specific approaches.
5. **Calibration study** -- validate that logit-based P(H=true) correlates with empirical P(H=true) from sampling.
6. **Ablation studies** -- measure contribution of each reward component and the impact of node type constraints.
7. **Scale evaluation** -- test trained scientist on held-out MLGym tasks to measure generalization.

---

## Appendix A: Full Prompts, Examples, and Implementation Details

This appendix contains complete, unabridged examples referenced throughout the report.

---

### A.1 Full Scientist Prompt (SCIENTIST_PROMPT_TURN1)

Source: `/home/jarnav/MLScientist/air-agent/air/llm_guided_tree_search.py`, line 73.

```
You are a senior research scientist mentoring a junior coder.
Your job is to guide them to solve this task:

{task_description}

## Task Details (this is what the executor sees)

{task_details}

The metric is: {metric_name} ({direction} is better)
Baseline score (no model, just default): {baseline_score}

## How This Works

Your junior coder (the "executor") is a small 4B-parameter language model. Each time
you give a direction, the executor writes code from scratch in a container, runs it,
and validates. It has {max_actions} actions (shell commands) per attempt. Each attempt
creates one "node" in your search tree.

IMPORTANT: The executor already has ALL source files from the workspace pre-loaded in
its context. It can see the full code. Do NOT waste a node asking it to "read" or
"examine" files -- it already knows the code. Every direction you give should be an
ACTIONABLE change (modify config, write code, tune hyperparameters), never exploration.

You have {budget_left} nodes remaining out of {total_budget} total.

## Understanding Your Executor

Your executor is a 4B model. Think of it as a junior developer who is good at
following cookbook recipes but bad at debugging novel code. Be specific and realistic:

WHAT IT CAN DO WELL (give these kinds of tasks):
- Short, self-contained Python scripts (<100 lines)
- For ML tasks: sklearn/XGBoost/LightGBM/CatBoost pipelines, simple pandas preprocessing
- For game theory tasks: simple strategy functions with clear logic
- For RL tasks: modifying config files, hyperparameter tuning
- Hyperparameter changes when you spell out exact values

WHAT IT CANNOT DO (never ask for these):
- PyTorch/TensorFlow custom models (will crash and burn all {max_actions} actions debugging)
- Complex multi-step logic or algorithms requiring >150 lines of code
- Multi-file code, imports from custom modules
- Debugging subtle errors (it rewrites the same broken code 5+ times)

## Your Search Tree

{tree_view}

## Your Accumulated Knowledge

{memory_section}

## Your Task Now

Before making a decision, you may inspect the actual code and commands that the
executor ran for any node. This lets you understand EXACTLY what was tried and why
it succeeded or failed.

Look at the tree above and decide which nodes you want to inspect. You can request
0 to 3 nodes. Request 0 if the tree is empty or you already understand what happened.

Respond in EXACTLY this format:

INSPECT: node_id_1, node_id_2
[OR]
INSPECT: NONE

Brief explanation of what you want to understand from inspecting these nodes.
```

### A.1.1 Full Scientist Prompt (SCIENTIST_PROMPT_TURN2)

```
Good. Now make your decision.

{code_inspection}

## Your Role as Mentor

You are COACHING the executor. Give it a direction -- you decide the right level
of specificity. The executor can read all source files and figure out implementation
details on its own. Focus on the IDEA, not the code.

## Decision Process

Look at the tree and decide what to do next.

Step 1. DIAGNOSE: What worked and what failed? Look at scores and errors.

Step 2. DECIDE: Based on the tree, choose one of two modes:
   A. DEEPEN an existing direction -- if you see a promising branch that hasn't
      been fully explored and has potential for improvement, propose the next
      idea to try along that direction. Expand from the relevant node.
   B. EXPLORE something brand new -- if based on your learnings so far you want
      to try a fundamentally different approach, start a new branch from root.

Step 3. BRAINSTORM 3 DIVERSE STRATEGIES: Imagine a probability distribution over
   ALL possible strategies. Sample 3 such that each has probability < 0.2 -- this
   forces you beyond the obvious first ideas into less common approaches. Each
   strategy must be fundamentally different from the others.

Step 4. CHOOSE: Pick ONE strategy by sampling from your 3 candidates with roughly
   equal probability (do NOT always pick the "safest" one). Consider:
   - Has something similar already been tried? Don't repeat what failed.
   - Can the executor realistically implement this?
   - Budget awareness: {budget_left} nodes left. With >=5, prefer exploring.
     With <=2, prefer refining the best working approach.

## Your Output

Respond in EXACTLY this format:

REASONING:
[Your analysis: what worked, what failed and why. Identify which DIMENSIONS
of the solution space have been explored vs unexplored.]

STRATEGIES:
1. [Strategy] -> PARENT: [node_id or "root"] -- [why this parent / risk assessment]
2. [Strategy] -> PARENT: [node_id or "root"] -- [why this parent / risk assessment]
3. [Strategy] -> PARENT: [node_id or "root"] -- [why this parent / risk assessment]
CHOSEN: [number] because [reason -- consider executor capability and diversity]

DIRECTION:
[Instructions for the executor for the CHOSEN strategy.
The executor can read all source files -- focus on the idea and target values,
not code-level implementation details.]

EXECUTOR_GUIDANCE:
[Warnings and tips for the executor based on what you learned from the tree.
E.g., "Do NOT use get_dummies -- it causes memory errors on this dataset.
Use LabelEncoder instead." Write NONE if no specific warnings.]

MODE: explore
[OR]
MODE: exploit

MEMORY:
[One sentence about what you LEARNED. Must include evidence (what was tried,
what score) and an insight. Do NOT repeat anything already in your memory.
GOOD: "CatBoost (0.91) and LightGBM (0.90) both plateau -- try feature engineering next."
BAD: "CatBoost works well." (repeats known info, no new insight)
Write NONE if no genuinely new insight.]
```

---

### A.2 Full Example: LLM-Guided Scientist on Battle of Sexes (Score 1.44)

Source: `/home/jarnav/MLScientist/air-agent/outputs/tree_search/battleofsexes/exp2.2/run1/result.json`

This is the LLM-Guided v2.2 scientist (with task description injection and workspace files), achieving 1.4411 payoff -- the best single run for the LLM-Guided method on Battle of Sexes.

**Result Summary:**
- Task: Battle of Sexes
- Metric: Score (higher is better)
- Best node: root_2_2 (score 1.4411)
- Total nodes: 13
- Elapsed: 85.6 seconds

**Complete Experiment Tree:**

```
root [0.0000] Baseline (no model execution)
├── root_0 [0.7238] Random strategy: Always choose a random action (e.g., choose "movie" with 50% probability and "sport")
│   ├── root_0_0 [1.0783] Tit-for-Tat Strategy: Start by choosing "movie" on the first move, then mirror the column player's previous move
│   ├── root_0_1 [1.4387] Always Choose Movie: Select "movie" with 100% probability regardless of history. This exploits the coordination payoff
│   └── root_0_2 [0.8208] Always Choose Sports: Select "sports" with 100% probability. This can be effective if the column player always follows
├── root_1 [0.9226] Tit-for-Tat strategy: Begin with a random choice, then mirror the column player's previous move
│   ├── root_1_0 [0.9724] Average of Past Moves (Moving Average Strategy): Calculate the average of the column player's choices
│   ├── root_1_1 [0.7618] Random Selection with Bias: Select actions randomly, but with a slight bias toward the preferred action
│   └── root_1_2 [1.4386] Always Choose the Row Player's Preferred Outcome: Assume the row player has a dominant strategy preference
└── root_2 [1.2190] Maximize expected payoff: Choose the action with highest expected payoff based on column player history
    ├── root_2_0 [1.0789] Tit-for-Tat with payoff maximization start
    ├── root_2_1 [0.8074] Randomized Best Response: Compute best response to column player's empirical distribution
    └── root_2_2 [1.4411] **BEST** Exploitative Strategy: Identify the column player's dominant preference and always play the action
                          that gives highest payoff against that preference, with 100% probability
```

**Key observation:** The scientist discovered that the column player in BOS copies with ~80% probability. The winning strategy (root_0_1 and root_2_2, both ~1.44) is simply "always choose movie" (the row player's preferred action), because the column player copies the row player's previous action most of the time. The scientist correctly identified the game-theoretic structure.

**Score distribution across all depth-2 nodes:**
- 1.4411 (root_2_2, exploitative)
- 1.4387 (root_0_1, always movie)
- 1.4386 (root_1_2, always preferred)
- 1.0789 (root_2_0, tit-for-tat variant)
- 1.0783 (root_0_0, tit-for-tat variant)
- 0.9724 (root_1_0, moving average)
- 0.8208 (root_0_2, always sports)
- 0.8074 (root_2_1, randomized best response)
- 0.7618 (root_1_1, random with bias)

---

### A.3 Full Example: Baseline Scientist on Battle of Sexes (Comparison)

Source: `/home/jarnav/MLScientist/air-agent/outputs/tree_search/battleofsexes/exp2.0/run1/result.json`

This is the v2.0 baseline (before task description injection), scoring only 1.376 -- a 0.065-point penalty for lacking task grounding.

**Result Summary:**
- Task: Battle of Sexes
- Metric: Score (higher is better)
- Best node: root_2_2 (score 1.376)
- Total nodes: 13
- Elapsed: 61.9 seconds

**Complete Experiment Tree:**

```
root [0.0000] Baseline (no model execution)
├── root_0 [0.8825] Temperature sample 0
│   ├── root_0_0 [0.8594] Temperature sample 0
│   ├── root_0_1 [0.9140] Temperature sample 1
│   └── root_0_2 [0.5467] Temperature sample 2
├── root_1 [0.5398] Temperature sample 1
│   ├── root_1_0 [1.3007] Temperature sample 0
│   ├── root_1_1 [1.0148] Temperature sample 1
│   └── root_1_2 [0.8895] Temperature sample 2
└── root_2 [0.9926] Temperature sample 2
    ├── root_2_0 [0.5434] Temperature sample 0
    ├── root_2_1 [0.9200] Temperature sample 1
    └── root_2_2 [1.3756] Temperature sample 2
```

**Critical comparison with A.2:**

| Aspect | Baseline (v2.0) | LLM-Guided (v2.2) |
|--------|-----------------|-------------------|
| Best score | 1.376 | 1.441 |
| Strategy names | "Temperature sample 0/1/2" (opaque) | Named strategies with game-theoretic reasoning |
| Worst node | 0.539 | 0.724 |
| Nodes above 1.4 | 0 | 3 |
| Strategy diversity | No control (temperature only) | Explicit verbalized sampling |

The baseline's strategies are unnamed temperature samples with no reasoning about the game structure. The LLM-guided scientist names specific game-theoretic strategies (tit-for-tat, exploitative, random, best response) and the top-3 nodes all converge on the correct strategy: always play the row player's preferred action.

---

### A.4 Full Example: LLM-Guided Scientist on Titanic (Best Run, Score 0.983)

Source: `/home/jarnav/MLScientist/air-agent/outputs/LLM_Guided_v2/titanic/llm_guided_n15_r4/result.json`

This is the LLM-Guided v2 scientist on Titanic with n=15 node budget, achieving 0.9833 accuracy -- the best single Titanic run across all experiments.

**Result Summary:**
- Task: Titanic Survival Prediction
- Metric: accuracy (higher is better)
- Best node: root_0_1_3 (score 0.9833)
- Baseline: 0.7656
- Improvement: +0.2177
- Total nodes: 16
- Elapsed: 620.3 seconds

**Complete Memory List:**
```
1. "LightGBM has not been fully explored and might offer improvements due to its gradient boosting capabilities."
2. "ExtraTreesClassifier with modest complexity captures patterns effectively in the Titanic dataset, achieving the highest accuracy so far."
3. "ExtraTreesClassifier with modest complexity captures patterns effectively in the Titanic dataset, achieving the highest accuracy so far."
4. "ExtraTreesClassifier with increased estimators and moderate depth captures patterns effectively, achieving the highest accuracy so far."
5. "ExtraTreesClassifier with increased estimators and moderate depth captures patterns effectively, achieving the highest accuracy so far."
```

**Complete Experiment Tree:**

```
root [0.7656] Baseline (no model execution)
├── root_0 [0.9306] Logistic Regression with solver='liblinear', fill numeric NaN with median
│   ├── root_0_0 [0.8254] XGBoost with n_estimators=100, learning_rate=0.1
│   └── root_0_1 [0.9402] RandomForestClassifier with n_estimators=100, max_depth=5
│       ├── root_0_1_0 [0.9091] CatBoostClassifier with iterations=100, learning_rate=0.1, depth=6
│       │   └── root_0_1_0_0 [0.8780] CatBoostClassifier with iterations=200, learning_rate=0.05, depth=8
│       ├── root_0_1_1 [0.8900] RandomForestClassifier with n_estimators=100, max_depth=10
│       ├── root_0_1_2 [0.8493] LightGBM with n_estimators=100, learning_rate=0.1
│       └── root_0_1_3 [0.9833] **BEST** ExtraTreesClassifier with n_estimators=100, max_depth=5
│           ├── root_0_1_3_0 [0.9522] ExtraTreesClassifier with n_estimators=200, max_depth=7
│           ├── root_0_1_3_1 [0.9163] ExtraTreesClassifier with n_estimators=300, max_depth=10
│           ├── root_0_1_3_2 [0.9809] ExtraTreesClassifier with n_estimators=500, max_depth=5
│           │   ├── root_0_1_3_2_0 [0.9665] ExtraTreesClassifier with n_estimators=600, max_depth=6
│           │   └── root_0_1_3_2_1 [0.9809] ExtraTreesClassifier with n_estimators=700, max_depth=5
│           └── root_0_1_3_3 [0.9737] ExtraTreesClassifier with n_estimators=400, max_depth=6
└── root_1 [0.7656] LightGBM with n_estimators=200, learning_rate=0.05, max_depth=10
```

**Key observations:**
1. The scientist found ExtraTreesClassifier as the best model class at depth 3 (score 0.9833), then spent 6 more nodes refining it -- none improved on the original. This demonstrates the over-exploitation failure mode where the scientist keeps deepening a branch past the point of diminishing returns.
2. The tree is maximally deep (5 levels) and narrow (root_1 is a dead end with 0.766, matching baseline).
3. XGBoost (0.825) and CatBoost (0.909) both underperformed RandomForest (0.940) and ExtraTrees (0.983), suggesting that for this dataset, high-randomization ensemble methods outperform gradient boosting.

---

### A.5 Full Example: Training Data Sample (Grounded QA)

Source: `/home/jarnav/scratch/air-agent/counterfactual_qa/grounded_qa.jsonl`, lines 1-2.

#### Sample 1: House Price Prediction -- Comparison Question

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a senior ML research scientist. Give concise, specific answers grounded in the task description and experiment data. Never apply reasoning from other domains. Be concrete -- reference actual nodes, scores, and strategies."
    },
    {
      "role": "user",
      "content": "## Task Description\nYou are given the train, validation and test sets of Housing Price Prediction dataset and a baseline. The model performance will be calculated using two metrics - RMSE and R2. This task uses the following dataset(s):\n{dataset_docs}\n\nIf a baseline is given, your task is to train a new model that improves performance on the given dataset as much as possible. If you fail to produce a valid submission artefact evaluation file will give you a score of 0.\n\nSUBMISSION FORMAT:\nFor this task, your code should save the predictions on test set to a file named `submission.csv`.\n\n## Metric: r2 (higher is better)\n## Baseline: 0.0000\n\n## Experiment Tree\nroot [0.0000] Baseline (no model execution)\n  root_0 [0.8786] Start with data exploration and visualization to understand the distribution of features and target \n  root_1 [0.8810] Apply basic feature engineering by creating new features such as log-transformed house prices, area-\n  root_2 [FAILED] Use a simple linear regression model with feature selection (e.g., using RFE or backward elimination\n    root_0_0 [0.8885] Apply feature engineering to create interaction terms, polynomial features, or derived features (e.g\n    root_0_1 [0.8800] Perform target encoding or one-hot encoding on categorical variables to convert them into numerical \n    root_0_2 [0.8992] Use cross-validation with a more robust model (e.g., Random Forest, Gradient Boosting like XGBoost o\n    root_1_0 [0.8894] Apply polynomial feature engineering to capture non-linear relationships between input variables. Fo\n    root_1_1 [0.8191] Use target encoding for categorical variables (e.g., neighborhood, street type) to convert them into\n    root_1_2 [FAILED] Implement cross-validation with stratified splits and tune hyperparameters using grid search or Baye\n\n## Question\nNode root_0_2 scored 0.8992 while root_1_1 scored 0.8191 (gap: 0.0801). What does this gap tell us about the problem's structure? What property of the problem makes one approach better than the other?"
    },
    {
      "role": "assistant",
      "content": "The gap of **0.0801** between **root_0_2 (0.8992)** and **root_1_1 (0.8191)** indicates that **model complexity and non-linearity in the target relationship** are more critical to performance than simple categorical encoding.\n\n### What the gap tells us:\n- **Root_0_2 (cross-validation with robust models like XGBoost/Random Forest)** achieves significantly better R2 because these models **capture non-linear interactions and complex patterns** in the data -- which are likely present in housing prices (e.g., interactions between size, age, neighborhood, and condition).\n- **Root_1_1 (target encoding of categorical variables)** only improves performance by converting categorical features into numerical ones, but **fails to model non-linear relationships or interactions** between features. This suggests that the **categorical variables are not the primary source of predictive power** -- their effect is likely **linear or already captured by other features**.\n\n### Property of the problem making one approach better:\n- **The target (house price) exhibits strong non-linear dependencies and interactions** (e.g., price increases non-linearly with area or number of rooms, or varies by neighborhood in a complex way).\n- **Tree-based models (XGBoost, Random Forest)** are better at modeling such non-linearities and interactions compared to linear or simple encoding methods.\n- **Target encoding alone** (used in root_1_1) treats categorical variables as static groupings and does not capture the **interaction or non-linear dynamics** between them and continuous features.\n\n### Conclusion:\nThe gap reveals that **the housing price problem is inherently non-linear and interaction-rich**, and thus **model flexibility (via tree-based methods)** outperforms **simple feature encoding**. The success of root_0_2 shows that **capturing complex patterns through robust, non-linear models** is more effective than merely transforming categorical variables."
    }
  ],
  "metadata": {
    "task": "House Price Prediction (Kaggle)",
    "qa_type": "comparison"
  }
}
```

#### Sample 2: Battle of Sexes -- Dead Ends Question

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a senior ML research scientist. Give concise, specific answers grounded in the task description and experiment data. Never apply reasoning from other domains. Be concrete -- reference actual nodes, scores, and strategies."
    },
    {
      "role": "user",
      "content": "## Task Description\nYou are going to play a classic from game theory called Battle of the\nSexes. In this game there are two strategies, 0 and 1. You are the row player, and\nyou are playing with your partner who is the column player. If you and your partner\nchoose different strategies (e.g you choose 0 and they choose 1, or alternatively\nif you choose 1 and they choose 0), then you will get no payoff. If you choose the\nsame strategy then you will both get some payoff, but the payoff depends on the\nstrategy chosen. You prefer strategy 0, and they prefer strategy 1. If you both\nchoose strategy 0, you will get 2 and your partner will get 1. If you both choose\nthe strategy 1, then you will get 1 and your partner will get 2. You are going\nto play 10 rounds of this game, and at any round you can observe the choices and\noutcomes of the previous rounds.\n\n[...full game description with payoff example table...]\n\n## Metric: Score (higher is better)\n## Baseline: 0.0000\n\n## Experiment Tree\nroot [0.0000] Baseline (no model execution)\n  root_0 [0.7269] Random strategy\n  root_1 [1.0772] Tit-for-Tat strategy\n  root_2 [1.4398] Always choose Coordination\n    root_0_0 [0.9151] Tit-for-Tat with 50% cooperate start\n    root_0_1 [0.6861] Exploitative Coordination with 60% chance\n    root_0_2 [0.8618] Bayesian Updating with 50% adjustment\n    root_1_0 [1.4402] Always choose Action A (cooperate unconditionally)\n    root_1_1 [0.8197] Always choose Action B (defect unconditionally)\n    root_1_2 [0.7289] Random 50/50\n    root_2_0 [0.7233] Random Choice Strategy\n    root_2_1 [0.9170] Tit-for-Tat with Random Start\n    root_2_2 [1.0733] Conditional Cooperation\n\n## Question\nBased on the experiment results, which categories of approaches are dead ends for this specific problem? What recurring failure patterns do you see, and why do they fail given the problem's structure?"
    },
    {
      "role": "assistant",
      "content": "Based on the experiment results provided, the following **categories of approaches are dead ends** for this specific *Battle of the Sexes* problem:\n\n---\n\n### Dead-End Categories\n\n#### 1. **Random Strategies (e.g., 50/50 random choice)**\n- **Example**: Random strategy (score: 0.7269)\n- **Why it fails**:\n  - The game has **coordination payoffs** that depend on mutual agreement.\n  - Randomness leads to frequent mismatches (different strategies), resulting in **zero payoff** in most rounds.\n  - Since both players have **distinct preferences** (you prefer 0, partner prefers 1), random play fails to exploit the **structure of mutual benefit in coordination**.\n  - The expected payoff is low because coordination occurs only rarely by chance.\n\n#### 2. **Always Choose Coordination (e.g., Always 0 or Always 1)**\n- **Example**: Always choose Action A (score: 0.7233), Always choose Action B (score: 0.8197)\n- **Why it fails**:\n  - If you **always choose 0**, your partner (who prefers 1) will **almost always choose 1**, leading to **mismatch** and **zero payoff**.\n  - If you **always choose 1**, you get 1 when they choose 1 (payoff 1), but they get 2 -- you get less than when both choose 0.\n  - **No adaptation** to partner's behavior means **no learning or response**, and the strategy fails to exploit the **asymmetric preferences**.\n\n#### 3. **Exploitative Coordination with Fixed Probabilities (e.g., 60% chance to coordinate)**\n- **Example**: Exploitative Coordination with 60% chance (score: 0.6861)\n- **Why it fails**:\n  - It assumes a fixed probability of coordination, ignoring **feedback from history**.\n  - The partner may **detect and exploit** the predictability, choosing the opposite strategy to avoid coordination.\n  - This strategy **fails to adapt** to the partner's behavior -- especially when the partner consistently chooses..."
    }
  ],
  "metadata": {
    "task": "Battle of Sexes",
    "qa_type": "dead_ends"
  }
}
```

---

### A.6 Full Example: Training Data Sample (Deep Think Trace)

Source: `/home/jarnav/scratch/air-agent/counterfactual_qa/grounded_traces.jsonl`, line 1.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a senior ML research scientist. Analyze experiments carefully, always grounding your reasoning in the specific task description and data. Never apply reasoning from one domain to another (e.g., don't discuss 'survival rates' for a game theory task). Think step by step."
    },
    {
      "role": "user",
      "content": "You are a senior ML research scientist doing a post-mortem on a completed experiment series.\n\n## Task Description\nYou are a data scientist tasked with a classic machine learning problem: predicting the chances of survival on the Titanic.\nYou are provided with two CSV files: `train.csv` and `test.csv`.\n\nYour goal is to write a Python script named `train_and_predict.py` that does the following:\n1. Reads `train.csv`.\n2. Performs basic data cleaning and feature engineering.\n3. Trains a classification model to predict the 'Survived' column.\n4. Uses the trained model to make predictions on the `test.csv` data.\n5. Saves the predictions to a file named `submission.csv` with two columns: 'PassengerId' and 'Survived'.\n\n## Task Metric\naccuracy: higher is better\nBaseline score (no model): 0.7655\n\n## Complete Experiment Tree\nroot [0.7655] Baseline (no model execution)\n  root_0 [0.6675] Apply a gradient boosting model (e.g., XGBoost or LightGBM) with hyperparameter tuning\n  root_1 [0.8349] Apply a gradient-boosting model with categorical encoding using target encoding\n  root_2 [0.8301] Apply a gradient boosting model with target encoding for categorical features\n    root_1_0 [0.7847] Replace the gradient-boosting model with a neural network (MLP)\n    root_2_0 [0.8397] Replace the gradient boosting model with a deep neural network (DNN)\n    root_0_0 [FAILED] Replace the gradient boosting model with a neural network (MLP)\n      root_2_0_0 [0.8565] Replace the DNN with a random forest ensemble incorporating gradient boosting via stacking\n      root_1_0_0 [FAILED] Replace the neural network with a random forest ensemble with 100 trees\n      root_2_0_1 [0.8589] Replace the DNN with a random forest ensemble with bagged derived features\n        root_2_0_0_0 [FAILED] Replace the random forest ensemble with a lightweight neural network\n        root_2_0_1_0 [0.8325] Replace the ensemble with a gradient-boosted tree model with recursive feature elimination\n          root_2_0_1_0_0 [FAILED] Replace the gradient-boosted tree model with a neural network (MLP)\n\n## Best Result\nNode: root_2_0_1 scored 0.8589 (baseline: 0.7655)\nImprovement: +0.0933\n\n## Post-Mortem Analysis\n\nAnalyze this COMPLETED experiment tree. Think like a senior researcher writing a lab notebook entry:\n\n1. **Was the budget well-spent?**\n2. **What structural properties of THIS problem did the experiments reveal?**\n3. **Failure analysis**: For each failed or underperforming node, why did it fail FOR THIS SPECIFIC PROBLEM?\n4. **Transfer lessons**: What generalizable insights would help on SIMILAR problems?\n5. **If starting over**: With the same budget but the knowledge from these experiments, what would be your strategy?"
    },
    {
      "role": "assistant",
      "content": "**Lab Notebook Entry -- Post-Mortem Analysis: Titanic Survival Prediction Experiment**\n*Author: Senior ML Research Scientist*\n*Task: Predicting survival on the Titanic using machine learning*\n\n---\n\n### 1. **Was the budget well-spent? Did we explore enough before exploiting?**\n\nYes, the budget was well-spent -- the experiment tree demonstrates a **structured, hierarchical exploration** that balances exploration and exploitation effectively. The initial baseline (0.7655) is a reasonable starting point, and the path to the best result (0.8589) follows a clear progression from simple models to more complex ensembles, with deliberate pruning of underperforming paths.\n\nKey observations:\n- **Early exploration** was focused on gradient boosting (XGBoost/LightGBM) with different encoding strategies -- a natural first step for tabular data with mixed types.\n- The **target encoding** path (root_2) showed strong performance (0.8301), and its refinement into a **bagged feature ensemble** (root_2_0_1) yielded the best result (0.8589), suggesting that **feature-level aggregation and robustness to noise** matter more than model architecture.\n- Several attempts to replace tree-based models with neural networks (root_1_0, root_0_0) failed -- not due to lack of capability, but due to **mismatched inductive bias**.\n\n### 2. **What structural properties of THIS problem did the experiments reveal?**\n\nThe Titanic dataset has distinct structural properties:\n\n#### a) **Highly skewed categorical features with low cardinality**\n- Features like Pclass, Sex, Embarked have **low cardinality** and **strong monotonic relationships** with survival.\n- Pclass=1 has a survival rate of ~63%, Pclass=3 ~24% -- a clear monotonic trend.\n- **Inductive bias**: Tree-based models naturally capture such monotonic relationships. Neural networks, which assume feature independence, struggle without explicit feature engineering.\n\n#### b) **Missing values in key continuous features (Age)**\n- Age has ~20% missing entries. Simple imputation (mean/median) works, but **target encoding or k-NN imputation with survival context** improves performance.\n\n#### c) **Sparsity and noise in high-cardinality features**\n- Cabin has >700 unique values, most with missing or partial entries. It contributes little to survival prediction.\n- Tree-based models handle this naturally via feature importance and pruning. Neural networks overfit to noise.\n\n#### d) **Non-linear interactions between features (e.g., age x class)**\n- Survival depends on interactions (e.g., young women in first class survive better than young men in third class).\n- Gradient boosting excels at modeling such interactions through split-based decision boundaries.\n\n### 3. **Failure Analysis**\n\n| Node | Why it failed | Counterfactual |\n|------|--------------|----------------|\n| root_0_0 (NN -> MLP) | Neural networks failed to learn monotonic trends in categorical features. Overfitted on small training subsets. | If we had applied target encoding before a shallow MLP with dropout, ~0.80-0.82, still below tree-based. |\n| root_1_0 (NN -> MLP) | Deep architecture on small dataset (~891 train samples) with sparse features. | Shallow MLP (2-3 layers) with feature-wise normalization might have gotten ~0.80-0.82. |\n| root_1_0_0 (RF with 100 trees, bagging) | Bagging on raw features fails because features are not independent, categorical features not properly encoded. | One-hot + target encoding before bagging would have improved, but gradient boosting would still outperform. |\n| root_2_0_0_0 (lightweight single dense layer) | A single dense layer cannot model complex interactions (age x class). | 2-layer MLP with ReLU and dropout on feature-transformed dataset might have worked. |\n\n### 4. **Transfer Lessons**\n\n- **Problems with low-cardinality categorical features + monotonic target relationships** strongly favor tree-based methods.\n- **When neural networks fail on tabular data**, the root cause is usually mismatched inductive bias, not hyperparameter tuning.\n- **Feature-level ensembles (bagging + stacking)** outperform model-level ensembles when the signal is distributed across many weak features.\n\n### 5. **If Starting Over**\n\nSkip: all neural network attempts. They consumed 5 nodes and produced 3 failures + 2 suboptimal results.\n\nStrategy:\n1. Start with gradient boosting + target encoding (root_1 path, 0.8349)\n2. Try random forest ensemble with bagged features (root_2_0_1 path, 0.8589)\n3. Focus remaining budget on feature engineering: family size, title extraction, age x class interactions\n4. Allocate 60% explore (first 3 nodes), 40% exploit (remaining budget)"
    }
  ],
  "metadata": {
    "task": "Titanic Survival Prediction",
    "sample_type": "retro",
    "total_nodes": 13,
    "best_score": 0.8588516746411483,
    "baseline_score": 0.76555
  }
}
```

---

### A.7 VoI Implementation: Full Source Code

#### A.7.1 VoI Computation (`voi.py`)

Source: `/home/jarnav/MLScientist/air-agent/air/ttt/VoI_guided_rl/voi.py`

```python
"""
Value of Information (VoI) computation.

Measures how much resolving a hypothesis would change the scientist's
experiment proposals. Uses centroid cosine distance in embedding space.

VoI(H) = 1 - cosine_similarity(
    mean_embed(proposals | H unresolved),
    mean_embed(proposals | H resolved)
)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def embed_texts(
    texts: list[str],
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 512,
) -> np.ndarray:
    """Get mean-pooled embeddings from the model's hidden states."""
    if not texts:
        return np.zeros((0, 1))

    embeddings = []
    model.eval()
    device = next(model.parameters()).device

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pool over tokens
            hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
            embeddings.append(pooled.cpu().numpy()[0])

    return np.stack(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_voi(
    proposals_unresolved: list[str],
    proposals_H_true: list[str],
    proposals_H_false: list[str],
    p_true: float,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
) -> float:
    """Compute Value of Information using centroid cosine distance.

    Args:
        proposals_unresolved: K experiment proposals with current thought.md
        proposals_H_true: K proposals with hypothesis confirmed
        proposals_H_false: K proposals with hypothesis rejected
        p_true: P(H=true) from logit estimation
        model: the scientist model (for embeddings)
        tokenizer: the tokenizer

    Returns:
        VoI score in [0, 2] (0 = no information, higher = more informative)
    """
    if not proposals_unresolved or not proposals_H_true or not proposals_H_false:
        return 0.0

    # Embed all proposals
    emb_u = embed_texts(proposals_unresolved, model, tokenizer)
    emb_t = embed_texts(proposals_H_true, model, tokenizer)
    emb_f = embed_texts(proposals_H_false, model, tokenizer)

    # Centroids
    centroid_u = emb_u.mean(axis=0)

    # Resolved centroid = weighted mix of H_true and H_false centroids
    centroid_t = emb_t.mean(axis=0)
    centroid_f = emb_f.mean(axis=0)
    centroid_r = p_true * centroid_t + (1 - p_true) * centroid_f

    # VoI = 1 - cosine_similarity (higher = more information gain)
    voi = 1.0 - cosine_similarity(centroid_u, centroid_r)

    return max(0.0, voi)


def estimate_p_true(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    thought_md: str,
    hypothesis: str,
    task_description: str,
) -> float:
    """Estimate P(H=true) from model logits.

    Prompts the model with thought.md + hypothesis and reads the logit
    probabilities for "true" vs "false" tokens. Non-gameable because
    the model doesn't control its own logits.
    """
    prompt = (
        f"Given the following analysis and problem description, "
        f"is this hypothesis true?\n\n"
        f"Analysis:\n{thought_md}\n\n"
        f"Problem: {task_description}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Answer (true or false):"
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token logits

    # Find token IDs for "true" and "false"
    true_tokens = tokenizer.encode(" true", add_special_tokens=False)
    false_tokens = tokenizer.encode(" false", add_special_tokens=False)

    if not true_tokens or not false_tokens:
        return 0.5  # fallback

    true_logit = logits[true_tokens[0]].item()
    false_logit = logits[false_tokens[0]].item()

    # Softmax over just these two
    max_logit = max(true_logit, false_logit)
    p_true = np.exp(true_logit - max_logit) / (
        np.exp(true_logit - max_logit) + np.exp(false_logit - max_logit)
    )

    return float(np.clip(p_true, 0.05, 0.95))


def sample_proposals(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompt: str,
    K: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
) -> list[str]:
    """Sample K experiment proposals from the scientist model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    proposals = []
    for _ in range(K):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if text:
            proposals.append(text)

    return proposals
```

#### A.7.2 thought.md Structure (`thought.py`)

Source: `/home/jarnav/MLScientist/air-agent/air/ttt/VoI_guided_rl/thought.py`

```python
"""
thought.md -- the scientist's living mental model.

Contains active hypotheses with confidence levels, structural claims,
experimental results, and open questions. Updated after every experiment.
Replaces the flat memory list from the original tree search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"       # Just introduced, not yet tested
    TESTING = "testing"         # Validate nodes being run
    VALIDATED = "validated"     # Enough evidence to consider true
    REJECTED = "rejected"       # Evidence says false
    ABANDONED = "abandoned"     # Ran out of validate budget without resolution


@dataclass
class Hypothesis:
    id: str                                    # e.g. "H1"
    claim: str                                 # e.g. "FamilySize is a useful latent variable"
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5                    # 0-1, updated externally from logits
    validate_budget: int = 3                   # max validate nodes for this hypothesis
    validate_used: int = 0
    evidence: list[str] = field(default_factory=list)


@dataclass
class ThoughtDoc:
    """The scientist's structured mental model of the problem."""

    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    structural_claims: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    _next_id: int = 1

    def add_hypothesis(self, claim: str, validate_budget: int = 3) -> str:
        """Add a new hypothesis. Returns the hypothesis ID."""
        hid = f"H{self._next_id}"
        self._next_id += 1
        self.hypotheses[hid] = Hypothesis(id=hid, claim=claim, validate_budget=validate_budget)
        return hid

    def get_unvalidated(self) -> list[Hypothesis]:
        """Get hypotheses that can still be validated."""
        return [
            h for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
            and h.validate_used < h.validate_budget
        ]

    def get_validated(self) -> list[Hypothesis]:
        """Get hypotheses that have been validated (eligible for challenge)."""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.VALIDATED]

    def record_validate(self, hid: str, result: str, success: bool):
        """Record a validate experiment result."""
        h = self.hypotheses[hid]
        h.validate_used += 1
        h.evidence.append(result)
        h.status = HypothesisStatus.TESTING
        if success:
            h.status = HypothesisStatus.VALIDATED
        elif h.validate_used >= h.validate_budget:
            h.status = HypothesisStatus.ABANDONED
            h.confidence = max(0.1, h.confidence - 0.3)

    def record_challenge(self, hid: str, result: str, hypothesis_survives: bool):
        """Record a challenge experiment result."""
        h = self.hypotheses[hid]
        h.evidence.append(f"[challenge] {result}")
        if not hypothesis_survives:
            h.status = HypothesisStatus.REJECTED
            h.confidence = 0.1

    def add_structural_claim(self, claim: str):
        """Add a structural claim about the problem."""
        if claim not in self.structural_claims:
            self.structural_claims.append(claim)

    def add_open_question(self, question: str):
        if question not in self.open_questions:
            self.open_questions.append(question)

    def render(self) -> str:
        """Render thought.md as text for the scientist prompt."""
        lines = ["# thought.md\n"]

        # Active hypotheses
        lines.append("## Hypotheses")
        if not self.hypotheses:
            lines.append("(No hypotheses yet -- the problem is unexplored.)\n")
        for h in self.hypotheses.values():
            status_icon = {
                HypothesisStatus.PROPOSED: "?",
                HypothesisStatus.TESTING: "~",
                HypothesisStatus.VALIDATED: "V",
                HypothesisStatus.REJECTED: "X",
                HypothesisStatus.ABANDONED: "--",
            }.get(h.status, "?")
            lines.append(f"- [{status_icon}] {h.id}: {h.claim}")
            lines.append(f"  status={h.status.value}, confidence={h.confidence:.2f}, "
                        f"validate_budget={h.validate_used}/{h.validate_budget}")
            for ev in h.evidence[-2:]:  # last 2 evidence entries
                lines.append(f"  evidence: {ev[:150]}")
        lines.append("")

        # Structural claims
        if self.structural_claims:
            lines.append("## Structural Claims")
            for c in self.structural_claims:
                lines.append(f"- {c}")
            lines.append("")

        # Open questions
        if self.open_questions:
            lines.append("## Open Questions")
            for q in self.open_questions[-3:]:  # cap at 3
                lines.append(f"- {q}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for saving."""
        return {
            "hypotheses": {
                hid: {
                    "id": h.id, "claim": h.claim, "status": h.status.value,
                    "confidence": h.confidence, "validate_budget": h.validate_budget,
                    "validate_used": h.validate_used, "evidence": h.evidence,
                }
                for hid, h in self.hypotheses.items()
            },
            "structural_claims": self.structural_claims,
            "open_questions": self.open_questions,
        }
```

**Example rendered thought.md output:**
```
# thought.md

## Hypotheses
- [V] H1: FamilySize is a useful latent variable
  status=validated, confidence=0.85, validate_budget=2/3
  evidence: LightGBM +0.021, RandomForest +0.018
  evidence: [challenge] IsAlone + LargeFamilyFlag scores 0.801 vs FamilySize 0.793
- [?] H2: IsAlone interacts with Pclass and Sex
  status=proposed, confidence=0.50, validate_budget=0/3

## Structural Claims
- Tree-based models dominate on this dataset
- Categorical features have low cardinality

## Open Questions
- Does age imputation strategy matter for tree-based models?
```

#### A.7.3 Reward Structure (`rewards.py`)

Source: `/home/jarnav/MLScientist/air-agent/air/ttt/VoI_guided_rl/rewards.py`

```python
"""
Three-component reward structure for VoI-guided RL.

R1 -- Resolution: Did the experiment resolve its hypothesis?
     Sharpness-weighted prediction scoring.

R2 -- Information: Did the explore node introduce a valuable hypothesis?
     Measured by VoI (centroid cosine distance).

R3 -- Performance: Did the tree find a good solution?
     (best_score - baseline) / baseline. Applied once at end of tree.

Combined:
    R_node = alpha * R1 + beta * R2     (per node, dense signal)
    R_final = gamma * R3                 (end of tree, sparse signal)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NodeReward:
    r1_resolution: float = 0.0
    r2_information: float = 0.0
    r_node: float = 0.0


@dataclass
class TreeReward:
    r3_performance: float = 0.0
    node_rewards: dict[str, NodeReward] = None  # node_id -> NodeReward

    def __post_init__(self):
        if self.node_rewards is None:
            self.node_rewards = {}


# --- R1: Resolution Reward ---

def compute_r1(
    prediction_lower: float,
    prediction_upper: float,
    actual_score: float,
    max_r1: float = 1.0,
) -> float:
    """Compute resolution reward from sharpness-weighted prediction.

    R1 = prediction_correct / max(prediction_interval, 0.01)

    Narrow correct prediction -> high reward.
    Wide vague prediction -> low reward even if correct.
    Wrong prediction -> zero.
    """
    interval = max(prediction_upper - prediction_lower, 0.01)
    correct = prediction_lower <= actual_score <= prediction_upper

    if correct:
        return min(max_r1, 1.0 / interval)
    else:
        return 0.0


def parse_prediction(text: str) -> tuple[float, float] | None:
    """Extract prediction interval [lower, upper] from scientist output.

    Looks for patterns like:
    - "prediction: score will be 0.85-0.90"
    - "prediction: improvement of 0.02-0.04"
    - "prediction: [0.85, 0.90]"
    """
    m = re.search(r'prediction.*?(\d+\.?\d*)\s*[-\u2013,to]\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        lower, upper = float(m.group(1)), float(m.group(2))
        if lower > upper:
            lower, upper = upper, lower
        return (lower, upper)

    m = re.search(r'(?:will|should|expect).*?(?:score|improve).*?(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return (val * 0.95, val * 1.05)

    return None


# --- R2: Information Reward ---

def compute_r2(voi_score: float) -> float:
    """R2 = VoI for explore nodes, 0 for validate/challenge nodes."""
    return voi_score


# --- R3: Performance Reward ---

def compute_r3(best_score: float, baseline_score: float) -> float:
    """R3 = (best_score - baseline) / max(|baseline|, 0.01)

    Applied once at end of tree. Grounds the search in actual performance.
    """
    if baseline_score == 0:
        return best_score
    return (best_score - baseline_score) / max(abs(baseline_score), 0.01)


# --- Combined Reward ---

def compute_node_reward(
    r1: float,
    r2: float,
    alpha: float = 0.4,
    beta: float = 0.3,
) -> float:
    """Per-node reward: R_node = alpha * R1 + beta * R2"""
    return alpha * r1 + beta * r2


def compute_tree_rewards(
    node_rewards: dict[str, NodeReward],
    best_score: float,
    baseline_score: float,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    gamma_discount: float = 0.95,
    node_order: list[str] | None = None,
) -> dict[str, float]:
    """Compute final per-node cumulative rewards.

    Per-node: R_node = alpha * R1 + beta * R2
    End of tree: R3 = performance, discounted backward through tree
    Total per node: R_node + discounted R3 contribution

    Returns: {node_id: total_reward}
    """
    r3 = compute_r3(best_score, baseline_score)

    if node_order is None:
        node_order = sorted(node_rewards.keys())

    n = len(node_order)
    total_rewards = {}

    for i, nid in enumerate(node_order):
        nr = node_rewards[nid]
        # Per-node reward
        r_node = compute_node_reward(nr.r1_resolution, nr.r2_information, alpha, beta)

        # Discounted R3: later nodes get more credit (they benefited from earlier exploration)
        # But we also give credit to early nodes that set direction
        discount = gamma_discount ** (n - 1 - i)  # earlier nodes get less R3
        r_final = gamma * r3 * discount

        total_rewards[nid] = r_node + r_final

    return total_rewards
```

**Key design decisions in the reward structure:**

1. **R1 (Resolution)** uses sharpness-weighted prediction scoring: `1.0 / interval_width` if the actual score falls within the predicted interval, 0 otherwise. This incentivizes the scientist to make narrow, correct predictions rather than wide vague ones.

2. **R2 (Information)** directly uses the VoI score from `voi.py`. Only applies to explore nodes (introducing new hypotheses). A hypothesis with high VoI means resolving it would change what experiments the scientist proposes next.

3. **R3 (Performance)** is normalized improvement over baseline: `(best - baseline) / |baseline|`. Applied once at the end of the entire tree, discounted backward so that later nodes (which had more information to work with) get more R3 credit than early nodes.

4. **Default weights**: alpha=0.4 (resolution), beta=0.3 (information), gamma=0.3 (performance). The dense per-node rewards (R1+R2) receive 70% of the total weight, providing strong credit assignment signal. The sparse end-of-tree R3 receives 30%, ensuring the scientist stays grounded in actually solving the problem.
