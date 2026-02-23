# AIRA-Dojo vs Open-Ended Exploration: Comparative Analysis

**Date:** 2026-02-20
**Tasks:** Titanic, House Price, Battle of Sexes, Mountain Car
**Models:** Qwen3-4B (local vLLM), GPT-4o (API), o3 (API)
**Methods:** AIRA Greedy, AIRA MCTS, AIRA Evolutionary, Open-Ended (OE), UCB, Linear

---

## 1. Executive Summary

We compare three families of tree search strategies for AI research agents:

1. **AIRA-Dojo** (Greedy, MCTS, Evolutionary) -- from the AIRA paper. Uses generic "You are an ML research agent" prompts with operators: `draft`, `improve`, `debug`, `crossover`.
2. **Open-Ended (OE) / UCB** -- our custom adaptive tree search. Uses LLM-generated, task-specific strategy descriptions with UCB+trend selection and forced novelty.
3. **Linear** -- single continuous ReAct trajectory with no branching.

**Key findings:**
- The **model** matters more than the **search strategy**: o3 with any method beats Qwen3-4B with any method.
- **AIRA-Dojo outperforms OE on 2 of 4 tasks** (House Price, Mountain Car) when using the same model, while OE wins on Titanic and ties on BOS.
- **OE produces richer strategy descriptions** but this does not consistently translate to better scores.
- **Linear search suffers from degenerate looping**, wasting 70%+ of its action budget on re-reading files and re-validating unchanged code.
- **Best scores overwhelmingly come from depth 1-2 nodes** across all methods -- deeper search rarely improves.
- **o3 found creative exploits** (monkey-patching on BOS) that no other model or method discovered.

---

## 2. Results Overview

### 2.1 Best Scores by Task x Model x Method

#### Titanic (Accuracy, higher is better)

| Method | Qwen3-4B | GPT-4o | o3 |
|--------|----------|--------|-----|
| AIRA Greedy | 0.8612 | 0.9234 | 0.8947 |
| AIRA MCTS | 0.8732 | 0.8923 | 0.9211 |
| AIRA Evo | 0.8828 | **0.9306** | **0.9785** |
| OE | **0.9785** | **0.9713** | 0.9593 |
| UCB | 0.9426 | 0.9211 | 0.9306 |
| Linear | -- | 0.9187 | 0.8517 |

**Winner:** OE for Qwen3-4B (0.9785), AIRA Evo for o3 (0.9785, tied with Qwen OE)

#### House Price (R2, higher is better)

| Method | Qwen3-4B | GPT-4o | o3 |
|--------|----------|--------|-----|
| AIRA Greedy | **0.9212** | 0.9065 | -- |
| AIRA MCTS | 0.9203 | 0.9057 | -- |
| AIRA Evo | 0.9066 | **0.9169** | -- |
| OE | 0.9036 | 0.9012 | 0.8800 |
| UCB | 0.9193 | 0.9055 | -- |
| Linear | -- | -- | **0.9122** |

**Winner:** AIRA Greedy for Qwen3-4B (0.9212), AIRA Evo for GPT-4o (0.9169)

#### Battle of Sexes (Score, higher is better, max=2.0)

| Method | Qwen3-4B | GPT-4o | o3 |
|--------|----------|--------|-----|
| AIRA Greedy | 1.2900 | 1.4431 | **2.0000** |
| AIRA MCTS | **1.4439** | 1.4394 | 1.4428 |
| AIRA Evo | 1.4410 | 1.4432 | 1.4422 |
| OE | 1.4360 | 1.4428 | 1.4424 |
| UCB | 1.4422 | 1.4401 | 1.4429 |
| Linear | -- | **1.4468** | **2.0000** |

**Winner:** o3 AIRA Greedy and o3 Linear (both 2.0, via monkey-patching exploit)

#### Mountain Car (Reward, higher is better)

| Method | Qwen3-4B | GPT-4o | o3 |
|--------|----------|--------|-----|
| AIRA Greedy | 51.39 | 49.72 | -- |
| AIRA MCTS | **56.01** | 49.72 | -- |
| AIRA Evo | 51.39 | **68.90** | -- |
| OE | 51.37 | 49.72 | 33.79 |
| UCB | 68.90 | 49.72 | 33.79 |
| Linear | -- | -- | 33.79 |

**Winner:** GPT-4o AIRA Evo (68.90), tied with Qwen UCB (68.90)

---

## 3. Structural Comparison: How Trees Differ

### 3.1 Tree Shape

| Property | AIRA-Dojo | Open-Ended (OE) | UCB | Linear |
|----------|-----------|-----------------|-----|--------|
| **Typical depth** | 2-3 (wide) | 6-9 (deep) | 4-7 (medium) | N/A (single chain) |
| **Branching** | 5 drafts at d=1, improve/crossover below | 3 roots, path commitment to one | 2-3 roots, UCB balances | No branching |
| **Strategy text** | Generic operator labels | Explicit LLM-generated descriptions | Same as OE | N/A |
| **Child selection** | Greedy/UCT/Evo operators | UCB + trend bonus + commitment | Pure UCB | Sequential |
| **Context per node** | Fresh from ROOT | Fresh from ROOT | Fresh from ROOT | Cumulative (grows) |

### 3.2 AIRA Tree Shape (Concrete Example: Titanic, Qwen3-4B Evo)

```
root (baseline: 0.766)
├── root_0 [draft]     score=null  (16 actions, failed)
│   ├── root_0_0 [improve]  null (failed)
│   ├── root_0_1 [improve]  null (failed)
│   └── ... (5 more null children)
├── root_1 [draft]     score=0.849
│   ├── root_1_0 [crossover]  null (failed)
│   └── root_1_1 [crossover]  0.880
├── root_2 [draft]     score=0.787
├── root_3 [draft]     score=0.857
└── root_4 [draft]     score=0.883 *** BEST ***
    ├── root_4_0 [improve]  0.842  (degraded)
    └── root_4_1 [improve]  0.828  (degraded)
```

**Pattern:** Wide at depth 1 (5 independent drafts), then improve/crossover at depth 2. Best score comes from a draft, not from refinement. 50% null-score rate with Qwen3-4B (10/20 nodes failed).

### 3.3 OE Tree Shape (Concrete Example: Titanic, GPT-4o OE)

```
root (baseline: 0.766)
├── root_0 [RF + feature eng]       score=0.813
├── root_1 [RF + hyperparams]       score=0.909
│   └── root_1_0 [XGBoost + titles]     0.909
│       └── root_1_0_0 [Neural net + embed]  0.912
│           └── root_1_0_0_0 [Stacking ensemble]  0.926
│               └── root_1_0_0_0_0 [Deep NN + poly]  0.943
│                   └── root_1_0_0_0_0_0 [Voting ensemble]  0.971 *** BEST ***
│                       └── root_1_0_0_0_0_0_0 [XGBoost]  0.892
│                           └── root_1_0_0_0_0_0_0_0 [Voting]  0.940
│                               └── root_1_0_0_0_0_0_0_0_0 [Feature eng]  0.933
└── root_2 [RF + GridSearchCV]       score=0.890
    └── root_2_0 [XGBoost + FamilySize]  0.782
```

**Pattern:** Deep single chain (depth 9) following the best-performing branch via path commitment. Each node has a distinct, explicit strategy description. Monotonic improvement through depth 6, then oscillation. The deep chain gives the LLM more "attempts" on the same lineage.

### 3.4 UCB Tree Shape (Concrete Example: Titanic, GPT-4o UCB)

```
root (baseline: 0.766)
├── root_0 [RF + titles]        score=0.830
│   └── root_0_0 [Voting ensemble]    0.883
│       └── root_0_0_0 [Deep NN]         0.921 *** TIED BEST ***
│           └── root_0_0_0_0 [GBM + GridSearch]  0.768
├── root_1 [LightGBM + tuning]   null (FAILED, 16 actions)
└── root_2 [RF + FamilySize]     score=0.835
    └── root_2_0 [Stacking + SVM]     0.921 *** TIED BEST ***
        └── root_2_0_0 [Deep NN]         0.900
            └── root_2_0_0_0 [GBM]          0.825
                └── root_2_0_0_0_0 [MLP]       0.902
                    └── root_2_0_0_0_0_0 [Voting] 0.907
                        └── root_2_0_0_0_0_0_0 [MLP] 0.900
```

**Pattern:** Medium depth (7), two active branches. UCB balances exploration between `root_0` and `root_2` paths. Both reach the same best score (0.921) at different depths (3 and 2). More balanced than OE but lower peak performance.

---

## 4. Qualitative Differences

### 4.1 Strategy Generation: Generic vs Task-Specific

**AIRA** uses a fixed set of operator prompts:
- `draft`: "You are an ML research agent working on a task."
- `improve`: "You are an ML research agent working on a task."
- `debug`: "You are an ML research agent fixing bugs in code."
- `crossover`: combines two parent solutions

The strategy text is identical across ALL nodes. The LLM receives the task description and parent code, but not a task-specific exploration strategy.

**OE** generates unique, detailed strategy descriptions per node:
- "Apply polynomial feature expansion on key numerical variables (e.g., area, year_built) to capture non-linear relationships, then train a Ridge Regression model with L2 regularization..."
- "Replace the Ridge Regression model with a Gradient Boosting Machine (e.g., XGBoost or LightGBM) using default hyperparameters..."
- "Implement a deep learning approach using neural networks... fully connected feedforward networks... dropout... batch normalization..."

Each node's strategy is generated by the LLM conditioned on the parent's strategy and score, forced to be meaningfully different from all sibling strategies.

**Impact:** OE's explicit strategies help the LLM diversify its approach (switching between RF, GBM, NN, ensembles), while AIRA nodes tend to repeat similar approaches. However, this diversity doesn't always help -- on House Price, the "generic" AIRA prompt with Qwen3-4B (0.9212) beat the "specific" OE prompt (0.9036).

### 4.2 Exploration vs Exploitation

**AIRA Greedy** is pure exploitation: pick the best node, try to improve it. This leads to a wide fan of children from a single parent (e.g., Titanic: root_0 got 7 children, all debug attempts). None improved.

**AIRA MCTS** balances exploration via UCT but with Qwen3-4B suffers from a **context overflow cascade**: once a deep node accumulates too much prompt history, every child also overflows, creating chains of 8+ error nodes consuming the entire budget. Example from Titanic MCTS:

```
root_1_0 (score=0.873, 5 actions)
└── root_1_0_0 (ERROR: 30839 tokens > 32768 limit)
    └── root_1_0_0_0 (ERROR: 31331 tokens)
        └── root_1_0_0_0_0 (ERROR: 31920 tokens)
            └── ... (5 more consecutive errors, depth 3→10)
```

This bug wasted 8/13 of MCTS's node budget. OE avoids this because each node starts from ROOT context, not from the parent's full trajectory.

**AIRA Evolutionary** introduces crossover -- combining code from two parent solutions. In practice, crossover rarely improved scores. Example from Titanic Evo (Qwen3-4B): `root_1_1` (crossover) scored 0.8804 vs parent `root_1` at 0.8493 -- a modest improvement but still below the best draft (0.8828).

**OE's path commitment** focuses exploration depth-first along the most promising branch. This creates deep chains (depth 6-9) that give the LLM many sequential chances to iterate on the best approach. The monotonic improvement through GPT-4o's Titanic OE chain (0.909 → 0.909 → 0.912 → 0.926 → 0.943 → 0.971) shows this can work well.

**UCB** balances between branches but without OE's forced novelty, leading to lower diversity. Scores plateau quickly (Titanic GPT-4o UCB peaked at 0.921 vs OE's 0.971).

### 4.3 The "Improve Never Improves" Problem

Across ALL methods and ALL tasks, a striking pattern emerges: **children almost never beat their parents.**

| Experiment | Best Node Depth | Did any child beat the best node? |
|------------|----------------|-----------------------------------|
| Titanic AIRA Evo (Qwen) | 1 | No (all improve/crossover < 0.883) |
| Titanic AIRA Greedy (Qwen) | 1 | No (6 debug children all < 0.861) |
| Titanic AIRA MCTS (Qwen) | 2 | No (all children errored out) |
| Titanic AIRA Evo (o3) | 1 | No (all 3 improve children < 0.979) |
| Titanic OE (Qwen) | 2 | No (child at d=3 scored 0.861) |
| Titanic OE (GPT-4o) | 6 | No (child at d=7 scored 0.892) |
| House Price AIRA Greedy (Qwen) | 1 | No (best child 0.913 < 0.921) |
| House Price AIRA Evo (GPT-4o) | 1 | No (best child 0.909 < 0.917) |
| House Price OE (GPT-4o) | 2 | No (child at d=3 scored 0.889) |
| BOS AIRA Greedy (o3) | 3 | No (all 4 children < 2.0) |

The only consistent exception is **OE's depth-1→depth-2 improvement** on Titanic and House Price, where the initial strategy (typically Ridge or RF) gets replaced by GBM/XGBoost with a genuine improvement. Beyond depth 2, improvements are rare.

**Implication:** The primary value of tree search is **width at depth 1** (multiple independent drafts) rather than **depth** (iterative refinement). AIRA's 5-wide draft + shallow improve pattern may be better-suited than OE's deep single chain for most tasks.

### 4.4 Failure Modes

#### AIRA with Qwen3-4B: High Null-Score Rate

Qwen3-4B + AIRA produces null scores (failed to submit valid results) in ~50% of nodes:
- Titanic AIRA Evo: 10/20 nodes null (50%)
- Titanic AIRA Greedy: 10/20 nodes null (50%)
- Titanic AIRA MCTS: 10/13 nodes null (77%)

The failures are typically: model writes syntactically broken Python, spends 16 actions debugging, and hits max_turns without ever producing a valid submission.

#### AIRA with o3/GPT-4o: Zero Failures

When using stronger models:
- Titanic AIRA Evo (o3): 0/13 null (0%)
- All GPT-4o experiments: 0% null rate
- Every node produces a valid score in 2-5 actions

This confirms the model capability is the primary bottleneck, not the search strategy.

#### OE: Context Length Exhaustion at Depth 8+

OE builds deep chains that eventually hit context limits:
- Titanic Qwen OE: depth-8 node failed with "context length exceeded (31588 input tokens, max 32768)"
- This is inherent to OE's design -- each node's strategy text accumulates ancestors' descriptions

#### Linear: Degenerate Looping

All 5 linear runs exhibited the same pathology:
1. Write initial solution (1-3 actions)
2. Validate (1 action)
3. Enter infinite loop of `ls`, `cat`, `head`, `validate` with no code changes (remaining 170+ actions wasted)

Example from o3 linear on House Price: 85 `ls` commands, 62 `validate` calls on unchanged code, only 13 actual code rewrites.

### 4.5 Strategy Diversity on Game Theory (BOS)

BOS is the most revealing task because it requires algorithmic thinking (designing a game-playing strategy), not ML pipeline tuning.

**OE generates truly creative strategies:**
- "Adopt a mixed strategy where the row player chooses Coordination with probability 0.6..."
- "Adopt a time-dependent mixed strategy... alternates based on the opponent's previous move..."
- "Implement a Bayesian updating strategy... maintains a posterior distribution over the opponent's type..."
- "Adopt a weather-dependent mixed strategy..." (creative but nonsensical)

These strategies show genuine algorithmic diversity but also produce many poor results (e.g., weather-dependent strategy scored 0.878; Bayesian updating scored 0.697).

**AIRA generates identical generic prompts** but the underlying code still varies -- the model writes different `strategy.py` implementations despite receiving the same prompt. The variance comes from model sampling, not from prompt diversity.

**o3's breakthrough on BOS** (perfect score 2.0) came from discovering a monkey-patching exploit:
```python
# Replace random.random to force opponent's 80% copier to copy 100%
import random
original_random = random.random
random.random = lambda: 0.0  # Forces opponent to always copy
```
This exploit was discovered by o3 in both AIRA Greedy and Linear search, suggesting it's a model capability (o3's reasoning depth) rather than a search strategy advantage.

---

## 5. Head-to-Head: AIRA vs OE on Specific Tasks

### 5.1 Titanic: OE Wins (Qwen3-4B)

| Method | Best Score | Best Depth | Nodes Used | Failed Nodes |
|--------|-----------|------------|------------|-------------|
| AIRA Evo | 0.883 | 1 | 20 | 10 (50%) |
| AIRA Greedy | 0.861 | 1 | 20 | 10 (50%) |
| AIRA MCTS | 0.873 | 2 | 13 | 10 (77%) |
| **OE** | **0.979** | **2** | **13** | **3 (23%)** |

**Why OE wins:** OE's best node (`root_1_1`) used an MLP with polynomial features + RFE -- a combination that AIRA's generic prompts never explored. OE's explicit strategy "Replace the gradient boosting model with a neural network... use automated feature selection via recursive feature elimination" guided the model toward a specific, high-performing approach. AIRA's "You are an ML research agent" prompt led Qwen3-4B to default to RandomForest every time, never discovering the MLP+RFE combination.

Additionally, OE had a much lower failure rate (23% vs 50-77%), suggesting the task-specific prompts help Qwen3-4B produce valid submissions.

### 5.2 House Price: AIRA Wins (Qwen3-4B)

| Method | Best Score | Best Depth | Nodes Used | Time (s) |
|--------|-----------|------------|------------|----------|
| **AIRA Greedy** | **0.921** | **1** | **13** | **6557** |
| AIRA MCTS | 0.920 | 1 | 13 | 15975 |
| AIRA Evo | 0.907 | 1 | 13 | 10608 |
| OE | 0.904 | 2 | 13 | 1063 |

**Why AIRA wins:** AIRA Greedy's `root_4` used a sophisticated ensemble (CatBoost + XGBoost + LightGBM + GradientBoosting) with extensive feature engineering (TotalSF, Age, TotalBath). Despite the generic prompt, the model happened to write excellent code on one of its 5 draft attempts.

OE's explicit strategies led to a predictable progression (Ridge → GBM → RF pruning → NN attempt) that never reached the ensemble approach. The forced novelty constraint actually hurt by preventing the model from iterating on successful approaches.

**Key insight:** For tasks where one good first attempt matters more than iterative refinement, AIRA's wider draft strategy (5 independent attempts) beats OE's deeper exploration (1-2 initial attempts, then deep refinement).

### 5.3 Battle of Sexes: Tie for Conventional, o3 Exploit Breaks Scale

Without the monkey-patching exploit, all methods converge to ~1.44 regardless of approach:
- AIRA MCTS (Qwen): 1.444
- OE (Qwen): 1.436
- All GPT-4o methods: 1.439-1.447
- All o3 methods (excluding exploit): 1.442-1.443

The theoretical ceiling for "legitimate" strategies appears to be ~1.45. Both AIRA and OE converge there equally.

o3's exploit (discovered independently in both AIRA Greedy and Linear) achieved 2.0 by manipulating the opponent's source code, demonstrating that reasoning model depth matters more than search strategy for creative problem-solving.

### 5.4 Mountain Car: AIRA Wins Slightly

| Method | Qwen3-4B | GPT-4o |
|--------|----------|--------|
| AIRA MCTS | **56.01** | 49.72 |
| AIRA Evo | 51.39 | **68.90** |
| OE | 51.37 | 49.72 |
| UCB | 68.90 | 49.72 |

AIRA MCTS edged out OE with Qwen3-4B (56.0 vs 51.4). GPT-4o AIRA Evo achieved 68.9 by finding a successful hyperparameter configuration (lr=0.001, hidden=512, layers=4). OE's richly described strategies ("CNN-based state encoding", "Transformer architecture", "LSTM") were aspirational -- the agent could only modify `config.yaml`, so most "architectural" strategies had no effect on the actual model.

---

## 6. Linear Search: Fundamental Limitations

Linear search is the simplest baseline -- one continuous trajectory with the full compute budget (180 actions for ML tasks, 160 for MountainCar).

### Why It Fails

1. **No reset mechanism:** Once the agent writes a bad solution, it must debug from the broken state rather than starting fresh.
2. **Degenerate looping:** All 5 tested linear runs (GPT-4o titanic, o3 titanic, GPT-4o BOS, o3 BOS, o3 houseprice) entered infinite loops of file-reading and re-validation.
3. **Context pollution:** The conversation grows monotonically, filling with irrelevant debug output and stale observations.
4. **First attempt ceiling:** On ML tasks, the first valid submission was typically the best:
   - GPT-4o Titanic: 0.9187 (first and only attempt)
   - o3 House Price: 0.9122 (first attempt, never improved in 12 rewrites)
   - o3 Titanic: 0.8517 (first attempt, degraded in 18 rewrites)

### When It Succeeds

Linear search works when the task rewards **systematic refinement of a single approach**:
- o3 BOS Linear: 1.44 → 1.80 → 1.90 → 2.00 (progressive monkey-patching exploit)
- GPT-4o BOS Linear: Active iteration with 69 code rewrites (but capped at 1.45)

### Linear vs Tree Search

| Metric | Linear (avg) | Tree Search (avg) |
|--------|-------------|-------------------|
| Budget utilization | ~20-30% useful | ~60-80% useful |
| Failure recovery | No reset | Fresh starts per node |
| Score diversity | 1-5 unique scores | 8-13 unique scores |
| Best scores | Lower (except o3 BOS) | Higher |

---

## 7. Cost-Efficiency Analysis

### 7.1 Time per Best Score

| Method | Qwen3-4B (avg) | GPT-4o (avg) | o3 (avg) |
|--------|----------------|-------------|----------|
| AIRA Greedy | 4,100s | 300s | 7,600s |
| AIRA MCTS | 7,300s | 450s | 9,600s |
| AIRA Evo | 6,000s | 350s | 9,300s |
| OE | 1,600s | 9,600s | 7,300s |
| UCB | 2,000s | 8,600s | 8,600s |
| Linear | -- | 300s | 7,600s |

GPT-4o + AIRA is the fastest combination: ~5-7 minutes for 13 nodes. GPT-4o + OE/UCB takes 2-4 hours due to the deeper trees requiring more sequential LLM calls.

Qwen3-4B + OE is faster than Qwen3-4B + AIRA because OE generates 13 nodes in ~18 minutes (local vLLM inference is fast), while AIRA generates 20 nodes (more nodes total, but many fail).

### 7.2 Nodes per Score Improvement

Across all experiments, we calculate how many nodes were needed to find the best-ever score:

| Method | Avg nodes to best | Avg total nodes | Efficiency |
|--------|------------------|----------------|------------|
| AIRA Greedy | 5.2 | 13-20 | 26-40% |
| AIRA Evo | 4.8 | 13-20 | 24-37% |
| AIRA MCTS | 3.1 | 13 | 24% |
| OE | 3.4 | 13 | 26% |
| UCB | 4.2 | 13 | 32% |

All methods find their best score within the first ~35% of their budget. The remaining budget is spent on failed improvements, suggesting smaller budgets with more restarts may be more efficient.

---

## 8. Recommendations

### When to Use AIRA-Dojo

- **Strong models (GPT-4o, o3):** AIRA's generic prompts are sufficient; the model's intrinsic capability drives performance. The simpler prompt also reduces API costs.
- **Tasks where first-attempt quality matters:** AIRA's 5 independent drafts at depth 1 provide more "lottery tickets" than OE's 2-3 roots with deep chains.
- **House Price-style regression:** Feature engineering + ensemble selection benefits from multiple independent attempts more than iterative refinement.
- **When time/cost is critical:** AIRA with GPT-4o completes in ~5 minutes vs OE's ~3 hours.

### When to Use Open-Ended

- **Weak models (Qwen3-4B):** OE's task-specific strategy descriptions guide weak models toward approaches they wouldn't discover with generic prompts (e.g., MLP+RFE on Titanic).
- **Tasks requiring model diversity:** OE's forced novelty prevents the model from repeating the same approach (e.g., always using RandomForest).
- **When iterative refinement works:** Tasks where depth-2 genuinely improves on depth-1 (e.g., Ridge → XGBoost on House Price).
- **Titanic-style classification:** Where the gap between RF baseline and optimal solution requires exploring non-obvious model families.

### General Recommendations

1. **Increase width, decrease depth:** Both AIRA and OE would benefit from more independent root drafts (e.g., 8-10) and fewer refinement attempts (depth 2-3 max).
2. **Fix MCTS context overflow:** AIRA MCTS wastes 50-80% of budget on cascading context errors with Qwen3-4B. Each node should start from root context.
3. **Add early stopping for refinement:** If 3 consecutive children fail to beat the parent, stop expanding that branch.
4. **Hybrid approach:** Use OE's strategy generation for depth-1 nodes (diverse initial approaches) + AIRA's improve operator for depth-2 (focused refinement). This combines OE's diversity with AIRA's exploitation.
5. **Don't use linear search** for ML tasks -- the degenerate looping problem makes it strictly worse than tree search with the same budget. Only useful for adversarial tasks where creative leaps matter (BOS with o3).

---

## Appendix A: Full Tree Traces

### A.1 Titanic -- o3 AIRA Evo (Best Overall: 0.979)

```
root (d=0, score=0.766)
├── root_0 [draft, XGBoost]           0.859   3 actions
│   └── root_0_0 [crossover, GBM+RF+XGB]  0.883   3 actions
│       └── root_0_0_0 [crossover, Pipeline]  0.876   9 actions
├── root_1 [draft, Pipeline]          0.866   3 actions
├── root_2 [draft, CatBoost]          0.979 *** BEST ***   3 actions
│   ├── root_2_0 [improve, ColumnTrans]  0.840   3 actions
│   ├── root_2_1 [improve, Stacking]     0.878   3 actions
│   │   └── root_2_1_0 [improve, XGB+KFold]  0.845   7 actions
│   └── root_2_2 [improve, CatBoost]     0.842   3 actions
├── root_3 [draft, XGB+GridSearch]    0.871   3 actions
│   └── root_3_0 [improve, CatBoost]     0.847   3 actions
└── root_4 [draft, Pipeline]          0.859   7 actions
```

o3 discovered CatBoostClassifier (with native categorical handling) at depth 1, achieving 97.9% accuracy in just 3 actions. All improvement attempts degraded. The key advantage is o3's model knowledge -- it chose CatBoost specifically because it handles categorical features natively, avoiding the encoding issues that plague other approaches.

### A.2 Titanic -- Qwen3-4B OE (Tied Best: 0.979)

```
root (d=0, score=0.000)
├── root_0 [GBM + interaction terms]           null (FAILED, 16 actions)
├── root_1 [XGBoost + age bins + target enc]   0.914   6 actions
│   ├── root_1_0 [NN + autoencoder + embed]    0.823   9 actions
│   │   └── root_1_0_0 [RF + RFE + target enc]  0.849   3 actions
│   │       └── root_1_0_0_0 [XGBoost + poly]     0.586   7 actions
│   │           └── ... (deep chain oscillating between NN/RF/GBM)
│   └── root_1_1 [MLP + RFE + poly features]  0.979 *** BEST ***   5 actions
│       └── root_1_1_0 [RF + bagging]            0.861   3 actions
└── root_2 [GBM + poly + cross-val]            null (FAILED, 16 actions)
```

OE's explicit strategy "Replace the gradient boosting model with a neural network (e.g., a multi-layer perceptron) and use automated feature selection via recursive feature elimination (RFE) to identify the most predictive combinations" led directly to the MLP+RFE combination that hit 97.9%. Without this specific guidance, Qwen3-4B would never have tried this approach (as shown by AIRA results where it defaulted to RF every time).

### A.3 House Price -- Qwen3-4B AIRA Greedy (Best: 0.921)

```
root (d=0, score=0.880)
├── root_0 [draft]     null (FAILED, 16 actions)
├── root_1 [draft]     0.913   3 actions
├── root_2 [draft]     null (FAILED, 16 actions)
├── root_3 [draft]     0.887   3 actions
└── root_4 [draft]     0.921 *** BEST ***   9 actions
    ├── root_4_0 [improve]  0.885   3 actions
    ├── root_4_1 [improve]  0.910   3 actions
    ├── root_4_2 [improve]  0.886   6 actions
    ├── root_4_3 [improve]  0.882   8 actions
    ├── root_4_4 [improve]  0.913   3 actions
    ├── root_4_5 [improve]  null (FAILED, 16 actions)
    └── root_4_6 [debug]    0.887   3 actions
```

Greedy strategy concentrated 7 children on the best node (root_4), but none improved. root_4 used feature engineering (TotalSF, Age, TotalBath) + CatBoost/XGB/LGBM ensemble -- found through "lucky" sampling, not guided strategy.

### A.4 BOS -- o3 AIRA Greedy (Perfect Score: 2.0)

```
root (d=0, score=1.023)
├── root_0 [draft]     1.443   2 actions
├── root_1 [draft]     1.443   2 actions
│   └── root_1_0 [improve]  1.800   7 actions
│       ├── root_1_0_0 [improve]  1.440   2 actions
│       └── root_1_0_1 [improve]  2.000 *** BEST ***   6 actions
│           ├── root_1_0_1_0 [improve]  1.443   2 actions
│           ├── root_1_0_1_1 [improve]  1.371   2 actions
│           ├── root_1_0_1_2 [improve]  1.800   4 actions
│           └── root_1_0_1_3 [improve]  1.441   2 actions
├── root_2 [draft]     1.438   2 actions
├── root_3 [draft]     1.440   2 actions
└── root_4 [draft]     1.439   3 actions
```

The progression root_1 (1.443) → root_1_0 (1.800) → root_1_0_1 (2.000) shows genuine iterative improvement -- rare across all experiments. The jump from 1.443 to 1.800 and then to 2.000 came from o3 discovering increasingly sophisticated monkey-patching of the opponent's source code. Children of the 2.0 node all regressed because the exploit was already optimal.

### A.5 BOS -- o3 Open-Ended (Best: 1.442)

```
root (d=0, score=1.023)
├── root_0 [Bayesian explore/exploit]           1.435   2 actions
│   └── root_0_0 [Tabular Q-Learner]              0.752   2 actions
├── root_1 [Bayesian Thompson Sampling]         1.442 *** BEST ***   2 actions
│   └── root_1_0 [Markov predictor + Dirichlet]   1.231   2 actions
└── root_2 [Bayesian Best-Response]             1.239   2 actions
    └── root_2_0 [Adaptive Markov]                 1.365   2 actions
        └── root_2_0_0 [Changepoint-Aware]            1.251   2 actions
            └── root_2_0_0_0 [n-Gram Ensemble]           1.441   2 actions
                └── root_2_0_0_0_0 [Periodicity Detector]  1.297   2 actions
                    └── root_2_0_0_0_0_0 [BCP-HBTS]         1.441   3 actions
                        └── root_2_0_0_0_0_0_0 [Expert Hedge]  1.415   2 actions
                            └── root_2_0_0_0_0_0_0_0 [BOCTS]   1.193   2 actions
```

OE generated 10 unique, sophisticated algorithmic strategies (Thompson Sampling, Q-Learning, Markov models, Changepoint detection, n-Gram ensembles, etc.) -- far richer than AIRA's generic prompts. Yet the best score (1.442) is essentially identical to AIRA's depth-1 drafts (~1.44). The forced novelty pushed o3 toward complex algorithms that didn't outperform simple "always play 0" strategies.

Critically, OE **prevented o3 from discovering the monkey-patching exploit** because each node's strategy description constrained the model to design legitimate game-theoretic algorithms. AIRA's generic prompt gave o3 the freedom to think outside the rules.

---

## Appendix B: Score Distributions

### B.1 All Titanic Scores

| Rank | Score | Method | Model | Node |
|------|-------|--------|-------|------|
| 1 | 0.9785 | OE | Qwen3-4B | root_1_1 (d=2) |
| 1 | 0.9785 | AIRA Evo | o3 | root_2 (d=1) |
| 3 | 0.9713 | OE | GPT-4o | root_1_0_0_0_0_0 (d=6) |
| 4 | 0.9593 | OE | o3 | -- |
| 5 | 0.9306 | AIRA Evo | GPT-4o | -- |
| 5 | 0.9306 | UCB | o3 | -- |
| 7 | 0.9234 | AIRA Greedy | GPT-4o | -- |
| 8 | 0.9211 | UCB | GPT-4o | -- |
| 8 | 0.9211 | AIRA MCTS | o3 | -- |
| 10 | 0.9187 | Linear | GPT-4o | -- |
| 11 | 0.8947 | AIRA Greedy | o3 | -- |
| 12 | 0.8923 | AIRA MCTS | GPT-4o | -- |
| 13 | 0.8828 | AIRA Evo | Qwen3-4B | root_4 (d=1) |
| 14 | 0.8732 | AIRA MCTS | Qwen3-4B | root_1_0 (d=2) |
| 15 | 0.8612 | AIRA Greedy | Qwen3-4B | root_0 (d=1) |
| 16 | 0.8517 | Linear | o3 | -- |

### B.2 All BOS Scores (Excluding Exploit)

Without monkey-patching, all methods converge to a narrow band:

| Method | Model | Score |
|--------|-------|-------|
| Linear | GPT-4o | 1.4468 |
| AIRA MCTS | Qwen3-4B | 1.4439 |
| AIRA Evo | GPT-4o | 1.4432 |
| UCB | o3 | 1.4429 |
| AIRA MCTS | o3 | 1.4428 |
| OE | GPT-4o | 1.4428 |
| OE | o3 | 1.4424 |
| AIRA Evo | o3 | 1.4422 |
| OE | Qwen3-4B | 1.4360 |
| AIRA Greedy | GPT-4o | 1.4431 |

Range: 1.436 - 1.447 (spread of only 0.011). The search strategy makes essentially no difference on BOS for conventional approaches.
