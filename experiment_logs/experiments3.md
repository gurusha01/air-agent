# Experiment 3: Adaptive Explore-Exploit Tree Search

## Insights from Experiment 2

### What Worked

1. **Tail VS is the best single mode** (avg rank 1.33 across 3 tasks). Asking the model to propose low-probability, unusual strategies finds non-obvious winners (SVM on titanic, Fourier strategies on battle of sexes).

2. **Verbalized sampling produces real diversity.** Tail VS yields 3/3 to 5/5 distinct winning model families across runs. Without VS, the model converges to GBM on every single run.

3. **Tree search itself is valuable.** Even No VS (temperature-only branching) beats single-trajectory baselines. Having 12 independent shots at the problem is better than 1.

### What Didn't Work

1. **Local VS is crippled by the root.** The root node establishes a baseline approach (typically RF or GBM). Local VS forces all children to stay within the root's model family. Since root → depth-1 → depth-2, the entire tree is trapped in one basin. The intended behavior (diverse siblings, each refined locally) doesn't emerge because all siblings inherit from the same root.

2. **Uniform VS converges to "safe" strategies.** Equal-probability prompting biases toward mainstream approaches (LogReg on titanic, "always choose X" on battle of sexes). These are reliable but not creative.

3. **No VS always converges to GBM.** Temperature=0.9 is not enough to escape the model's default strategy. On houseprice, all 5/5 runs used GBM. The model has a strong prior toward gradient boosting.

4. **Fixed mode for the entire tree is suboptimal.** Tail VS wastes budget on already-solved problems (houseprice where GBM is optimal). Local VS can't escape bad basins. The ideal mode depends on the current state of the search.

### The Core Problem

Experiment 2 uses a **fixed search policy** — every node is expanded with the same mode (tail/uniform/local/none). But the optimal expansion strategy depends on context:

- **Early in search / low scores**: We haven't found anything good yet → **explore** (Tail VS) to cover diverse strategies
- **Found a promising node / high score**: We have a strong approach → **exploit** (Local VS) to refine it (hyperparameter tuning, feature engineering, ensemble variations)
- **Many failed siblings**: The approaches we've tried from this state don't work → inform the LLM about past failures when exploring further
- **High-scoring node**: Could still benefit from exploration (maybe an even better paradigm exists) OR exploitation (squeeze more accuracy from the current approach)

This is the classic **explore-exploit tradeoff**, but applied to tree search over ML strategies.

---

## Experiment 3 Design

### Key Idea: Adaptive Mode Selection

Instead of a fixed mode for the whole tree, **choose explore vs exploit per-node** based on the current search state.

Two decisions at each expansion:
1. **Which node to expand?** (selection policy)
2. **How to expand it?** Explore (Tail/Uniform VS) or Exploit (Local VS)

### Decision 1: Which Node to Expand (Selection Policy)

**The core difficulty:** Decision 2 (explore vs exploit) is relatively straightforward — it's a binary choice based on how good the current node is. Decision 1 is much harder because **the right selection signal depends on whether we're about to explore or exploit.**

- **If exploiting:** child score mean is a natural signal. Expand the node with the highest score and refine it. Clear and simple.
- **If exploring:** score is **misleading**. A node scoring 0.94 (SVM) might be a local optimum with no room for fundamentally different approaches. A node scoring 0.72 (crashed neural net) might have enormous potential — the approach was right but the implementation failed. Score measures "how good is this solution" not "how much unexplored potential exists in this region."

So the question becomes: **what signals indicate high exploration value?**

#### Approaches from the Literature

**A. UCT / MCTS (Monte Carlo Tree Search)**
```
UCB(node) = mean_child_score + C * sqrt(ln(total_visits) / visits(node))
```
- Balances exploitation (high mean reward) with exploration (under-visited nodes)
- Classic approach, well-understood tradeoffs
- **Problem for us:** UCB needs hundreds of visits per node to converge. With only 3 children per node and expensive LLM executions, visit counts are too small for the exploration bonus to be meaningful. UCB was designed for cheap random rollouts in games, not expensive LLM-guided strategy execution.

**B. LATS (Language Agent Tree Search)**
- Most directly relevant prior work — MCTS with LLM agents
- Uses UCT for selection (same small-visit-count problem)
- Key innovation: **LLM self-reflection as backpropagation**. After a node fails, the LLM reflects on *why* it failed, and that reflection is propagated up the tree as context for future expansions
- The reflection means a failed node's future children get context like "NN failed because tensorflow wasn't installed" — avoiding repeated failure modes
- This is relevant to our "past-attempt-aware" idea

**C. AlphaZero's PUCT**
```
PUCT(node) = mean_child_score + C * prior(node) * sqrt(total_visits) / (1 + visits(node))
```
- Adds a **prior probability** from a neural network to guide exploration before any evaluation
- The prior says "this move is probably good" even before trying it
- **Our analog:** use the LLM itself as the prior. Ask it "how promising is this strategy region?" before committing to expand there. Costs one extra LLM call but is informed by domain knowledge.

**D. Quality-Diversity / MAP-Elites**
- Instead of maximizing score, maintain the best solution in each **niche** of behavior space
- Define a strategy taxonomy (tree-based, linear, neural, ensemble, etc.) and ensure coverage across categories before going deep in any one
- **Our analog:** track which strategy families have been tried globally. Prioritize expanding from nodes where the most families remain unexplored.

**E. Progressive Widening**
- Don't expand all children at once. Add one child at a time, only adding more as visit count grows
- Naturally limits over-commitment to any one node
- Tree grows organically — promising regions get more children

#### Practical Signals for Exploration Selection

Given the constraints of our setting (expensive nodes, small branching factor, no cheap rollouts), here are signals that work:

**1. Child variance (information-theoretic)**
```
exploration_value(node) = std(child_scores)
```
Nodes whose children have high score variance are in "interesting" regions — the strategy space around them is rich and sensitive to choices. Nodes with low variance have been saturated.
- Pro: Cheap to compute, meaningful signal
- Con: Needs ≥2 children. Uninformative at expansion time for leaf nodes.

**2. Regret-based**
```
exploration_value(node) = global_best - best_child(node)
```
Expand the node whose best child is furthest below the global best. Intuition: this node's region has the most "room to improve." If a node is already near the global best, exploring from it has lower expected marginal value.
- Pro: Cheap, works with any number of children, naturally shifts attention
- Con: A node might have low regret because it was explored well (not because the region is exhausted)

**3. Under-expansion (visit count proxy)**
```
exploration_value(node) = 1 / (num_children + 1)
```
Prefer nodes that haven't been expanded much yet. They have the most unexplored subtree.
- Pro: Simplest possible signal, no score needed
- Con: Ignores quality entirely — might waste budget on clearly bad regions

**4. LLM-assessed potential**
Ask the LLM: "Given this node used [strategy] and scored [X], how much room for improvement exists if we try fundamentally different approaches? Rate 1-10."
This is the PUCT prior idea — use domain knowledge to estimate exploration value.
- Pro: Most informed signal. The LLM knows "RF at 87% on titanic has room, but GBM at 90.2% on houseprice is near-ceiling"
- Con: Extra LLM call per candidate node. Adds latency and cost.

**5. Strategy coverage gap (MAP-Elites inspired)**
Track which strategy families (tree-based, linear, SVM, neural, ensemble) have been tried from each node. Expand where the most families are untried.
```
exploration_value(node) = count(untried_families_from_node)
```
- Pro: Explicitly maximizes diversity
- Con: Requires defining strategy families, which is fuzzy. Hard to categorize arbitrary LLM output.

#### Recommended Approach: Regret + Under-expansion

For a practical first implementation, combine regret and under-expansion:

```
explore_select(node) = (global_best - best_child(node)) / (num_children + 1)
```

This favors nodes that are (a) far below the global best (high untapped potential) AND (b) haven't been expanded much yet (under-explored). It's cheap — no extra LLM calls — and avoids the UCT small-sample problem.

For exploiting, simply:
```
exploit_select(node) = node.score
```

This gives us a clean two-phase selection:
1. Decide explore or exploit (based on tree state)
2. If exploring → rank by `regret / (children + 1)`, pick top
3. If exploiting → rank by score, pick top

### Decision 2: How to Expand (Explore vs Exploit)

**Heuristic-based:**
- If `node.score < threshold` → Explore (Tail VS): we haven't found a good approach from this state
- If `node.score >= threshold` → Exploit (Local VS): we have something promising, refine it
- Threshold could be relative (e.g., top-k% of all scores so far) or absolute (task-specific)

**Score-variance based:**
- If the node's children have high variance → keep exploring (the space is rich)
- If the node's children have low variance → switch to exploit (we've saturated exploration)

**Budget-decay:**
- Start with explore (Tail VS) at low depths, switch to exploit (Local VS) at high depths
- Simple, doesn't require runtime decisions
- Problem: the optimal switch point varies by task

**LLM-decided:**
- Present the LLM with the current tree state, scores, and strategies tried
- Ask it: "Should we try something fundamentally new (explore) or refine the current approach (exploit)?"
- The LLM understands ML — it might know that "RF at 87% probably can't reach 95%, try SVM"

### Informing the LLM About Past Attempts

Key insight from the user: when choosing to explore from a node, the LLM should know what was already tried and failed. This avoids re-proposing the same strategies.

**Context to provide when exploring:**
```
We've already tried the following approaches from this state:
- Child 1: SVM with RBF kernel → score 0.72 (failed to beat baseline)
- Child 2: XGBoost with target encoding → crashed (import error)
- Child 3: Stacking ensemble (RF+LR+SVM) → score 0.88

Generate NEW strategies that are fundamentally different from all of the above.
Avoid: SVM, XGBoost, stacking ensembles.
```

**Context to provide when exploiting:**
```
The best approach so far from this state:
- Parent: GradientBoosting with default params → score 0.90

Generate variations that REFINE this approach:
- Different hyperparameters (learning_rate, n_estimators, max_depth)
- Feature engineering additions (interactions, polynomial, encoding)
- Preprocessing variations (scaling, imputation strategy)
- Ensemble the parent model with similar models
```

### Proposed Architecture

```
                         ┌─────────────┐
                         │  Root Node   │
                         │  (baseline)  │
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              [Explore]    [Explore]    [Explore]      ← depth 1: always explore
              SVM 0.72     GBM 0.90     NN 0.85
                           (best!)
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              [Exploit]    [Exploit]    [Explore]      ← depth 2: adapt per node
             GBM tuned    GBM+feat    "try LR?"
               0.91         0.93        0.88
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              [Exploit]  [Exploit] [Exploit]           ← depth 3: exploit the best
              GBM+feat   GBM+feat  GBM+feat
              +scaling   +encoding +ensemble
               0.94       0.92      0.95
```

### Differences from Experiment 2

| Aspect | Exp 2 | Exp 3 |
|--------|-------|-------|
| Mode selection | Fixed for entire tree | Adaptive per-node |
| Which node to expand | BFS (all nodes at each depth) | Selection policy (UCB, best-first, etc.) |
| Explore vs exploit | One mode always | Decided per expansion based on scores |
| Past attempt context | Not provided | Failed siblings shown to LLM when exploring |
| Tree shape | Fixed (bf=3, depth=2, BFS) | Variable (expand promising branches deeper) |
| Node budget | Fixed 13 nodes | Fixed budget but allocated adaptively |

---

## Implementation

File: `air/adaptive_tree_search.py` (separate from `air/tree_search.py` to preserve Exp 2 reproducibility).

All experimental knobs are CLI arguments:

```bash
cd /home/ubuntu/MLScientist/MLGym
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py \
    --use-regret --use-depth \
    --context global \
    --node-budget 12 \
    --task-config tasks/titanic.yaml \
    --verbose
```

### CLI Flags

| Flag | Signal | Description |
|------|--------|-------------|
| `--use-variance` | (a) Child variance | Prefer nodes with high score variance among children |
| `--use-regret` | (b) Regret | Prefer nodes whose best child is far below global best |
| `--use-llm-guidance` | (c) LLM prior | Ask model to rate exploration potential (expensive) |
| `--use-coverage` | (d) QD coverage | Prefer nodes with untried strategy families |
| `--use-depth` | (e) Depth + visits | Prefer shallow, under-expanded nodes |
| `--context parent` | Parent only | Child sees only parent conversation |
| `--context global` | Global tree | Child also sees tree summary (what worked/failed) |

Signals are combined additively after min-max normalization. If none are enabled, selection is random (baseline).

---

## Sub-Experiments to Run

### Ablation structure

Each signal is tested independently, then in combination, across both context modes. This gives us clean attribution of which signals matter.

Naming: `exp3.{signal}.{context}` where signal = {rand, var, reg, llm, cov, dep, all} and context = {p, g}.

### Exp 3.0: Random selection baseline (no signals)

No selection signals → random node selection, adaptive explore/exploit.
Tests whether the adaptive explore/exploit decision alone (without smart selection) beats Exp 2.

```bash
# 3.0.p — random selection, parent context
--context parent --node-budget 12

# 3.0.g — random selection, global context
--context global --node-budget 12
```

### Exp 3.1: Individual signal ablations

Each signal alone. Tests the individual contribution of each component.

```bash
# 3.1a — child variance only
--use-variance --context parent
--use-variance --context global

# 3.1b — regret only
--use-regret --context parent
--use-regret --context global

# 3.1c — LLM guidance only
--use-llm-guidance --context parent
--use-llm-guidance --context global

# 3.1d — strategy coverage only (QD)
--use-coverage --context parent
--use-coverage --context global

# 3.1e — depth + visits only
--use-depth --context parent
--use-depth --context global
```

### Exp 3.2: Regret + depth (recommended combination)

The cheapest practical combination. Regret targets under-performing regions; depth prevents over-committing to one branch.

```bash
# 3.2.p
--use-regret --use-depth --context parent

# 3.2.g
--use-regret --use-depth --context global
```

### Exp 3.3: All cheap signals (no LLM guidance)

Variance + regret + coverage + depth. Tests whether combining multiple cheap signals beats any individual one.

```bash
# 3.3.p
--use-variance --use-regret --use-coverage --use-depth --context parent

# 3.3.g
--use-variance --use-regret --use-coverage --use-depth --context global
```

### Exp 3.4: All signals including LLM guidance

The "kitchen sink" — every signal enabled. Tests the marginal value of the expensive LLM prior on top of cheap signals.

```bash
# 3.4.p
--use-variance --use-regret --use-llm-guidance --use-coverage --use-depth --context parent

# 3.4.g
--use-variance --use-regret --use-llm-guidance --use-coverage --use-depth --context global
```

### Experiment matrix summary

| Exp | Var | Reg | LLM | Cov | Dep | Context | Question answered |
|-----|-----|-----|-----|-----|-----|---------|-------------------|
| 3.0 | | | | | | p / g | Does adaptive explore/exploit alone help? |
| 3.1a | x | | | | | p / g | Does child variance help selection? |
| 3.1b | | x | | | | p / g | Does regret help selection? |
| 3.1c | | | x | | | p / g | Does LLM guidance help selection? |
| 3.1d | | | | x | | p / g | Does strategy coverage help selection? |
| 3.1e | | | | | x | p / g | Does depth/visit count help selection? |
| 3.2 | | x | | | x | p / g | Does the cheap recommended combo work? |
| 3.3 | x | x | | x | x | p / g | Do all cheap signals together beat individuals? |
| 3.4 | x | x | x | x | x | p / g | Does adding LLM guidance improve over cheap? |

Total: 9 configs x 2 contexts = 18 experiments per task.
With 5 runs each and 3 tasks = 270 runs (probably start with 1 task first).

---

## Open Questions

1. **What's the right node budget?** Exp 2 used 12 model-executed nodes. Should Exp 3 use the same budget (to compare fairly) or more (since adaptive allocation should use budget more efficiently)?

2. **How to set the explore/exploit threshold?** Relative (top-k% of all scores so far) or absolute (task-specific)? Relative is more general but requires seeing enough scores first. Variance-based switching (high child variance → keep exploring) is another option.

3. **Should the tree be balanced or lopsided?** BFS guarantees breadth. Adaptive selection might produce degenerate trees (one branch goes to depth 10, others abandoned at depth 1). Is that bad? Probably fine if the budget is fixed — the question is whether the budget is better spent deep or wide.

4. **Does past-attempt context actually help?** Siblings have separate conversation histories — they don't know what each other tried. Explicit context about failed siblings should help avoid redundant strategies. This is essentially the LATS "reflection-as-backpropagation" idea.

5. **Regret-based vs LLM-guided selection?** Regret-based is cheap and principled but ignores domain knowledge (a node at 0.87 with RF and a node at 0.87 with NN have same regret but very different potential). LLM-guided is informed but expensive. Could start with regret-based (Exp 3.3) and add LLM prior on top (Exp 3.4) to measure the marginal value of domain knowledge.

6. **How to handle the root expansion?** Depth-1 nodes always branch from baseline (no parent strategy to refine). Should depth-1 always be explore? Or should we allow exploit at depth-1 if the baseline approach is already strong (e.g., GBM on houseprice)?

---

## Results: Titanic (single run per config)

### Summary Table

| Exp | Signals | Context | Best | Avg | Scored/12 | Max Depth | Time |
|-----|---------|---------|------|-----|-----------|-----------|------|
| 3.0.p | random | parent | 0.8947 | 0.8864 | 6 | 1 | 1194s |
| 3.0.g | random | global | 0.8756 | 0.8056 | 4 | 1 | 1417s |
| 3.1a.p | variance | parent | 0.9211 | 0.8362 | 11 | 3 | 906s |
| 3.1a.g | variance | global | 0.8876 | 0.7805 | 8 | 2 | 1211s |
| **3.1b.p** | **regret** | **parent** | **0.9282** | **0.8469** | **11** | **4** | **1165s** |
| **3.1b.g** | **regret** | **global** | **0.9378** | **0.8278** | **9** | **4** | **1355s** |
| 3.1d.p | coverage | parent | 0.9234 | 0.7435 | 9 | 9 | 1282s |
| 3.1d.g | coverage | global | 0.9234 | 0.8657 | 8 | 6 | 1215s |
| 3.1c.p | llm-guidance | parent | 0.9234 | 0.8823 | 11 | 4 | 906s |
| 3.1c.g | llm-guidance | global | 0.9115 | 0.8223 | 7 | 2 | 1272s |
| 3.1e.p | depth | parent | 0.8923 | 0.8257 | 7 | 1 | 1255s |
| 3.1e.g | depth | global | 0.8589 | 0.8263 | 5 | 1 | 1226s |
| **3.2.p** | **regret+depth** | **parent** | **0.9450** | **0.8919** | **5** | **1** | **1333s** |
| 3.2.g | regret+depth | global | 0.8756 | 0.8033 | 5 | 1 | 1330s |
| 3.3.p | all cheap | parent | 0.9211 | 0.7714 | 9 | 6 | 1128s |
| 3.3.g | all cheap | global | 0.8804 | 0.7857 | 12 | 6 | 839s |

**Best score: 3.2.p (regret+depth, parent) = 0.9450**

Reference from Exp 2: Tail VS best = ~0.89, No VS best = ~0.88

### Ranking by Best Score

1. **3.2.p** (regret+depth, parent) — 0.9450
2. **3.1b.g** (regret, global) — 0.9378
3. **3.1b.p** (regret, parent) — 0.9282
4. **3.1c.p** (llm-guidance, parent) — 0.9234
5. **3.1d.p** (coverage, parent) — 0.9234
6. **3.1d.g** (coverage, global) — 0.9234
7. **3.1a.p** (variance, parent) — 0.9211
8. **3.3.p** (all cheap, parent) — 0.9211
9. **3.1c.g** (llm-guidance, global) — 0.9115
10. **3.0.p** (random, parent) — 0.8947
11. **3.1e.p** (depth, parent) — 0.8923
12. **3.1a.g** (variance, global) — 0.8876
13. **3.3.g** (all cheap, global) — 0.8804
14. **3.0.g** (random, global) — 0.8756
15. **3.2.g** (regret+depth, global) — 0.8756
16. **3.1e.g** (depth, global) — 0.8589

### Key Observations

#### 1. Regret is the strongest individual signal

Regret alone (3.1b) outperforms all other single signals by a clear margin. Both parent (0.9282) and global (0.9378) variants of regret beat all other configurations except the regret+depth combo. This confirms the hypothesis: expanding nodes that are furthest below the global best focuses search budget on regions with the most room for improvement.

#### 2. Regret+depth is the best combination

The recommended cheap combo (3.2.p = 0.9450) is the overall winner. Regret targets under-performing regions; depth prevents over-committing to one deep branch. The synergy is clear — regret alone drives deep into promising branches (depth 4), but adding depth preference keeps the search balanced.

However, this only holds for parent context. The global variant (3.2.g = 0.8756) performed poorly, suggesting the global context confuses the model when combined with depth-based balancing.

#### 3. Parent context generally outperforms global context

Of the 7 signal configurations, parent context wins in 5 out of 7:
- Random: parent 0.8947 vs global 0.8756
- Variance: parent 0.9211 vs global 0.8876
- Coverage: tie (0.9234 vs 0.9234)
- Depth: parent 0.8923 vs global 0.8589
- Regret+depth: parent 0.9450 vs global 0.8756
- All cheap: parent 0.9211 vs global 0.8804

The one exception: **regret alone** — global (0.9378) > parent (0.9282). When using regret to guide selection, the global tree context helps the model see what's been tried and propose something genuinely different. But for most other signals, the global context is noise that distracts the model.

#### 4. Depth signal alone is useless (or harmful)

Depth+visit count (3.1e) barely beats random baseline and is the weakest individual signal. It favors shallow/unexpanded nodes regardless of their quality, wasting budget on unpromising breadth expansion. Depth only helps as a **regularizer** combined with regret (3.2).

#### 5. Coverage drives deep but inconsistent trees

Coverage/QD (3.1d) produces the deepest trees (max depth 9 for parent), as it keeps pushing to try untried strategy families at each node. The scores are good (0.9234) but the average across all nodes is low (0.7435), meaning many nodes along the way fail. The signal successfully diversifies but doesn't efficiently focus on promising regions.

#### 6. All-signals-combined doesn't win

Combining all cheap signals (3.3) doesn't beat the best individual signals. The score (0.9211) equals variance alone, and is well below regret alone (0.9282) or regret+depth (0.9450). When all signals are combined with equal weight, the strong signals (regret) get diluted by weak ones (depth). Signal selection matters more than signal aggregation.

#### 7. LLM guidance is mid-tier — not worth the extra cost

LLM guidance (3.1c.p = 0.9234) ranks 4th, tied with coverage. It's better than random but clearly worse than regret (0.9282) and regret+depth (0.9450). Given that LLM guidance requires an extra LLM inference call per node selection step (asking the model to rate each candidate's exploration potential), the marginal improvement over cheap signals doesn't justify the cost.

Interestingly, LLM guidance with parent context has the highest average score (0.8823) of any signal — it produces consistently good nodes. But its best score is lower than regret's, suggesting it's "safe" but not bold enough in its selection choices.

#### 9. Failure rate correlates with tree depth

Experiments with more failed nodes (score=null, hit max_actions without validate):
- Shallow trees (depth 1): 5-8 failures out of 12
- Deep trees (depth 4-9): 1-4 failures out of 12

This makes sense: deep trees inherit working code from their parent. The root's initial 3 children (Tail VS) have the highest failure rate because they start from scratch.

#### 10. Tree structure analysis for top experiments

**3.2.p (regret+depth, best=0.9450):** Flat tree (depth 1). All 12 nodes are direct children of root. Only 5 scored successfully. The winning node (root_10, 0.9450) was simply a well-engineered direct attempt. The regret+depth combo kept trying new breadth-first approaches rather than going deep.

**3.1b.g (regret, global, best=0.9378):** Deep tree (depth 4). Shows interesting strategy evolution:
```
root → GBM (0.83) → Neural Net (0.82) → Stacked Autoencoder (0.94!)
```
The regret signal kept expanding the weakest branch (GBM at 0.83), which led to NN, then to a creative stacked autoencoder that achieved 0.9378.

**3.1b.p (regret, parent, best=0.9282):** Deep tree (depth 4). Similar evolution:
```
root → Random Forest (0.85) → GBM (0.85) → NN embedding (0.93!)
```
The regret signal forced exploration into neural approaches, eventually finding a NN with embeddings that scored 0.9282.

### Preliminary Conclusions

1. **Adaptive tree search with selection signals outperforms Exp 2's fixed BFS.** Best Exp 3 score (0.9450) vs best Exp 2 Tail VS (~0.89 mean). Single-run comparison, but +5.5% is a strong directional signal.

2. **Regret is the key signal.** It appears in all top-3 configurations. It focuses budget on under-performing regions where improvements are most likely.

3. **Parent context is the safer default.** Global context only helps with regret (where knowing the full tree prevents revisiting failed approaches). For other signals, it adds noise.

4. **More signals ≠ better.** Regret alone or regret+depth outperforms the kitchen sink. Signal selection should be parsimonious.

### Next Steps

- [x] Run LLM guidance (3.1c.p/g) — Result: mid-tier (0.9234 / 0.9115), not worth the extra cost
- [x] Investigate why 3.2.p stays flat (depth 1) → root-selection bug (see v2 below)
- [ ] Multi-run (5x) for top configs for statistical significance
- [ ] Extend to other tasks (houseprice, battleofsexes)
- [ ] Consider signal weighting — currently all enabled signals get equal weight after normalization

---

## Results v2: After Root-Selection Bug Fix

### Bug Fix

**Root cause:** Root node (score=0.0, depth=0) was always the top candidate for signals
using regret (regret = global_best - 0.0 = max) or depth (bonus = 1/(0+1) = 1.0). This
caused regret+depth to always expand from root → flat depth-1 GBM monoculture.

**Fix:** Exclude root from candidates once it has >= `initial_breadth` children, unless
no other valid candidates exist (edge case: all initial children fail).

**LLM guidance change:** Prompt now asks for two axes:
- **Interestingness** (0-1): Is this direction novel? Are there unexplored approach families?
- **Depth potential** (0-1): Is this approach promising enough for in-depth refinement?

The LLM's recommendation also drives the explore/exploit decision: if interestingness > depth_potential → explore (Tail VS), else → exploit (Local VS). This replaces the simple percentile threshold when --use-llm-guidance is enabled.

### v1 vs v2 Comparison

| Experiment | v1 Best | v1 Depth | v1 #Fam | v2 Best | v2 Depth | v2 #Fam | Delta |
|------------|---------|----------|---------|---------|----------|---------|-------|
| regret, parent | 0.9282 | 4 | 4 | 0.8947 | 4 | 4 | -0.034 |
| regret, global | 0.9378 | 4 | 5 | 0.9306 | 5 | 6 | -0.007 |
| llm-guid, parent | 0.9234 | 4 | 3 | 0.8708 | 3 | 5 | -0.053 |
| llm-guid, global | 0.9115 | 2 | 3 | 0.8995 | 3 | 3 | -0.012 |
| depth, parent | 0.8923 | 1 | 2 | 0.8971 | 3 | 4 | +0.005 |
| depth, global | 0.8589 | 1 | 2 | 0.8828 | 4 | 4 | +0.024 |
| **regret+depth, parent** | **0.9450** | **1** | **1** | **0.9282** | **5** | **5** | **-0.017** |
| **regret+depth, global** | **0.8756** | **1** | **1** | **0.9306** | **5** | **3** | **+0.055** |
| all cheap, parent | 0.9211 | 6 | 3 | 0.8325 | 5 | 4 | -0.089 |
| all cheap, global | 0.8804 | 6 | 5 | 0.8804 | 6 | 4 | 0.000 |

### Key Observations (v2)

#### 1. Root fix eliminates flat-tree bug

All experiments now explore depth ≥ 3 (vs depth 1 in v1 for regret+depth and depth-only).
The regret+depth combo (3.2v2) now produces depth-5 trees with 3-5 model families instead
of flat GBM monoculture. This confirms the v1 results were artifacts of root domination.

#### 2. Regret+depth global context massively improved

3.2v2.g jumped from 0.8756 → 0.9306 (+5.5%). This was the most broken experiment in v1
(root always selected, global context added noise to an already-broken search). With the
fix, global context now matches parent context (0.9306 vs 0.9282).

#### 3. Single-run scores are noisy

Regret-parent dropped from 0.9282 → 0.8947 despite no code changes to regret itself (only
the root exclusion). This ±3% variance suggests single-run comparisons are unreliable.
Need multi-run experiments for real conclusions.

#### 4. Regret+depth now produces evolutionary strategy chains

3.2v2.p tree shows genuine strategy evolution:
```
root → GBM (0.93) → feature eng (0.90) → GBM variant (0.86) → Neural (0.86) → deeper...
```
3.2v2.g similarly: root → GBM (0.93) → feature eng (0.93) → Neural (0.87) → deeper

This is qualitatively different from v1's flat "try 12 GBMs" approach.

#### 5. LLM guidance with interestingness/depth prompt shows more diversity but lower scores

3.1cv2.p went from 3 → 5 families (more diverse) but best score dropped 0.9234 → 0.8708.
The LLM might be over-exploring — rating too many nodes as "interesting" and not enough as
"worth exploiting." The interestingness prompt may need calibration.

#### 6. All-cheap-signals remains weak

3.3v2.p dropped to 0.8325 (worst v2 result). Combining all signals with equal weight
dilutes the useful ones. This confirms v1's finding: more signals ≠ better.

### Updated Ranking (v2 only, by best score)

1. **3.1bv2.g** (regret, global) — 0.9306
2. **3.2v2.g** (regret+depth, global) — 0.9306
3. **3.2v2.p** (regret+depth, parent) — 0.9282
4. **3.1cv2.g** (llm-guidance, global) — 0.8995
5. **3.1ev2.p** (depth, parent) — 0.8971
6. **3.1bv2.p** (regret, parent) — 0.8947
7. **3.1ev2.g** (depth, global) — 0.8828
8. **3.3v2.g** (all cheap, global) — 0.8804
9. **3.1cv2.p** (llm-guidance, parent) — 0.8708
10. **3.3v2.p** (all cheap, parent) — 0.8325

### Overall v2 Conclusions

1. **Regret remains the strongest signal**, but regret+depth is now competitive (not inflated by root bug).
2. **Global context is now viable** — v1's parent > global pattern was partly a root-bug artifact. With the fix, global context performs equally or better.
3. **Single-run variance is too high** (±3-5%) for reliable signal attribution. Need 5-run experiments.
4. **The root exclusion fix is critical** — without it, depth-based signals are useless.
5. **LLM interestingness prompt needs calibration** — more diverse but lower scores suggests over-exploration.

---

## Results v3: GPT-Guided LLM Guidance

### Motivation

v2 showed that local Qwen3-4B as the LLM guidance model has a systematic exploit bias: it always
rates depth_potential > interestingness → 100% exploit mode → pure GBM hyperparameter tuning.
Three bugs were found and fixed:
1. `_llm_mode_hint` stored on `SelectionPolicy` but checked on `AdaptiveTreeSearch` (dead code)
2. Sticky LLM scores (same node always rated highest) → fixed with child-count decay
3. Systematic exploit bias from small local model → **solution: use frontier model for guidance**

Added `--guidance-model` CLI flag + `GuidanceClient` class (uses OpenAI API). The local Qwen
model still executes ML strategies; the frontier model only rates node interestingness/depth
potential for selection decisions.

### Experiments

| Exp | Guidance Model | Context | Best | Depth | Nodes | Time | Families |
|-----|---------------|---------|------|-------|-------|------|----------|
| 3.1c_gpt4omini.p | gpt-4o-mini | parent | 0.8971 | 5 | 13 | 956s | 5 (GBM, RF, NN, SAE, Other) |
| 3.1c_gpt4omini.g | gpt-4o-mini | global | **0.9115** | 5 | 13 | 733s | 2 (GBM, Other) |
| 3.1c_gpt52.p | gpt-5.2 | parent | 0.8971 | 2 | 13 | 1650s | 2 (GBM, NN) |
| 3.1c_gpt52.g | gpt-5.2 | global | 0.8995 | 2 | 13 | 909s | 2 (GBM, NN) |

For reference, v2 Qwen-guided LLM guidance:
- 3.1cv2.p (Qwen): 0.8708, depth 3, 5 families
- 3.1cv2.g (Qwen): 0.8995, depth 3, 3 families

### Tree Structures

**GPT-4o-mini parent (3.1c_gpt4omini.p):**
```
root [0.00]
├── root_0 [0.64] GBM + categorical features
│   └── root_0_0 [0.64] → Neural network
│       └── root_0_0_0 [0.64] → Random forest
├── root_1 [0.90] GBM + target encoding  ← best
│   └── root_1_0 [None] Refine target encoding (failed)
└── root_2 [0.85] GBM + feature eng
    └── root_2_0 [0.87] → Neural network (MLP)
        ├── root_2_0_0 [0.85] → Random forest
        │   └── root_2_0_0_0 [0.89] → GBM + polynomial
        │       └── root_2_0_0_0_0 [0.87] Hybrid features
        └── root_2_0_1 [0.85] MLP + residual
            └── root_2_0_1_0 [0.78] Stacked autoencoder
```
Good diversity (5 families) but best score only 0.8971. The explore-heavy tree produces many
mediocre nodes. The root_0 branch is stuck at 0.64 for 3 levels — the guidance model keeps
selecting it as "interesting" even though it can't improve.

**GPT-4o-mini global (3.1c_gpt4omini.g):**
```
root [0.00]
├── root_0 [None] (failed)
├── root_1 [0.91] GBM + feature engineering
│   └── root_1_0 [0.91] Weighted stratified CV
│       ├── root_1_0_0 [0.91] Target encoding for Embarked
│       │   └── root_1_0_0_0 [0.89] GBM + feature interactions
│       ├── root_1_0_1 [0.91] Box-Cox transformation
│       └── root_1_0_2 [0.90] Log transform Age
└── root_2 [0.91] GBM + feature engineering
    └── root_2_0 [0.91] Weighted median imputation
        └── root_2_0_0 [0.9115] Dynamic weighted median  ← best
            └── root_2_0_0_0 [0.91] Enhanced imputation
                └── root_2_0_0_0_0 [0.91] Sigmoid transform
```
Best GPT-guided result. Very consistent exploitation — scores tightly clustered 0.90-0.9115.
The global context helps the model refine incrementally. Low diversity (all GBM variants) but
effective steady improvement. Essentially pure exploit mode driven by GPT-4o-mini's guidance.

**GPT-5.2 parent (3.1c_gpt52.p):**
```
root [0.00]
├── root_0 [0.89] GBM + feature engineering
│   ├── root_0_0 [0.46] → Neural network (collapsed)
│   ├── root_0_1 [None] → Neural network (failed)
│   ├── root_0_2 [None] → Neural network (failed)
│   ├── root_0_3 [0.90] → Neural network  ← best
│   ├── root_0_4 [None] → Neural network (failed)
│   ├── root_0_5 [None] → Neural network (failed)
│   ├── root_0_6 [0.63] → Neural network
│   ├── root_0_7 [0.88] → Neural network
│   └── root_0_8 [None] → Neural network (failed)
├── root_1 [0.64] GBM + target encoding
└── root_2 [0.88] GBM + target encoding
```
**Pathological behavior.** GPT-5.2 always selects root_0 and always suggests "replace with
neural network." Result: 9 NN children of the same parent, 5/9 failed (None), tree stuck at
depth 2. GPT-5.2's interestingness assessment is essentially: "GBM is boring, NN is interesting"
repeated 9 times. Slowest experiment (1650s) due to GPT-5.2 API latency.

**GPT-5.2 global (3.1c_gpt52.g):**
```
root [0.00]
├── root_0 [0.83] GBM + feature engineering
│   ├── root_0_0 [0.87] Interaction features
│   ├── root_0_1 [0.66] → Neural network
│   ├── root_0_2 [0.82] → Neural network
│   ├── root_0_3 [0.90] → Neural network  ← best
│   ├── root_0_4 [0.88] → Neural network
│   ├── root_0_5 [0.85] → Neural network
│   ├── root_0_6 [0.90] → Neural network
│   ├── root_0_7 [None] → Neural network (failed)
│   └── root_0_8 [0.65] → Neural network
├── root_1 [None] (failed)
└── root_2 [None] (failed)
```
Same pattern: GPT-5.2 fixates on root_0, spams NN variants. With global context, fewer failures
(1/9 vs 5/9) and slightly better scores, but still depth-2 with zero strategic diversity.

### Key Findings

#### 1. GPT-5.2 is a poor guidance model for tree search

Despite being the more capable model, GPT-5.2 produces the worst tree search behavior:
- **Sticky node selection**: Always picks the same node (root_0) as most "interesting"
- **Repetitive strategy suggestions**: Always suggests "replace with neural network"
- **Shallow trees**: Depth 2 (vs depth 5 for GPT-4o-mini)
- **High failure rate**: 5/9 children failed in parent context
- **Slowest**: 1650s for parent context (vs 956s for GPT-4o-mini)

The likely explanation: GPT-5.2 has strong opinions about what's "interesting" and those opinions
are consistent — it always rates the same node highest and always suggests the same alternative.
This is the opposite of what we need for diverse exploration.

#### 2. GPT-4o-mini is the better guidance model

GPT-4o-mini produces more balanced trees:
- Deeper exploration (depth 5)
- More diverse node selection (spreads budget across branches)
- With global context, achieves the best GPT-guided score (0.9115)
- Faster (733-956s vs 909-1650s for GPT-5.2)

GPT-4o-mini's less confident assessments lead to more varied selection, which is actually
better for search diversity.

#### 3. Global context helps GPT-guided search

For both models, global context produces better or equal scores:
- GPT-4o-mini: global 0.9115 > parent 0.8971
- GPT-5.2: global 0.8995 ≥ parent 0.8971

When the guidance model can see the full tree, it makes more informed selection decisions.

#### 4. Frontier model guidance still doesn't beat cheap signals

| Method | Best Score | Cost |
|--------|-----------|------|
| Regret+depth (v2, free) | 0.9306 | $0 |
| Regret alone (v2, free) | 0.9306 | $0 |
| GPT-4o-mini guidance | 0.9115 | ~$0.01 |
| GPT-5.2 guidance | 0.8995 | ~$0.10 |
| Qwen guidance (local) | 0.8995 | $0 |

The simple regret-based signal (free, no API calls) outperforms all LLM guidance variants
by 2-3%. This suggests that **the explore/exploit decision and node selection are better
solved by mathematical signals** (regret = how far below global best) than by LLM judgment
(which model family is more "interesting").

### Updated Overall Ranking (all versions)

| Rank | Experiment | Best | Version | Notes |
|------|-----------|------|---------|-------|
| 1 | 3.2.p (regret+depth) | 0.9450 | v1 | Root bug inflated (flat tree) |
| 2 | 3.1b.g (regret) | 0.9378 | v1 | Genuine deep search |
| 3 | 3.2v2.g (regret+depth) | 0.9306 | v2 | Root fix, deep tree |
| 4 | 3.1bv2.g (regret) | 0.9306 | v2 | Root fix, deep tree |
| 5 | 3.2v2.p (regret+depth) | 0.9282 | v2 | Root fix, deep tree |
| 6 | 3.1b.p (regret) | 0.9282 | v1 | Deep NN evolution chain |
| 7 | 3.1c_gpt4omini.g | 0.9115 | v3 | Best GPT-guided |
| 8 | 3.1c_gpt52.g | 0.8995 | v3 | Shallow NN-spam tree |
| 9 | 3.1cv2.g (Qwen LLM) | 0.8995 | v2 | Local model guidance |
| 10 | 3.1c_gpt4omini.p | 0.8971 | v3 | Most family diversity (5) |
| 11 | 3.1c_gpt52.p | 0.8971 | v3 | Depth 2, 5 NN failures |

### Conclusions

1. **LLM guidance (whether local or frontier) is consistently outperformed by regret-based
   selection.** Regret is cheap, principled, and produces better results.

2. **Smaller guidance models > larger ones for search diversity.** GPT-4o-mini's less confident
   assessments lead to more varied node selection, which is better for tree search. GPT-5.2's
   strong opinions create pathological fixation on single nodes.

3. **The explore/exploit decision should be signal-based, not LLM-based.** Mathematical
   heuristics (regret, depth) make better selection decisions than "interestingness" ratings.

4. **Regret+depth remains the best approach** (v2 scores: 0.9282-0.9306), and it's free.

---

## Results v4: UCB and Open-Ended Selection Strategies

### Motivation

v1-v3 all used the same selection architecture: independently toggleable signals, each
normalized to [0,1] and summed with equal weight. This has three problems:

1. **Min-max normalization + equal weighting dilutes strong signals.** When regret is combined
   with depth, each contributes ~0.5 per candidate after normalization. If 5 candidates all
   have similar regret but different visit counts, the depth signal dominates. Signals fight
   each other instead of cooperating.

2. **No visit-count decay on selection.** The GPT-5.2 pathology (root_0 selected 9 times) and
   the root-selection bug both stem from the same issue: nothing penalizes repeatedly selecting
   the same node. The depth signal has `1/(depth*(n_children+1)+1)` but it's easily overridden.

3. **Selection and explore/exploit are decoupled.** The signal policy picks a node, then
   `_decide_mode()` independently decides explore vs exploit using a percentile threshold.
   These should be coupled — a node with high regret AND low visits should be explored;
   a node with low regret AND high score should be exploited.

v4 introduces two new `--selection-strategy` options that replace the signal-based approach
with principled MCTS formulas.

### Strategy 1: Vanilla UCB (`--selection-strategy ucb`)

Classic UCB1 (Upper Confidence Bound) from the MCTS literature, adapted for our setting.

**Selection formula:**

```
score(node) = Q(node) + C × √(ln(N) / (n_children + 1))
               ‾‾‾‾‾‾   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
              exploitation      exploration bonus
```

Where:
- **Q(node)** = best_child_score / global_best (normalized exploitation value; 1.0 means
  this node's best child equals the global best)
- **C** = exploration constant (default √2 ≈ 1.41; higher C = more exploration)
- **N** = total node expansions so far across the tree
- **n_children** = number of times this node has been expanded (= len(children))

**Explore/exploit coupling:**
After selecting a node, the mode is determined by which term dominates:
- If exploration_bonus > Q → **explore** (Tail VS: "try fundamentally different approach")
- If Q > exploration_bonus → **exploit** (Local VS: "refine current approach")

This couples selection and mode into a single formula. A node with few children has high
exploration bonus → UCB picks it AND explores from it. A node near the global best with
many children → Q dominates → UCB picks it AND exploits from it.

**Key property:** Visit-count decay is built-in. After expanding a node k times, its
exploration bonus drops as `1/√(k+1)`. This prevents the GPT-5.2 pathology (9 expansions
of the same node) and the root-selection bug (root always wins) without needing special-case
exclusion logic.

**Limitations:** UCB minimizes cumulative regret — it converges to the best node. In
scientific exploration, sometimes you need to go through a "valley" to reach a higher peak.
A path like 0.8 → 0.6 → 0.66 → 0.7 → 0.98 would be abandoned by UCB at the 0.6 step
because Q is low and the exploration bonus decays after 2-3 children. UCB doesn't do
open-ended exploration.

### Strategy 2: Open-Ended Exploration (`--selection-strategy open-ended`)

UCB augmented with two mechanisms for open-ended scientific exploration.

**Core insight:** In scientific discovery, you sometimes need to go through a "valley" to
reach a higher peak on the other side. A search trajectory like:

```
0.80 → 0.60 → 0.66 → 0.70 → 0.50 → 0.68 → 0.98
```

would be abandoned by vanilla UCB at the 0.60 step (low Q, exploration bonus decays).
But the direction was promising — it just needed more commitment. Open-ended exploration
adds two mechanisms to prevent premature abandonment:

#### Mechanism 1: Trend Bonus

Instead of only looking at absolute score, measure the **improvement trajectory** along
the path from root to each node.

```
score(node) = Q(node) + C × √(ln(N)/(n+1)) + W_trend × trend(node)
                                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                                 new: trend bonus
```

Where `trend(node)` = recency-weighted average of score deltas along the root→node path:

```
path = [score_root, score_d1, score_d2, ..., score_node]
deltas = [path[i] - path[i-1] for i in 1..len(path)]
trend = Σ(delta_i × weight_i) / Σ(weight_i)    where weight_i = i (linear recency)
```

Only positive trends contribute (negative trends are clamped to 0).

**Example:** A path [0.60, 0.62, 0.66, 0.70] has trend ≈ +0.04 (improving).
A path [0.80, 0.81, 0.80, 0.79] has trend ≈ -0.004 (stagnating → clamped to 0).
The first path gets a bonus even though its absolute score (0.70) is lower than the
second path's (0.79). This keeps the search committed to improving directions.

**Parameters:**
- `--trend-weight` (default 0.5): How much the trend bonus matters vs UCB terms.
  Higher = more willingness to follow improving-but-low-scoring branches.

#### Mechanism 2: Path Commitment

When a branch is selected and its latest expansion showed improvement (child score >
parent score), **commit** to continuing that branch for the next K expansions without
going back to global selection. This gives "valleys" time to recover.

**Rules:**
1. After selecting a node via UCB+trend, if its branch shows positive trend → start
   commitment. Mark the branch as "committed."
2. On subsequent iterations, if committed: find the latest leaf of the committed branch
   and expand it directly (skip global selection). Always use **explore** mode during
   commitment (looking for breakthroughs).
3. Commitment continues as long as the latest child improved over its parent, up to
   a maximum of K steps.
4. Commitment breaks when: (a) the latest child is worse than its parent, or
   (b) K committed steps have been taken.

**Parameters:**
- `--commitment-threshold` (default 2): Maximum committed steps before re-evaluating.
  Higher = more patience with valleys. Lower = quicker to abandon unpromising directions.

**Example with K=2:**
```
Step 1: UCB selects node A (trend positive) → start commitment to A
Step 2: [COMMITTED] Expand A's latest leaf → child improves → continue
Step 3: [COMMITTED] Expand again → child degrades → commitment breaks
Step 4: Back to global UCB selection
```

### Experiment Configuration

14 experiments total: 5 UCB variants × 2 contexts + 4 open-ended variants + 3 baselines.

| Exp | Strategy | C | Context | Trend W | Commit K | Question |
|-----|----------|---|---------|---------|----------|----------|
| ucb_c1.0.p | ucb | 1.0 | parent | — | — | Less exploration |
| ucb_c1.41.p | ucb | 1.41 | parent | — | — | Standard UCB |
| ucb_c2.0.p | ucb | 2.0 | parent | — | — | More exploration |
| ucb_c1.41.g | ucb | 1.41 | global | — | — | UCB + global context |
| ucb_c2.0.g | ucb | 2.0 | global | — | — | High explore + global |
| oe_t0.3_k2.p | open-ended | 1.41 | parent | 0.3 | 2 | Low trend weight |
| oe_t0.5_k2.p | open-ended | 1.41 | parent | 0.5 | 2 | Default open-ended |
| oe_t0.5_k3.p | open-ended | 1.41 | parent | 0.5 | 3 | More commitment |
| oe_t1.0_k2.p | open-ended | 1.41 | parent | 1.0 | 2 | High trend weight |
| oe_t0.5_k2.g | open-ended | 1.41 | global | 0.5 | 2 | Open-ended + global |
| oe_t1.0_k2.g | open-ended | 1.41 | global | 1.0 | 2 | High trend + global |
| regret.p | signals | — | parent | — | — | Baseline: regret only |
| regret.g | signals | — | global | — | — | Baseline: regret + global |
| regret_depth.g | signals | — | global | — | — | Baseline: regret+depth |

### Results

*(Experiments running — results will be filled in as they complete)*

| Rank | Exp | Best | Depth | Nodes | Time | Notable |
|------|-----|------|-------|-------|------|---------|
| 1 | **oe_t0.3_k2.p** | **0.9785** | 8 | 13 | 1088s | NEW ALL-TIME BEST. NN breakthrough at d2. |
| 2 | **oe_t1.0_k2.g** | **0.9641** | 4 | 13 | 1004s | NN replacement at d2. High trend + global. |
| 3 | **oe_t0.5_k3.p** | **0.9569** | 5 | 13 | 1498s | 4/5 init children FAILED → transformer at d3. |
| 4 | regret.g | 0.9474 | 5 | 13 | 768s | Regret baseline — strong but below open-ended. |
| 5 | ucb_c1.0.p | 0.9426 | 8 | 13 | 1178s | Valley crossing: 0.86→0.65→0.62→0.94! |
| 6 | ucb_c1.41.g | 0.9402 | 6 | 13 | 1158s | Deep chain: d6 node found best. |
| 7 | ucb_c1.41.p | 0.9019 | 5 | 13 | 1293s | GBM→NN→RF→GBM cycling, no breakthrough. |
| 8 | ucb_c2.0.g | 0.8876 | 7 | 13 | 1198s | Over-explores but global context helps. |
| 9 | ucb_c2.0.p | 0.8804 | 5 | 13 | 1197s | Over-explores: GNN (0.56), LSTM attempts. |
| 10 | oe_t0.5_k2.p | 0.8565 | 7 | 13 | 1134s | Default open-ended — middling. |
| 11 | regret.p | 0.8541 | 4 | 13 | 1191s | Regret parent — weaker than global. |
| 12 | oe_t0.5_k2.g | 0.8517 | 6 | 13 | 980s | Default open-ended + global — middling. |
| 13 | oe_t1.0_k2.p | 0.8086 | 5 | 13 | 1328s | High trend weight parent — over-committed. |
| 14 | regret_depth.g | 0.6220 | 4 | 13 | 856s | Regret+depth — worst result (possible bug?). |

### Full Analysis: Titanic v4

#### 1. Open-ended exploration is the new best strategy

The top 3 results are ALL from the open-ended strategy:
- oe_t0.3_k2.p: **0.9785** (all-time best across ALL experiments)
- oe_t1.0_k2.g: **0.9641**
- oe_t0.5_k3.p: **0.9569**

These beat both vanilla UCB (best 0.9426) and the regret baselines (best 0.9474).
The trend bonus + path commitment mechanisms successfully produce higher-scoring trees.

#### 2. Trend weight sweet spot: lower is better for parent, higher for global

| Trend weight | Parent | Global |
|-------------|--------|--------|
| 0.3 | **0.9785** | — |
| 0.5 (k=2) | 0.8565 | 0.8517 |
| 0.5 (k=3) | 0.9569 | — |
| 1.0 | 0.8086 | **0.9641** |

For parent context, low trend weight (0.3) works best — just enough to keep improving
branches alive without over-committing to every positive delta. High trend (1.0) with
parent context over-commits and gets stuck in valleys.

For global context, high trend weight (1.0) works best — the global tree summary
provides enough information that strong trend commitment leads to focused exploitation
rather than blind valley-following.

#### 3. Commitment threshold K=3 helps

oe_t0.5_k3.p (0.9569) vs oe_t0.5_k2.p (0.8565) — same trend weight, only difference
is K=3 vs K=2. More patience with committed branches gives them time to develop. K=3
found a transformer at depth 3 after 4 initial children failed. K=2 broke commitment
too early and scattered budget.

#### 4. UCB C-value: lower is better for expensive expansions

| C | Parent | Global |
|---|--------|--------|
| 1.0 | **0.9426** | — |
| 1.41 | 0.9019 | **0.9402** |
| 2.0 | 0.8804 | 0.8876 |

For expensive expansions (~90s per node), lower C is preferable. High C scatters budget
across exotic but failing approaches (GNN, LSTM, spatial clustering). Low C commits
harder to promising branches. Standard √2 is too exploratory for this setting.

Global context consistently helps UCB: ucb_c1.41.g (0.9402) >> ucb_c1.41.p (0.9019).
The tree summary prevents revisiting failed strategies.

#### 5. Regret baseline is strong but limited

regret.g (0.9474) beats all UCB variants, but loses to the top 3 open-ended results.
regret.p (0.8541) is surprisingly weak — high single-run variance.
regret_depth.g (0.6220) is anomalously bad, likely a run where most nodes failed.

#### 6. Valley-crossing happens with both UCB and open-ended

**ucb_c1.0.p**: 0.86 → 0.65 → 0.62 → 0.94 (forced by failures, "accidental" valley crossing)
**oe_t0.5_k3.p**: 4 initial children failed → found NN (0.95) → transformer (0.9569)

Open-ended makes valley-crossing more likely through path commitment. Vanilla UCB only
does it when forced (no other options). This is the key advantage.

### Updated Overall Ranking (ALL Experiment 3 versions)

| Rank | Experiment | Best | Version | Strategy |
|------|-----------|------|---------|----------|
| 1 | oe_t0.3_k2.p | **0.9785** | v4 | open-ended (trend=0.3, commit=2) |
| 2 | oe_t1.0_k2.g | 0.9641 | v4 | open-ended (trend=1.0, commit=2, global) |
| 3 | oe_t0.5_k3.p | 0.9569 | v4 | open-ended (trend=0.5, commit=3) |
| 4 | regret.g | 0.9474 | v4 | signals (regret, global) |
| 5 | 3.2.p (regret+depth) | 0.9450 | v1 | signals (root bug inflated) |
| 6 | ucb_c1.0.p | 0.9426 | v4 | UCB (C=1.0) |
| 7 | ucb_c1.41.g | 0.9402 | v4 | UCB (C=1.41, global) |
| 8 | 3.1b.g (regret) | 0.9378 | v1 | signals (regret, global) |
| 9 | 3.2v2.g (regret+depth) | 0.9306 | v2 | signals (regret+depth, global) |
| 10 | 3.1bv2.g (regret) | 0.9306 | v2 | signals (regret, global) |

### First Result: ucb_c1.0.p — 0.9426 (Valley Crossing!)

The very first UCB experiment produced the most interesting tree in all of Experiment 3.

```
root [0.00]
├── root_0 [0.77] GBM + feature engineering
│   └── root_0_0 [0.79] → Neural network (MLP)
│       ├── root_0_0_0 [0.93] → SVM (RBF kernel)  ← strong!
│       │   └── root_0_0_0_0 [None] → GBM replacement (failed)
│       └── root_0_0_1 [0.78] → NN + learning rate scheduling
│           └── root_0_0_1_0 [0.86] → GBM (back to trees)
│               └── root_0_0_1_0_0 [0.65] → Deep autoencoder  ← DROPS
│                   └── root_0_0_1_0_0_0 [0.62] → Contrastive learning  ← DROPS MORE
│                       └── root_0_0_1_0_0_0_0 [0.94] → Transformer  ← BREAKTHROUGH!
│                           └── root_0_0_1_0_0_0_0_0 [None] → GBM ensemble (failed)
├── root_1 [None] (failed)
└── root_2 [None] (failed)
```

**The valley-crossing chain:**
```
root_0_0_1_0 [0.86] GBM
  → root_0_0_1_0_0 [0.65] Deep autoencoder       ← -0.21 drop
    → root_0_0_1_0_0_0 [0.62] Contrastive learning  ← -0.03 further drop
      → root_0_0_1_0_0_0_0 [0.94] Transformer       ← +0.32 BREAKTHROUGH!
```

This is exactly the open-ended exploration pattern: 0.86 → 0.65 → 0.62 → 0.94.
The search went through a deep valley (autoencoder and contrastive learning both failed)
before finding a transformer approach that achieved the best score in the entire tree.

**Why UCB produced this:** With root_1 and root_2 both failed (None), and root_0_0_0_0
also failed, the only expandable nodes were along the root_0_0_1 branch. UCB's exploration
bonus kept the deep leaf attractive since n_children=0 for each new node. In a sense,
UCB was "forced" into open-ended exploration because all the breadth-first options had
been exhausted. With a tree where more breadth-first options succeed, UCB might not
go this deep.

**Strategy diversity:** The tree traverses 6+ model families:
GBM → NN/MLP → SVM → NN+LR scheduling → GBM → Autoencoder → Contrastive → Transformer

This is far more diverse than any previous experiment, and notably includes approaches
(contrastive learning, transformers) that never appeared in the signal-based experiments.
