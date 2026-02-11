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

Options to consider:

**A. UCB-style scoring (UCT)**
```
UCB(node) = mean_child_score + C * sqrt(ln(total_visits) / visits(node))
```
- Balances exploitation (high mean score) with exploration (under-visited nodes)
- Classic MCTS approach, well-understood tradeoffs
- C is a tunable exploration constant

**B. Best-first (greedy)**
```
select(node) = argmax(node.score)
```
- Always expand the highest-scoring node
- Pure exploitation — might miss better paradigms elsewhere
- Simple, and the right choice if we believe refinement > exploration

**C. Score-gap based**
```
priority(node) = node.score - best_sibling.score  (or vs baseline)
```
- Expand nodes that are closest to the best — "almost there, might improve"
- Skip clearly failed branches

**D. LLM-guided selection**
- Show the LLM the current tree with all scores and ask it which node to expand
- The LLM has domain knowledge about which approaches have room for improvement
- Most expensive option but potentially most informed

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

## Sub-Experiments to Run

### Exp 3.0: Baseline — Reproduce Exp 2.1 (Tail VS) with deeper trees
- Same as Exp 2.1 but with depth=3 (bf=3, 40 nodes)
- Does more compute with fixed tail VS improve results?
- Establishes the "more nodes" baseline before adding adaptive logic

### Exp 3.1: Score-threshold adaptive mode
- Depth 1: always Tail VS (explore)
- Depth 2+: if parent.score >= median(all_scores) → Local VS (exploit), else → Tail VS (explore)
- Simplest adaptive policy — does it beat fixed Tail VS?

### Exp 3.2: Past-attempt-aware exploration
- Same as Exp 2.1 (Tail VS everywhere) but when generating strategies, include summary of all sibling strategies and their scores
- Tests whether "don't repeat what failed" improves diversity and scores

### Exp 3.3: UCB node selection + adaptive mode
- Instead of BFS, use UCB to select which node to expand next
- Mode: Explore if node has <2 children, Exploit if node has >=2 children and score is in top-k
- Fixed budget of N node expansions, allocated by UCB
- Variable-depth tree — promising branches go deeper

### Exp 3.4: LLM-guided selection and mode
- After each expansion round, show the LLM the full tree with scores
- Ask: (1) which node to expand, (2) explore or exploit
- Most expensive but most flexible
- The LLM can reason: "RF at 0.87 has plateaued, but GBM at 0.90 might improve with feature engineering"

---

## Open Questions

1. **What's the right node budget?** Exp 2 used 12 model-executed nodes. Should Exp 3 use the same budget (to compare fairly) or more (since adaptive allocation should use budget more efficiently)?

2. **How to set the explore/exploit threshold?** Relative (top-k%) or absolute (task-specific)? Relative is more general but requires seeing enough scores first.

3. **Should the tree be balanced or lopsided?** BFS guarantees breadth. Adaptive selection might produce degenerate trees (one branch goes to depth 10, others abandoned at depth 1). Is that bad?

4. **Does past-attempt context actually help?** The LLM might already "know" what it tried (from conversation history). Explicit context might be redundant — or it might be critical since siblings have separate conversations.

5. **UCB vs best-first vs LLM-guided?** UCB is principled but treats nodes as bandits (ignores ML domain knowledge). Best-first is greedy. LLM-guided is most informed but adds latency and cost. Might need to try all three.
