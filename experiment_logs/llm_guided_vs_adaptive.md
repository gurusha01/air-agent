# LLM-Guided vs Adaptive MCTS vs AIRA Vanilla

## Overview

Three tree search methods for automated ML/RL research, all sharing the same executor (Qwen3-4B) and MLGym container environment. The key difference is **how the next node to expand is chosen** and **what context the executor receives**.

| | LLM-Guided | Adaptive MCTS | AIRA Vanilla |
|---|---|---|---|
| **Selection** | LLM scientist reasons about tree | Formula-based (UCB/signals/softmax) | UCT with backpropagation |
| **Who decides** | Separate scientist model | Algorithm | Algorithm |
| **Executor context** | Global memory + scientist direction | Parent conversation or global summary | Fresh from root + operator prompt |
| **Diversity mechanism** | Verbalized sampling in scientist prompt | Coverage signal, temperature | Draft/Debug/Improve operators |
| **Code** | `air/llm_guided_tree_search.py` | `air/adaptive_tree_search.py` | `air/aira_dojo/search.py` |

---

## 1. LLM-Guided Tree Search

**Architecture:** Two-model system where a **scientist** (LLM) analyzes the full tree and decides what to try next, then an **executor** (LLM) implements it in the container.

### How it works

```
For each budget step:
  1. Scientist sees: full tree state (all nodes, scores, strategies, errors)
  2. Scientist decides: which node to expand, what direction, explore vs exploit
  3. Executor receives: parent context + scientist's direction + guidance
  4. Executor runs multi-turn ReAct in MLGym container
  5. Score is recorded, memory is updated
```

### Selection mechanism

The scientist uses a **two-turn prompt**:
- **Turn 1**: Choose which nodes to inspect (read their full action logs)
- **Turn 2**: Diagnose what worked/failed, brainstorm 3 diverse strategies via verbalized sampling, choose one

The scientist sees the full tree with scores and can reason about *why* things failed, not just *that* they failed. It decides whether to **deepen** a promising branch or **explore** a new direction from root.

### Key features

- **Memory system**: Accumulates insights across iterations (capped at 5 entries). Scientist writes one-sentence learnings after each step.
- **Code inspection**: Scientist can request to read the executor's actual action logs from specific nodes, seeing what commands were run and what errors occurred.
- **Workspace file injection**: All source files are pre-loaded into the root node's context so no budget is wasted on "read the code" nodes.
- **Verbalized sampling**: Scientist must brainstorm 3 strategies with probability < 0.2 each, forcing diversity beyond obvious first ideas.

### Strengths

- Can reason about failure modes (bug vs bad approach vs near-ceiling)
- Can coordinate across the tree (avoid repeating failed ideas)
- Natural language memory transfers insights between branches
- Works well on structured tasks (tabular ML, game theory)

### Weaknesses

- Scientist quality bottlenecks everything (weak scientist = poor decisions)
- When scientist gives overly specific code-level instructions, executor may fail to implement them
- Extra LLM call per step adds latency
- Scientist and executor can disagree on approach

---

## 2. Adaptive MCTS (our custom method)

**Architecture:** Single-model system. A formula selects the next node, then the executor runs independently in the container.

### How it works

```
For each budget step:
  1. Formula selects which node to expand (UCB, signals, softmax, open-ended)
  2. Generate strategy via verbalized sampling (explore or exploit mode)
  3. Executor receives: parent context (or global summary) + strategy
  4. Executor runs multi-turn ReAct in MLGym container
  5. Score is recorded
```

### Selection mechanisms (multiple options)

- **UCB**: Classic `Q + C * sqrt(ln(N) / n)`. Q = best_child / global_best.
- **Signals**: Weighted combination of variance, regret, coverage, depth, LLM-guidance signals. Each normalized to [0,1].
- **Open-ended**: UCB + trend bonus (reward branches showing improvement) + path commitment (stick with improving branches for K steps).
- **Softmax**: Temperature-scaled softmax over node scores.

### Key features

- **Explore/exploit modes**: Explore uses Tail VS (low-probability strategies), exploit uses Local VS (refine current approach).
- **Global context mode**: Executor sees a summary of the entire tree (what worked, what failed, scores).
- **Reflexion**: Optional self-critique step where executor reflects on failures before retrying.
- **No scientist overhead**: Selection is O(1) formula evaluation.

### Strengths

- Simple, no extra model calls for selection
- Executor acts independently and can improvise (often ignores the proposed strategy and does something better)
- Proven UCB guarantees on exploration
- Fast iteration

### Weaknesses

- Can't reason about *why* a node failed
- Formula can't distinguish "bad approach" from "fixable bug"
- Executor ignoring the strategy means the tree structure is less meaningful
- No cross-branch learning (each node is semi-independent)

---

## 3. AIRA Vanilla (from arXiv:2507.02554)

**Architecture:** Single-model system with structured operators (Draft, Debug, Improve) and UCT-based MCTS with backpropagation.

### How it works

```
Phase 1: Execute root baseline
Phase 2: Draft N initial solutions
Phase 3: MCTS loop
  For each remaining budget step:
    1. UCT traversal from root to select leaf
    2. Choose operator based on node state:
       - Buggy node → DEBUG (fix errors)
       - Leaf with score → DRAFT (new approach)
       - Internal node → IMPROVE (refine)
    3. Execute in container with scoped memory
    4. Backpropagate score up ancestor chain
```

### Selection mechanism

UCT (Upper Confidence bounds for Trees):
- `UCT = Q_normalized + c * sqrt(ln(N_parent) / N_child)`
- Q normalized via global min/max of all scores
- Traverse tree from root, always picking highest-UCT child until reaching a leaf
- After execution, backpropagate: update visit_count and cumulative_value for all ancestors

### Key features

- **Typed operators**: Draft (create new), Debug (fix errors), Improve (refine working solution). Each has a distinct prompt.
- **Scoped memory**: Each node starts fresh from root conversation (avoids context accumulation). Adds operator-specific context about siblings or ancestors.
- **Backpropagation**: Scores propagate up the tree, affecting future UCT selection.
- **No reflexion** (in vanilla config): Each node gets a clean slate.

### Strengths

- Clean per-node context (no accumulation / context blowup)
- Backpropagation gives principled exploration/exploitation
- Operator types match natural research workflow
- Well-studied MCTS guarantees

### Weaknesses

- No cross-branch reasoning (UCT is purely score-based)
- Draft operator generates generic approaches (no scientist guidance)
- Debug operator may waste budget on unfixable approaches
- Less diversity than verbalized sampling

---

## Key Insight: Executor Independence

A surprising finding: in both Adaptive MCTS and LLM-Guided, **the executor often ignores the proposed strategy**. The difference is in the consequences:

| | What happens when executor ignores strategy |
|---|---|
| **Adaptive MCTS** | Executor does simple config changes (e.g., `num_hidden_units: 512`) that work. The tree structure becomes somewhat meaningless but scores are good. |
| **LLM-Guided** | Executor *tries* to follow the scientist's specific code-level instructions but fails (complex sed patterns, syntax errors). When it falls back to its own ideas, results improve. |

**Implication**: The scientist should give **high-level directions** (what to try) not **code-level instructions** (how to implement). The executor is better at figuring out implementation details on its own.

---

## Mountaincar Results (n5, Qwen3-4B, 5 runs)

| Method | Mean Best | Min | Max | Scores |
|--------|----------|-----|-----|--------|
| **Adaptive MCTS** | **65.39** | 51.39 | 68.90 | 68.90, 68.90, 68.90, 51.39, 68.90 |
| AIRA Vanilla | 56.82 | 44.54 | 68.90 | 51.39, 68.90, 68.90, 44.54, 51.39 |
| LLM-Guided (old prompt) | 50.65 | 49.72 | 54.36 | 49.72, 54.36, 49.72, 49.72, 49.72 |
| LLM-Guided v5 (new prompt) | 68.90* | — | 68.90 | *single test run |

*Baseline: 33.79*

The v5 scientist prompt (with verbalized sampling and natural deepen/explore decisions) achieved 68.90 on its first test run, matching the best Adaptive MCTS result.
