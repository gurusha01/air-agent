# Tree Search Improvements — Next Experiment Batch

## Problem Diagnosed (2026-02-22)

Analysis of sweep-qwen3 results shows **tree degeneration into chains**:
- Most experiments have `branches=1` — the tree is a single chain, not a tree
- When a child becomes the highest-scoring node, it gets selected again, creating depth without breadth
- Performance drops at n=30/60 because the chain drifts away from good solutions with no backtracking
- High node failure rates (30-50%) waste budget

## Three Improvements to Implement

### Improvement 1: Sibling-Aware Strategy Generation

**Problem:** `_generate_strategy()` only sees `parent.strategy`. It doesn't know what siblings already tried, so it can propose the same strategy repeatedly.

**Note:** `build_sibling_summary()` already exists and is injected into the *executor's* prompt (the agent writing code), but the *strategy generator* (the LLM proposing what to try) is blind to siblings.

**Fix:** Pass sibling strategies into `_generate_strategy()`:
```python
def _generate_strategy(self, parent, mode):
    sibling_strats = []
    for cid in parent.children:
        if cid in self.nodes:
            sib = self.nodes[cid]
            score_str = f"{sib.score:.4f}" if sib.score is not None else "FAILED"
            sibling_strats.append(f"- {sib.strategy[:100]} → {score_str}")

    strategies = self.llm.generate_strategies(
        ..., sibling_info=sibling_strats  # NEW parameter
    )
```

Modify `STRATEGY_PROMPT_TAIL` to include:
```
Already tried from this parent (DO NOT repeat these):
{sibling_info}
```

**Cost:** Zero extra LLM calls. Just adds context to existing call.

### Improvement 2: LLM-Guided Node Selection

**Problem:** UCB and open-ended selection only use numeric scores. They can't reason about "this node uses Random Forest but hasn't tried neural nets yet."

**Note:** `_llm_guidance_batch()` already exists in the signals-based policy but is NOT wired to UCB/open-ended.

**Fix:** Add LLM guidance as a bonus term in the open-ended UCB formula:
```
score = Q(node) + C*sqrt(ln(N)/(n+1)) + W_trend*trend + W_llm*llm_prior
```

**Cost management:** Call LLM guidance every K=5 steps (not every step), cache scores between calls with decay.

### Improvement 3: Forced Minimum Breadth per Expansion

**Problem:** When a node is selected, exactly 1 child is created. That child often becomes highest-scoring, gets selected again → chain.

**Fix:** When a node is selected, create `min_children` (e.g., 3) children before returning to global selection.

**Adaptive variant:**
1. Create first child. If it scores much worse than parent, stop.
2. If reasonable, create remaining children (which will be sibling-aware via Improvement 1).
3. Cap at `min_children` or budget remaining.

This effectively inverts the commitment threshold: instead of "commit to going deeper," "commit to exploring siblings before going deeper."

## Implementation Order

1 → 3 → 2 (cheapest fix first, most expensive last)

## Key Files to Modify

- `air/tree_search.py` — `STRATEGY_PROMPT_TAIL`, `generate_strategies()` signature
- `air/adaptive_tree_search.py` — `_generate_strategy()`, `_expand_one()`, selection policies
- `air/run_parallel.py` — new sweep suite with improved methods

## New Experiment Matrix

Compare old methods vs improved methods on same tasks/budgets to measure the impact of each improvement independently.
