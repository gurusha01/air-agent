# Experiment 4: LLM-Guided Tree Search

## Motivation

Analysis of Feb22 baseline experiments reveals two fundamental problems with formula-based tree search (UCB, Open-Ended, Softmax):

### Problem 1: The executor model can't implement ambitious strategies

The 4B executor frequently crashes when attempting neural nets, hierarchical RL, or complex ensembles. Nodes that try these approaches spend 6-7 actions rewriting broken code without recognizing the approach is doomed. This wastes budget on strategies the executor can't deliver.

**Evidence from Feb22 runs:**
- Nodes attempting PyTorch models: ~70% crash rate, avg 12+ actions before giving up
- Nodes attempting sklearn pipelines: ~10% crash rate, avg 5 actions to validate
- The search tree doesn't learn from these failures — UCB/Softmax see the score (or lack thereof) but can't distinguish "bad approach" from "fixable implementation bug"

### Problem 2: Formula-based selection can't reason about *why* things failed

UCB sees that a node scored 0.85 and another scored 0.92. It can compute which to expand. But it can't reason about:
- Whether the 0.85 node used a fundamentally limited approach (linear model on nonlinear data)
- Whether the 0.85 node had a fixable bug that a second attempt could resolve
- Whether the 0.92 node is near its ceiling or has room for hyperparameter tuning
- Whether an untried approach (e.g., ensemble of the top two) could break through

A human scientist would do error analysis, assess the landscape, and make informed decisions about where to invest budget. Formula-based selection can't do this.

## Design: Two-Model Architecture

### Solution: Replace formula-based selection with an LLM scientist

| Role | Model | Purpose |
|------|-------|---------|
| **Scientist** (selector) | Larger model via API (GPT-4o) | Analyzes tree, decides what to expand, writes memory |
| **Executor** (worker) | Qwen3-4B via vLLM | Implements strategy in container (unchanged) |

The scientist sees the full tree state — all nodes, their strategies, scores, errors, and action counts — and reasons like a human researcher about what to try next. The executor remains unchanged: it writes code, runs experiments, and validates in the MLGym container.

---

## Iteration History

### v1: Initial Implementation

**Key design decisions:**
1. Two separate models (scientist for reasoning, executor for coding)
2. Memory system (accumulates insights across iterations)
3. Structured output parsing (REASONING/ACTION/DIRECTION/MODE/MEMORY via regex)
4. VS (Verbalized Sampling) integration — after scientist decides direction, VS generates the actual strategy

**Architecture: 3 phases**
```
Phase 1: Execute root baseline
Phase 2: Generate initial_breadth children via Tail VS (diverse sampling)
Phase 3: Scientist-guided loop
    for each remaining budget step:
        1. Build tree view
        2. Scientist decides: which node, direction, mode
        3. Generate strategy via VS with scientist's direction
        4. Execute in container
        5. Update tree and memory
```

**Results (BOS task):**
- n5 mean: 1.21, n15 mean: 1.20 (WORSE than n5 — scaling inverted!)
- Worst run (n15_r1): 0.85 — spent all 15 nodes on LightGBM refinement
- Best run (n15_r4): 0.96 — tried 7 different approaches

**Failure modes identified:**
1. **Over-exploitation**: Scientist spent 10+ nodes refining one approach (LightGBM tuning)
2. **Context blowup**: Accumulated memory made prompts too long → "max_tokens too large" errors killed 9 nodes in worst run
3. **Direction confusion**: Strategy text combined scientist direction with VS suggestion. Executor followed VS suggestion (game theory for BOS) instead of scientist's direction (CatBoost for BOS)

### v2: Remove VS, Scientist Brainstorms Directly

**User feedback:** "I wanted the scientist itself to do VS and then choose, not as a separate component."

**Changes:**
- Removed Phase 2 (initial breadth via VS entirely)
- Removed `_generate_strategy()` method
- Scientist now handles ALL budget steps from step 1, including the first node
- Added CANDIDATES section to scientist prompt — brainstorm 3-5 strategies before choosing
- Scientist's direction is passed directly to executor (no VS intermediary)
- Memory capped at 5 entries to prevent context blowup

**Also changed scientist prompt:**
- Rewritten as "mentor" role instead of "researcher" — focused on coaching the executor
- Added EXECUTOR_GUIDANCE field — warnings/tips passed directly to executor's context
- Added exploitation limit: "Do NOT expand the same node more than 3 times"
- Added detailed executor capability description (what it can/can't do)
- Added examples for different task types (ML, game theory, RL)

**Architecture simplified to 2 phases:**
```
Phase 1: Execute root baseline
Phase 2: Scientist-guided loop (all budget steps)
    for each step:
        1. Build tree view
        2. Scientist brainstorms candidates, picks best
        3. Direction passed directly to executor
        4. Execute in container
        5. Update tree and memory
```

**Bug fix:** `parent.score` could be None when expanding a failed node → TypeError on `f"{parent.score:.4f}"`. Fixed with None-safe formatting.

**Initial v2 results (BOS, before task-awareness fix):**
- n15 mean: ~1.20 (same as v1)
- Root cause: scientist prompt only received task name ("battleOfSexes (game_theory)") but no actual task details. For BOS, scientist recommended ML models (CatBoost) while executor wrote game strategy functions.

### v2.1: Task Awareness

**Problem diagnosed:** The scientist had no idea what the executor's task actually was. It saw "Battle of Sexes" and guessed it should recommend ML models. But the executor saw the full system prompt describing a game theory task where it writes strategy functions.

**Fix:** Injected the executor's full context into the scientist prompt:
- `task.system_prompt` — the executor's system prompt
- `task.root_task_desc` — the task description with data preview

Added as `{task_details}` section: "Task Details (this is what the executor sees)"

Also updated:
- Executor capability section to be task-type-aware (not ML-only)
- Examples to include game theory and RL
- Diversity check to not be ML-specific

**Results (BOS only, re-run after fix):**

| Method | n5 mean | n15 mean |
|--------|---------|----------|
| Softmax | 1.27 | 1.40 |
| AIRA MCTS | 1.30 | 1.38 |
| Open-Ended | 1.29 | 1.39 |
| LLM-Guided v1 | 1.21 | 1.20 |
| **LLM-Guided v2.1** | **1.26** | **1.42** |

Massive improvement! v2.1 is now the best method on BOS at n15. Scaling works: n5→n15 improves (was inverted in v1).

**Per-run BOS n15 scores:**
- r1: 1.4394, r2: 1.4406, r3: 1.4438, r4: 1.3163, r5: 1.4424
- 4 out of 5 runs score ~1.44 (near optimal), 1 run at 1.32

### v2.2: Workspace File Injection (Current)

**Problem diagnosed on mountaincar:** The scientist wasted entire nodes telling the executor to "read src/networks.py" or "cat src/config.yaml" before making changes. On a 5-node budget, spending 2-3 nodes just reading files is fatal.

**Analysis of mountaincar runs:**
- Failed runs (r4, r5): Score = 33.79 (baseline, zero improvement). Scientist's first direction was always "Read src/networks.py to understand the architecture." Executor spent 15-21 actions reading files, hit max-action limit, FAIL. Wasted 2-3 of 5 nodes on pure exploration.
- Successful run (r2): Score = 75.52. The node that scored 75.5 did exactly 4 actions: (1) modify config.yaml via Python one-liner (lr=3e-4, num_train_envs=64), (2) rm checkpoints, (3) train, (4) validate.

**Root cause:** When expanding from root (depth 0→1), the executor gets zero file context — just system prompt and task description. `child_msgs = copy.deepcopy(parent.conversation_history)` inherits parent's conversation, but root has no file reads. So the scientist correctly identified the executor needs context, but wasted nodes on it.

**Three changes made:**

1. **Inject workspace files into root context:** New `_read_workspace_files()` method reads all `.py`, `.yaml`, `.json`, `.cfg`, `.txt`, `.sh` files from the workspace (excluding checkpoints/__pycache__, truncating files >5000 chars) and appends them to the root node's task description. Every child node inherits this context.

2. **Tell the scientist files are pre-loaded:** Added to scientist prompt: "The executor already has ALL source files from the workspace pre-loaded in its context. Do NOT waste a node asking it to 'read' or 'examine' files — it already knows the code. Every direction should be an ACTIONABLE change."

3. **Remove file-reading instructions:** Changed diversity check guidance from "Read the task details carefully" to "The executor already has all source files in context — don't ask it to read code."

**Expected impact:** No more wasted nodes on exploration. Every node from step 1 should be an actionable change (modify hyperparameters, write code, etc.).

---

## Files

| File | Description |
|------|-------------|
| `air/llm_guided_tree_search.py` | Main implementation (~600 lines) |
| `air/run_parallel.py` | Modified: `is_llm_guided` field, llm-guided suite, command builder |
| `air/plot_feb22.py` | Modified: includes LLM-Guided v1 and v2 in comparison plot |
| `experiment_logs/experiments4.md` | This file |

## Output Directories

| Directory | Description |
|-----------|-------------|
| `outputs/LLM_Guided/` | v1 results (preserved, all tasks) |
| `outputs/LLM_Guided_v2/` | v2+ results (BOS re-run after task-awareness fix, mountaincar re-run after file injection) |
| `outputs/Feb22_Baselines/` | Baseline results (softmax, aira_mcts, oe) — 115/120 complete |

## Comparison Plan

Run the same setup as Feb22 baselines:
- 4 tasks × [5, 15] budget × 5 runs
- Compare against: softmax + reflexion, AIRA MCTS, open-ended
- Metrics: best score, score variance across runs, tree diversity

Currently completed: BOS (v2.1), mountaincar (v2.2 running).
Titanic and houseprice have partial v2 results from before bug fixes — may need re-run.
