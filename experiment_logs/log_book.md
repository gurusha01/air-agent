# Experiment Log Book

This file tracks all training runs with their W&B run names and key details.

---

## Exp 2.3 - Tree Search with Local Refinement Sampling

**Date:** 2026-02-10
**Status:** Completed - Best: 84.7% accuracy

### Config
| Parameter | Value |
|-----------|-------|
| branching_factor | 3 |
| max_depth | 2 |
| max_actions | 15 |
| temperature | 0.9 |
| sampling_mode | **local** (stay within same model family) |
| Model | Qwen/Qwen3-4B-Instruct-2507 (via vLLM) |

### Results

- Best: 0.8469 (root_1_2), +8.1% over baseline
- Runtime: 7.9 min, 10 nodes, 1 failure
- Score range: 0.81–0.85 (extremely narrow, std=0.009)
- All nodes are RF/normalization variations — no algorithm diversity
- **Worst of all search modes** — local refinement can't escape RF

### Output
Directory: `outputs/tree_search/exp2.3/`

---

## Exp 2.2 - Tree Search with Uniform Verbalized Sampling

**Date:** 2026-02-10
**Status:** Completed - Best: 88.8% accuracy

### Config
| Parameter | Value |
|-----------|-------|
| branching_factor | 3 |
| max_depth | 2 |
| max_actions | 15 |
| temperature | 0.9 |
| sampling_mode | **uniform** (equal probability, no tail bias) |
| Model | Qwen/Qwen3-4B-Instruct-2507 (via vLLM) |

### Results

- Best: 0.8876 (root_0_2, stacking ensemble), +12.2% over baseline
- Runtime: 15.8 min, 13 nodes, 2 failures
- Score range: 0.61–0.89 (wide due to broken target encoding attempts)
- Worse than no-VS (90.0%) — "reasonable" strategies harder to implement than they sound

### Output
Directory: `outputs/tree_search/exp2.2/`

---

## Exp 2.0 - Tree Search WITHOUT Verbalized Sampling

**Date:** 2026-02-10
**Status:** Completed - Best: 90.0% accuracy

### Config
| Parameter | Value |
|-----------|-------|
| branching_factor | 3 |
| max_depth | 2 |
| max_actions | 15 |
| temperature | 0.9 |
| verbalized_sampling | **OFF** |
| Model | Qwen/Qwen3-4B-Instruct-2507 (via vLLM) |

### Results

- Best: 0.8995 (root_0_0), +13.4% over baseline
- Runtime: 6.6 min, 10 nodes, 1 failure
- Score range: 0.82–0.90 (narrow, all RF variants)
- Depth-2 improved +2.4% over depth-1

### Output
Directory: `outputs/tree_search/exp2.0/`

---

## Exp 2.1 - Tree Search WITH Verbalized Sampling

**Date:** 2026-02-10
**Status:** Completed - Best: 94.0% accuracy

### Config
Same as Exp 2.0 but with **verbalized sampling ON**.

### Results

- Best: 0.9402 (root_2, SVM+RBF), +17.5% over baseline
- Runtime: 18 min, 10 nodes, 2 failures
- Score range: 0.72–0.94 (wide, diverse strategies)
- **Beats GPT-5.2 (92.8%)**

### Key Comparison (2.0 vs 2.1)
- VS finds non-obvious strategies (SVM+RBF) that temperature alone misses
- VS: higher peak (94% vs 90%) but wider variance and more failures
- Without VS: siblings get nearly identical scores (0.8995, 0.8995)
- With VS: siblings differ dramatically (FAIL, 0.8349, 0.7201)

### Output
Directory: `outputs/tree_search/`
Viewer: `uv run streamlit run air/tree_viewer.py --server.port 8502`

---

## Exp 1.6 - Continuous GRPO-Style Reward

**Date:** 2026-02-09
**W&B Run Name:** `exp1.6-titanic-0208`
**Status:** Completed (200 steps) - Policy collapsed after step 100

### Reward Function
`reward = (improvement * 5 - 0.5) + format_penalty`, clipped to [-1, 1]

### Results
- Peaked steps 80-100: mean 0.890, max 0.986
- Policy collapsed: mean dropped to 0.852, seq_len 5500→1300
- Every batch had gradient (vs 26% zero-gradient in exp 1.5)

### Trajectories
Directory: `outputs/trajectories_exp1.6/`

---

## Exp 1.5 - Episode-Done Fix + Threshold Reward

**Date:** 2026-02-07
**W&B Run Name:** `exp1.5-titanic-0207`
**Status:** Completed (200 steps) - No learning (zero-variance reward)

### Key Finding
92% of trajectories got same reward (-0.2) → GRPO advantages ≈ 0 → no gradient.

### Trajectories
Directory: `outputs/trajectories_exp1.5/`

---

## Exp 1.4 - Sparse High-Bar Reward (episode_done bug)

**Date:** 2026-02-07
**W&B Run Name:** `exp1.4-titanic-0207`
**Status:** Superseded (episode_done bug meant rewards never fired)

### Trajectories
Directory: `outputs/trajectories_exp1.4/`

---

## GPT-5.2 Benchmark

**Date:** 2026-02-09
**Method:** MLGym `run.py` with `litellm:openai/gpt-5.2`
**Result:** 92.8% accuracy, 25 steps, $0.55 cost
**Trajectory:** `trajectories/ubuntu/litellm-openai/gpt-5.2__titanicSurvival__*/`

---

## Exp 1.3 - Titanic with Heredoc File Writing (Bash Syntax Fix)

**Date:** 2026-02-06
**W&B Run Name:** `exp1.3-titanic-0206`
**Status:** Superseded by Exp 1.4

### Config
| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Task | titanic |
| max_steps | 200 |
| batch_size | 16 |
| rollouts_per_example | 8 |
| max_turns | 20 |
| Learning Rate | 1e-5 |
| LoRA Rank/Alpha | 16/32 |
| seq_len | 32768 |

### Key Fixes
1. **Heredoc file writing** - Use `cat << 'ENDOFFILE' > file.py` instead of `edit` command
   - MLGym's `_check_syntax()` runs `bash -n` on Python code, causing syntax errors
   - Heredoc bypasses bash syntax checking
2. **Multi-command extraction** - Extract first command only instead of rejecting entirely

### Early Results
- ~29% trajectories achieving positive improvement
- Mean improvement: 7.3% (when successful)
- Max improvement: 9.09%
- Reward improved from -0.125 (step 0) to +0.375 (step 1)

### Trajectories
Directory: `outputs/trajectories_exp1.3/`

---

## Exp 1.2 - Titanic with Early Termination Fix + Context Length

**Date:** 2026-02-05 to 2026-02-06
**W&B Run Name:** `exp1.2-titanic-0205` (trainer: `0hkse3pb`)
**W&B URL:** https://wandb.ai/Gurusha-personal/mlgym-rl/runs/0hkse3pb
**Status:** Superseded by Exp 1.3 (edit command bash syntax issue)

### Config
| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Task | titanic |
| max_steps | 200 |
| batch_size | 16 |
| rollouts_per_example | 8 |
| max_turns | 20 |
| Learning Rate | 1e-5 |
| LoRA Rank/Alpha | 16/32 |
| **seq_len** | **32768** |
| **max_model_len** | **32768** |

### Key Fixes
1. **MLGym max_steps increased** - From 40 to 400 (fixes parallel rollout step exhaustion)
2. **Context length increased** - From 8K to 32K tokens (fixes rollouts stopping at turn 3)
3. **Trajectory saving fixed** - Now saves when max_turns reached via check_done()
4. **validate/submit commands** - Added to command list + reload after reset
5. **Policy step tracking** - Recorded at rollout START, not save time (π_N in filenames)

### Reward Function
```
+0.5  correct tool call (no error)
-0.5  incorrect tool call (error/traceback)
+10 × improvement at episode end
```

### Trajectories
Directory: `outputs/trajectories_exp1.2/`
Filename format: `{task}_pi{policy_step}_{timestamp}.json`

---

## Exp 1.1 - Titanic with Tool Call Rewards (Early Termination Bug)

**Date:** 2026-02-05
**W&B Run Name:** `exp1.1-titanic-2130` (trainer: `vsg7uokk`)
**W&B URL:** https://wandb.ai/Gurusha-personal/mlgym-rl/runs/vsg7uokk
**Status:** Stopped (trajectories ending early at 1-3 turns)

### Issue
Trajectories ended at 1-3 turns instead of max_turns=20 because MLGym's internal `max_steps` was set too low (40). With 8 parallel rollouts, each calling `env.step()`, the budget was exhausted after ~5 turns.

### Notes
- Reached step 10/200 before stopping
- Fixed in Exp 1.2 by increasing max_steps

---

## Previous Runs (Archived)

### exp1.2-single-task-tool-reward (Multiple runs)
- Early iterations while debugging MLGym command loading
- Commands weren't working ("command not found" errors)

### exp1.1-full-multi-task
- Attempted multi-task training before fixing single task
- Crashed due to TypeError (baseline_score=None)

---

## Naming Convention

Format: `exp{major}.{minor}-{task}-{MMDD}`

- **major**: Experiment series (1 = initial training)
- **minor**: Iteration within series
- **task**: Primary task name
- **MMDD**: Date (month + day)

Example: `exp1.2-titanic-0206` = Experiment 1.2, titanic task, Feb 6th
