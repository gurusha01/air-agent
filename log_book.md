# Experiment Log Book

This file tracks all training runs with their W&B run names and key details.

---

## Exp 1.3 - Titanic with Heredoc File Writing (Bash Syntax Fix)

**Date:** 2026-02-06
**W&B Run Name:** `exp1.3-titanic-0206`
**Status:** Running

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
