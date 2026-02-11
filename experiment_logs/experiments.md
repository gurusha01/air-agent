# Experiments

## Experiment 1.0: Infrastructure Validation (Debug Run)

**Date:** 2026-02-05
**Goal:** Verify MLGym + prime-rl integration works end-to-end
**Status:** Completed Successfully

### Configuration

| Parameter | Value |
|-----------|-------|
| Config File | `configs/mlgym/rl_debug.toml` |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA Rank | 16 |
| max_steps | 10 |
| batch_size | 8 |
| max_turns | 10 |
| Tasks | titanic only |

### Results

| Metric | Value |
|--------|-------|
| Training Steps Completed | 10/10 |
| Container Errors | 0 |
| Loss | 0.0 (reward issue) |

### Notes

- Infrastructure verified working
- Container stability fix verified (cd before reset)
- Reward signal was 0 because model didn't call `validate`

---

## Experiment 1.1: Single Task with Tool Call Rewards

**Date:** 2026-02-05
**Goal:** Train with dense reward signal from tool call success/failure
**Status:** Stopped (issues found, moved to Exp 1.2)

### Configuration

| Parameter | Value |
|-----------|-------|
| Config File | `configs/mlgym/rl_full.toml` |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 1e-5 |
| max_steps | 200 |
| batch_size | 16 |
| rollouts_per_example | 8 |
| max_turns | 20 |
| num_examples_per_task | 10 |
| seq_len | 8192 |
| Tasks | titanic only |

### Reward Function

```
+0.5  for correct tool call (no error in observation)
-0.5  for incorrect tool call (error/traceback in observation)
+10 × improvement  at final validate (episode end)
```

### Key Fixes Applied

1. **MLGym commands loaded** - `open`, `edit`, `validate` now work in container
2. **Environment variables set** - `WINDOW=100` fixes jq errors in `open` command
3. **Updated prompt** - Tells model files are in `data/` directory

### Training Math

```
1 task × 10 examples = 10 examples
batch_size=16, rollouts_per_example=8 → 2 examples per step
1 epoch = 10/2 = 5 steps
200 steps = 40 epochs
```

### Monitoring

```bash
# Trainer progress
tail -f /home/ubuntu/MLScientist/MLGym/outputs/logs/trainer.stdout

# Orchestrator
tail -f /home/ubuntu/MLScientist/MLGym/outputs/logs/orchestrator.stdout

# View trajectories
cd /home/ubuntu/MLScientist/air-agent
uv run python air/view_trajectory.py --latest --dir /home/ubuntu/MLScientist/MLGym/outputs/trajectories_exp1.1

# Streamlit viewer
uv run streamlit run air/compare_trajectories.py --server.port 8502
```

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.1-titanic-tool-reward
- **URL:** https://wandb.ai/Gurusha-personal/mlgym-rl

### Artifacts

- Trajectories: `outputs/trajectories_exp1.1/`
- Checkpoints: `outputs/checkpoints/`
- Training log: `outputs/exp1.1_training.log`

---

## Experiment 1.2: Fixed Trajectory Saving + Longer Context

**Date:** 2026-02-05 to 2026-02-06
**Goal:** Fix trajectory saving and context length issues from Exp 1.1
**Status:** Superseded by Exp 1.3 (edit command bash syntax issue)

### Configuration

| Parameter | Value |
|-----------|-------|
| Config File | `configs/mlgym/rl_full.toml` |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 1e-5 |
| max_steps | 200 |
| batch_size | 16 |
| rollouts_per_example | 8 |
| max_turns | 20 |
| num_examples_per_task | 10 |
| **seq_len** | **32768** (increased from 8192) |
| **max_model_len** | **32768** (increased from 8192) |
| Tasks | titanic only |

### Key Fixes Applied (vs Exp 1.1)

1. **Context length increased** - 8K → 32K tokens (Qwen3 supports 262K)
2. **Trajectory saving fixed** - Now saves when max_turns reached via check_done()
3. **validate/submit commands** - Added to command list + reload after reset
4. **Policy step tracking** - Recorded at rollout START, not save time

### Reward Function

Same as Exp 1.1:
```
+0.5  for correct tool call (no error in observation)
-0.5  for incorrect tool call (error/traceback in observation)
+10 × improvement  at final validate (episode end)
```

### Monitoring

```bash
# Trainer progress
tail -f /home/ubuntu/MLScientist/MLGym/outputs/logs/trainer.stdout

# Orchestrator
tail -f /home/ubuntu/MLScientist/MLGym/outputs/logs/orchestrator.stdout

# View trajectories (terminal)
cd /home/ubuntu/MLScientist/air-agent
uv run python air/view_trajectory.py --latest --dir /home/ubuntu/MLScientist/MLGym/outputs/trajectories_exp1.2

# Streamlit viewer (compare trajectories)
uv run streamlit run air/compare_trajectories.py --server.port 8502
```

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.2-titanic-0205
- **Run URL:** https://wandb.ai/Gurusha-personal/mlgym-rl/runs/0hkse3pb

### Artifacts

- Trajectories: `outputs/trajectories_exp1.2/`
- Checkpoints: `outputs/checkpoints/`

### Notes

- Peak VRAM: ~17.2 GiB (fits on 24GB A10G with 32K context)
- Trajectory filenames now include policy version (π_N) to track which policy generated them

---

## Experiment 1.3: Heredoc File Writing (Bash Syntax Fix)

**Date:** 2026-02-06
**Goal:** Fix MLGym edit command bash syntax errors by using heredoc approach
**Status:** Running

### Configuration

Same as Exp 1.2, with updated system prompt for heredoc file writing.

| Parameter | Value |
|-----------|-------|
| Config File | `configs/mlgym/rl_full.toml` |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| max_steps | 200 |
| batch_size | 16 |
| rollouts_per_example | 8 |
| max_turns | 20 |
| seq_len | 32768 |
| Tasks | titanic only |

### Key Fixes Applied (vs Exp 1.2)

1. **Heredoc file writing** - Changed from `create`/`edit` to `cat << 'ENDOFFILE' > file.py`
   - MLGym's `_check_syntax()` runs `bash -n` on ALL input including Python code
   - This caused bash syntax errors when editing Python files
   - Heredoc approach bypasses this issue

2. **Multi-command extraction** - Extract first command only instead of rejecting
   - Model sometimes outputs multiple commands in one response
   - Now extracts and executes first command, ignores trailing ones

### System Prompt Changes

```
To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > train_and_predict.py
import pandas as pd
# your code here
ENDOFFILE
```

### Early Results (Before Full Training)

- ~29% of trajectories achieving positive improvement over baseline
- Mean improvement: 7.3% (when successful)
- Max improvement: 9.09%
- Step 0 completed, reward improved from -0.125 to +0.375 by step 1

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.3-titanic-0206
- **Trajectories:** `outputs/trajectories_exp1.3/`

---

## Experiment 1.4: Sparse High-Bar Reward

**Date:** 2026-02-07
**Goal:** Incentivize model to push beyond plateau by only rewarding high accuracy (>=93%)
**Status:** Superseded by Exp 1.5 (episode_done bug, reward never fired)

### Critical Bug Found

Episode-end rewards were **never applied** because `state["episode_done"]` was never set to True.
The `check_done()` method returned True (max_turns) but didn't set the flag.
Only the per-step -0.1 tool call penalty was active.

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.4-titanic-0207
- **Trajectories:** `outputs/trajectories_exp1.4/`

---

## Experiment 1.5: Episode-Done Fix + Submit + Diversity Nudge

**Date:** 2026-02-07
**Goal:** Fix episode_done bug so rewards actually fire; add submit instruction; encourage model diversity
**Status:** Completed (200 steps) - No learning due to zero-variance reward

### Key Fixes (vs Exp 1.4)

1. **episode_done bug fix** - `check_done()` now sets `state["episode_done"] = True` before returning True
2. **Submit instruction in prompt** - ~75% of trajectories exit via submit
3. **Diversity nudge in prompt** - "Try different models (RF, XGB, LR, SVM, etc.)"

### Reward Function

Same as Exp 1.4 (threshold-based):

| Condition | Reward |
|-----------|--------|
| Correct tool call | 0.0 |
| Wrong tool call | -0.1 |
| Final accuracy >= 0.93 | +1.0 |
| Mediocre improvement (above baseline, < 0.93) | -0.2 |
| No improvement (at or below baseline) | -1.0 |

### Results (200 steps)

- 2720 trajectories, 92% got reward -0.2 (mediocre), 4.4% got +1.0 (>=0.93), 3.5% got -1.0
- **No learning**: 26% of batches had all-same reward → zero GRPO advantage → zero gradient
- Score flat across training: mean ~0.86 from start to finish
- Max accuracy achieved: 0.974 (but too rare to consistently guide learning)

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.5-titanic-0207
- **Trajectories:** `outputs/trajectories_exp1.5/`

---

## Experiment 1.6: Continuous GRPO-Style Reward

**Date:** 2026-02-09
**Goal:** Fix zero-variance reward problem with continuous reward in [-1, 1]
**Status:** Completed (200 steps) - Policy collapsed after step 100

### Reward Function

```
reward = improvement_reward + format_penalty, clipped to [-1, 1]

improvement_reward = improvement * 5 - 0.5
    0% improvement  → -0.5
    10% improvement →  0.0
    20% improvement → +0.5
    30%+ improvement → +1.0

format_penalty = -(error_count / total_steps) * 0.5

No validation at all → -1.0
```

### Results (200 steps)

- **Every batch had gradient signal** (0% zero-gradient batches vs 26% in exp 1.5)
- Peaked at steps 80-100: mean 0.890, max 0.986, 17% trajectories >=0.90
- **Policy collapsed after step ~100**: mean dropped from 0.890 → 0.852, seq_len from 5500 → 1300 tokens
- Model learned to generate shorter (degenerate) responses
- Likely cause: learning rate too high or no KL penalty to anchor to reference policy

### W&B

- **Project:** mlgym-rl
- **Run Name:** exp1.6-titanic-0208
- **Trajectories:** `outputs/trajectories_exp1.6/`

---

## GPT-5.2 Benchmark

**Date:** 2026-02-09
**Goal:** Establish ceiling performance on titanic task with frontier model
**Method:** MLGym `run.py` with `litellm:openai/gpt-5.2`, 50 max_steps, temp 0.7

### Results

| Metric | Value |
|--------|-------|
| Accuracy | **92.8%** |
| Improvement over baseline | +16.3% |
| Steps used | 25 |
| Cost | $0.55 |

### Comparison

| Model | Best Accuracy | Mean Accuracy |
|-------|--------------|---------------|
| GPT-5.2 (50 steps) | 92.8% | 92.8% |
| Qwen3-4B (exp 1.6 peak) | **98.6%** | 89.0% |
| Qwen3-4B (exp 1.6 π₀) | 93.1% | 87.0% |
| Baseline (majority class) | — | 76.6% |

**Key insight:** Qwen3-4B peak exceeds GPT-5.2, meaning model capacity is sufficient. The challenge is training stability.

---

## Experiment Roadmap

### Phase 1: Baselines (No Training)

#### Exp 2.0: Zero-Shot Baseline
**Goal:** Measure base model performance without any training
- Run model on MLGym tasks with simple prompt
- Record validation scores across tasks
- Establishes lower bound for comparison

#### Exp 2.1: Exploration Prompt Baseline
**Goal:** Test if prompting alone improves exploration
- Prompt model to explore diverse solutions
- Instruct: "Don't stop until best gains are reached"
- Compare to zero-shot baseline

#### Exp 2.2: Exploration + Memory Baseline
**Goal:** Test if logging past attempts helps
- Same exploration prompt as 2.1
- Add log of previously tried approaches to context
- Model can see what didn't work and avoid repeating

---

### Phase 2: RL Training with Sparse Rewards

#### Exp 3.0: GRPO/AIPO with Performance Delta
**Goal:** Train with simple improvement-based rewards
- Reward = (final_score - baseline_score)
- Use GRPO or AIPO algorithm
- Compare to prompting baselines

---

### Phase 3: Dense Rewards

#### Exp 4.0: Step-wise Dense Rewards
**Goal:** Provide feedback at every step, not just episode end
- Current reward function: +0.5 correct tool, -0.5 error, +10×improvement
- Include examples showing iterative improvement
- Goal: prevent model from "giving up" early

#### Exp 4.1: Dense Rewards + Diversity Bonus
**Goal:** Encourage exploration of different strategies
- Add diversity term to reward function
- Reward novel actions/approaches
- Penalize repetitive behavior

#### Exp 4.2: Diversity + Memory + Dense Rewards
**Goal:** Combine all improvements
- Dense step-wise rewards
- Diversity bonus for novel approaches
- Log of past attempts in context

---

### Phase 4: MCTS-Based Methods

#### Exp 5.0: MCTS Tree Search Inference
**Goal:** Use MCTS for better action selection at inference time
- Build search tree over possible actions
- UCT for exploration-exploitation balance
- Select best trajectory from tree

#### Exp 5.1: MCTS + Diversity in UCT
**Goal:** Inject diversity into tree search
- Modify UCT formula to favor diverse branches
- Diversity-aware exploration prompt
- Compare to standard MCTS

#### Exp 5.2: Train on MCTS Trajectories
**Goal:** Distill MCTS search into the model
- Generate high-quality trajectories using MCTS
- Train model to imitate MCTS-selected actions
- Goal: get MCTS-quality results without search overhead

---

## Success Metrics

| Metric | Description |
|--------|-------------|
| Validation Score | Task-specific metric (accuracy, F1, etc.) |
| Improvement over Baseline | Delta from zero-shot performance |
| Sample Efficiency | Steps/trajectories needed to reach target |
| Diversity | Unique strategies discovered across rollouts |
| Consistency | Variance in final scores across runs |
