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
