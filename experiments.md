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
| LoRA Alpha | 32 |
| Learning Rate | 1e-5 |
| max_steps | 10 |
| batch_size | 8 |
| rollouts_per_example | 4 |
| max_turns | 10 |
| num_examples_per_task | 10 |
| seq_len | 4096 |
| Tasks | titanic only |

### GPU Allocation

| GPU | Usage |
|-----|-------|
| 0 | vLLM inference server |
| 2-6 | FSDP trainer (5 GPUs) |
| 7 | MLGym Docker container |

### Results

| Metric | Value |
|--------|-------|
| Training Steps Completed | 10/10 |
| Container Errors | 0 |
| Checkpoints Saved | Steps 5, 10 |
| Throughput | ~500-670 tokens/s |
| Peak Memory | 6.7 GiB |
| Total Time | ~25 minutes |
| Loss | 0.0 (reward issue) |

### Observations

1. **Loss = 0.0**: Reward signal not flowing properly
   - Root cause: Model doesn't call `validate` command frequently
   - Reward only computed on validation score changes

2. **Infrastructure Working**:
   - Container stability fix verified (0 flake8 errors)
   - Orchestrator-trainer communication working
   - Trajectory saving working

### Artifacts

- Checkpoints: `outputs/checkpoints/step_5/`, `outputs/checkpoints/step_10/`
- Trajectories: `outputs/trajectories_debug/`
- W&B Project: mlgym-rl-debug

---

## Experiment 1.1: Full Multi-Task Training

**Date:** 2026-02-05
**Goal:** Large-scale training across multiple MLGym tasks
**Status:** Running

### Configuration

| Parameter | Value |
|-----------|-------|
| Config File | `configs/mlgym/rl_full.toml` |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 1e-5 |
| max_steps | 200 (~20 epochs) |
| batch_size | 32 |
| rollouts_per_example | 8 |
| max_turns | 50 |
| num_examples_per_task | 10 |
| seq_len | 8192 |
| Checkpoint Interval | 20 steps |

### Tasks (4 total)

| Task | Type | Config |
|------|------|--------|
| titanic | Tabular Classification | titanic.yaml |
| prisonersDilemma | Game Theory | prisonersDilemma.yaml |
| imageClassificationFMnist | Image Classification | imageClassificationFMnist.yaml |
| regressionKaggleHousePrice | Regression | regressionKaggleHousePrice.yaml |

### Training Math

```
Total examples = 4 tasks × 10 examples = 40 examples
Examples per step = batch_size / rollouts_per_example = 32 / 8 = 4
Steps per epoch = 40 / 4 = 10 steps
Total epochs = 200 / 10 = 20 epochs
```

### GPU Allocation

| GPU | Usage |
|-----|-------|
| 0 | vLLM inference server |
| 2-6 | FSDP trainer (5 GPUs) |
| 7 | MLGym Docker container |

### Expected Runtime

- ~2 hours per epoch (estimate based on debug run)
- Total: ~40 hours for 20 epochs

### Artifacts

- Checkpoints: `outputs/checkpoints/step_20/`, `step_40/`, etc.
- Trajectories: `outputs/trajectories_exp1.1/`
- W&B Project: mlgym-rl
- W&B Run Name: exp1.1-full-multi-task

### Command

```bash
# Run in tmux session
tmux new-session -d -s mlgym_training

tmux send-keys -t mlgym_training "cd /home/ubuntu/MLScientist/MLGym && \
source /home/ubuntu/MLScientist/air-agent/.venv/bin/activate && \
uv run --project /home/ubuntu/MLScientist/air-agent rl @ \
/home/ubuntu/MLScientist/air-agent/configs/mlgym/rl_full.toml 2>&1 | \
tee /home/ubuntu/MLScientist/air-agent/outputs/exp1.1_training.log" Enter
```

### Monitoring

```bash
# Attach to tmux
tmux attach -t mlgym_training

# Check trainer progress
tail -f outputs/logs/trainer.stdout

# Check orchestrator
tail -f outputs/logs/orchestrator.stdout

# Check W&B
# https://wandb.ai/Gurusha-personal/mlgym-rl
```

### Results

| Metric | Value |
|--------|-------|
| Training Steps Completed | TBD |
| Final Loss | TBD |
| Final Reward | TBD |
| Total Time | TBD |

### Notes

- Started: 2026-02-05 ~03:XX UTC
- Running in tmux session: `mlgym_training`

---

## Future Experiments

### Experiment 2: Reward Signal Improvements (Planned)

**Goal:** Get non-zero rewards flowing through training

Options to explore:
1. Episode-end rewards based on final validation score
2. Add `validate` to required action sequence
3. Dense rewards based on code quality metrics

### Experiment 3: Scaling Study (Planned)

**Goal:** Test larger models and longer training

- Qwen3-8B model
- 1000+ training steps
- More tasks
