# Quick Start Guide

## TL;DR

```bash
# 1. Setup veRL (one-time)
cd /home/ubuntu/MLScientist/air-agent
./setup_verl.sh

# 2. Configure environment
export MLGYM_CONFIG_ROOT="/home/ubuntu/MLScientist/MLGym/configs"
export MLGYM_WORKSPACE_PATH="/home/ubuntu/MLScientist/MLGym/workspace"

# 3. Run training
uv run python -m air.verl_orchestrator
```

See `VERL_SETUP.md` for detailed setup instructions.

## What This Does

Trains a language model on MLGym tasks using GRPO (Group Relative Policy Optimization):

1. **Rollout**: Model interacts with MLGym environment for 50 steps
2. **Collect**: Gathers 16 trajectories (4 groups of 4)
3. **Train**: Updates model weights using GRPO
4. **Repeat**: Server auto-updates, next rollout uses improved model

## Key Innovation

**No manual server restarts!** veRL automatically syncs updated weights to the vLLM inference server after each training step.

## File Overview

- `verl_orchestrator.py` - Main training loop (~100 lines)
- `rollout.py` - Episode collection with token masking (~200 lines)
- `README_VERL.md` - Full documentation

## Configuration

Edit `verl_orchestrator.py`:

```python
# Line 29-34: Basic settings
task = "battleOfSexes"
base_model = "Qwen/Qwen3-4B-Instruct-2507"
num_iterations = 10
rollouts_per_iteration = 16
group_size = 4
max_steps_per_episode = 50

# Line 48-61: veRL config
config = GRPOConfig(
    vllm_gpu_ids="0,1",           # Inference GPUs
    trainer_gpu_ids="2,3,4,5,6,7", # Training GPUs
    learning_rate=1e-5,
    lora_rank=16,
    ...
)
```

## Expected Output

```
Iteration 1/10
============================================================
Collecting group 1/4...
Collecting group 2/4...
Collecting group 3/4...
Collecting group 4/4...

Results:
  Mean reward: 0.234
  Mean steps: 38.5
  Training stats: {'loss': 0.523, 'kl': 0.012}
Checkpoint saved to checkpoints/verl_grpo_v0

Iteration 2/10
============================================================
...
```

Rewards should increase over iterations.

## GPU Requirements

Minimum: 2 GPUs (1 for vLLM, 1 for training)
Recommended: 8 GPUs (2 for vLLM, 6 for training)

## Troubleshooting

**"ModuleNotFoundError: verl"**
```bash
pip install verl
```

**"CUDA out of memory"**
- Reduce `rollouts_per_iteration`
- Use fewer GPUs: `trainer_gpu_ids="2,3"`

**"Can't connect to vLLM"**
- Check GPUs: `nvidia-smi`
- Increase startup wait in veRL config

**"Reward not improving"**
- Lower learning rate: `learning_rate=1e-6`
- Increase group size: `group_size=8`
- Add reward shaping in `rollout.py:get_trajectory_reward_from_env()`

## Next Steps

1. **Test**: Run 1-2 iterations on small task
2. **Monitor**: Watch reward progression
3. **Scale**: Increase rollouts and iterations
4. **Experiment**: Try different tasks and hyperparameters

