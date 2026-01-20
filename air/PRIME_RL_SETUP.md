# Prime-RL + MLGym Integration Guide

This guide explains how to train AI research agents on MLGym tasks using prime-rl's distributed RL training infrastructure.

## Overview

The integration consists of:

1. **MLGym Environment Wrapper** (`air/mlgym_env.py`) - Adapts MLGym environments to work with prime-rl's verifiers interface
2. **MLGym Orchestrator** (`air/prime_orchestrator.py`) - Coordinates rollout collection from MLGym environments
3. **Configuration Files** (`configs/mlgym/`) - TOML configs for prime-rl components

## Quick Setup

### 1. Ensure Dependencies Are Cloned

Your directory structure should look like:
```
/data4/parth/
├── air-agent/    (this repo)
├── MLGym/        (clone from https://github.com/facebookresearch/MLGym)
└── prime-rl/     (clone from https://github.com/PrimeIntellect-ai/prime-rl)
```

If not already cloned:
```bash
cd /data4/parth
git clone https://github.com/facebookresearch/MLGym.git
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
```

### 2. Install Dependencies

```bash
cd /data4/parth/air-agent

# Remove old lock file if it exists
rm -f uv.lock

# Sync dependencies
uv sync
```

### 3. Set Environment Variables

```bash
export MLGYM_CONFIG_ROOT="/data4/parth/MLGym/configs"
export MLGYM_WORKSPACE_PATH="/data4/parth/MLGym/workspace"

# Optional: Set OpenAI API key for vLLM server
export OPENAI_API_KEY="dummy"
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

### 4. Verify Installation

```bash
uv run python -c "
import air
import mlgym
import prime_rl
print('✓ All imports successful!')
print(f'  air: {air.__version__}')
"
```

## Running Training

### Option 1: Standalone Mode (Simpler)

This mode runs the orchestrator directly without the full prime-rl distributed setup:

```bash
# 1. Start vLLM inference server (Terminal 1)
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 --trust-remote-code

# 2. Run standalone training (Terminal 2)
uv run python -m air.prime_orchestrator \
    --task battleOfSexes \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-steps 10 \
    --batch-size 16 \
    --output-dir outputs/standalone_run
```

Or use the run script:
```bash
# Check environment first
uv run python -m air.run_training check

# Run standalone training
uv run python -m air.run_training standalone \
    --task battleOfSexes \
    --steps 10
```

### Option 2: Full Distributed Mode (Production)

For full distributed training with weight synchronization:

```bash
# Terminal 1: Start vLLM Inference Server
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 --trust-remote-code \
    --enable-lora

# Terminal 2: Start RL Trainer
uv run trainer @ configs/mlgym/train.toml

# Terminal 3: Start Orchestrator
uv run orchestrator @ configs/mlgym/orch.toml
```

## Configuration

### Orchestrator Config (`configs/mlgym/orch.toml`)

Key settings:
- `batch_size`: Number of samples per training batch
- `rollouts_per_example`: Number of rollouts per problem
- `max_steps`: Maximum training steps
- `env`: Environment configuration (MLGym task settings)

### Trainer Config (`configs/mlgym/train.toml`)

Key settings:
- `model`: Model configuration (name, LoRA settings)
- `optim`: Optimizer settings (learning rate, etc.)
- `loss`: Loss function settings

## Architecture

```
┌─────────────────────────────────────────┐
│        Prime-RL Trainer                 │
│  - FSDP2 distributed training           │
│  - LoRA/full fine-tuning                │
│  - Weight broadcast to inference        │
└──────────────────┬──────────────────────┘
                   │ Training batches
                   ▼
┌─────────────────────────────────────────┐
│    MLGym Orchestrator                   │
│  - Collects rollouts from environments  │
│  - Computes GRPO advantages             │
│  - Prepares training batches            │
└──────────────────┬──────────────────────┘
                   │ Rollout requests
                   ▼
┌─────────────────────────────────────────┐
│   vLLM Inference Server                 │
│  - Fast batched inference               │
│  - LoRA adapter hot-loading             │
│  - Automatic weight updates             │
└──────────────────┬──────────────────────┘
                   │ Generations
                   ▼
┌─────────────────────────────────────────┐
│   MLGym Environment                     │
│  - Docker containerized execution       │
│  - Multi-step agent interaction         │
│  - Reward from task evaluation          │
└─────────────────────────────────────────┘
```

## Files Created

| File | Description |
|------|-------------|
| `air/mlgym_env.py` | Verifiers-compatible MLGym environment wrapper |
| `air/prime_orchestrator.py` | Orchestrator for collecting MLGym rollouts |
| `air/run_training.py` | CLI script for running training |
| `configs/mlgym/orch.toml` | Orchestrator configuration |
| `configs/mlgym/train.toml` | Trainer configuration |
| `configs/mlgym/infer.toml` | Inference server configuration |

## Troubleshooting

### "Module not found: mlgym"
```bash
# Ensure MLGym is cloned and uv sync has been run
cd /data4/parth/air-agent
uv sync
```

### "Connection refused to vLLM server"
```bash
# Ensure server is running on the correct port
curl http://localhost:8000/v1/models
```

### "CUDA out of memory"
- Reduce batch size in config
- Use smaller model or enable LoRA
- Check GPU memory with `nvidia-smi`

### "Docker connection error" (MLGym)
```bash
# Ensure Docker is running and accessible
docker ps
```

## Next Steps

1. **Test with small runs**: Start with 5-10 steps to verify setup
2. **Monitor rewards**: Check `outputs/*/rollouts/` for saved trajectories
3. **Scale up**: Increase batch size and training steps
4. **Experiment**: Try different MLGym tasks and hyperparameters

## References

- [Prime-RL Documentation](https://github.com/PrimeIntellect-ai/prime-rl)
- [MLGym Repository](https://github.com/facebookresearch/MLGym)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
