# air-agent

Training AI Research Agents with Reinforcement Learning using MLGym + prime-rl.

## Overview

This project trains language models to solve ML research tasks (data science, model training, hyperparameter tuning) using reinforcement learning. The agent interacts with MLGym environments through Docker containers, receiving rewards based on validation score improvements.

## Project Structure

```
air-agent/
├── air/
│   ├── mlgym_env.py          # Verifiers-compatible MLGym environment wrapper
│   ├── visualize.py          # Streamlit trajectory visualizer
│   └── __init__.py
├── configs/
│   └── mlgym/
│       ├── rl.toml           # Full training config
│       └── rl_debug.toml     # Debug config (10 steps, 1 task)
├── air_mlgym.py              # Entry point for prime-rl (env_id="air_mlgym")
├── simple_env.py             # Simple test environment (no Docker)
├── experiments.md            # Experiment logs and results
├── PROJECT_NOTES.md          # Issues, solutions, and debugging tips
└── tests/
    └── test_mlgym_env.py
```

## Dependencies

- **MLGym**: ML research benchmark with Docker-based task environments
- **prime-rl**: Distributed RL training infrastructure
- **verifiers**: Multi-turn environment interface

All dependencies are installed via editable installs in `pyproject.toml`.

## Quick Start

### 1. Install

```bash
cd air-agent
uv sync
```

### 2. Pull MLGym Docker Image

```bash
docker pull aigym/mlgym-agent:latest
```

### 3. Clean GPU Processes (Important!)

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}
```

### 4. Run Training

**IMPORTANT:** Run from the MLGym directory so subprocesses can find task configs:

```bash
cd /home/ubuntu/MLScientist/MLGym
source /home/ubuntu/MLScientist/air-agent/.venv/bin/activate

# Debug run (10 steps, ~25 minutes)
uv run --project /home/ubuntu/MLScientist/air-agent rl @ /home/ubuntu/MLScientist/air-agent/configs/mlgym/rl_debug.toml

# Full run
uv run --project /home/ubuntu/MLScientist/air-agent rl @ /home/ubuntu/MLScientist/air-agent/configs/mlgym/rl.toml
```

### 5. Monitor Training

```bash
# Trainer progress
tail -f outputs/logs/trainer.stdout

# Orchestrator/environment logs
tail -f outputs/logs/orchestrator.stdout
```

### 6. Visualize Trajectories

```bash
uv run streamlit run air/visualize.py -- --trajectory_dir outputs/trajectories_debug
```

## GPU Allocation (8x A10G Example)

The system requires separate GPUs for inference, training, and the MLGym Docker container:

| GPU | Usage |
|-----|-------|
| 0 | vLLM inference server |
| 1 | (skip - often occupied) |
| 2-6 | FSDP trainer (5 GPUs) |
| 7 | MLGym Docker container |

Configure in `rl_debug.toml`:
```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [2, 3, 4, 5, 6]

[[orchestrator.env]]
args = { env_gpu = "7", ... }
```

## Configuration

Key parameters in `configs/mlgym/rl_debug.toml`:

| Parameter | Description | Debug Value |
|-----------|-------------|-------------|
| `max_steps` | Training gradient updates | 10 |
| `batch_size` | Trajectories per training step | 8 |
| `rollouts_per_example` | Rollouts per prompt | 4 |
| `max_turns` | Agent actions per trajectory | 10 |
| `num_examples_per_task` | Prompts in dataset pool | 10 |
| `task_configs` | MLGym tasks to train on | ["titanic.yaml"] |

## Key Features

- **Branching trajectory strategy**: Each turn is a separate training sample
- **Delta improvement rewards**: `reward = current_score - previous_score` at each `validate`
- **Container lifecycle management**: Automatic container reset and recovery
- **Multiple tasks**: titanic, prisonersDilemma, imageClassificationFMnist, etc.
- **W&B logging**: Training metrics and sample trajectories
- **Trajectory saving**: JSON files for visualization and debugging

## Troubleshooting

See [PROJECT_NOTES.md](PROJECT_NOTES.md) for detailed debugging information.

Common issues:
1. **"pip folder not found"** - Fixed in mlgym_env.py (cd before reset)
2. **GPU OOM** - Kill old vLLM processes before starting
3. **Trainer hangs** - Clean up stale processes from previous runs
4. **Task configs not found** - Run from MLGym directory

## Experiment Results

See [experiments.md](experiments.md) for experiment logs.

### Debug Run (2026-02-05)
- 10/10 training steps completed
- 0 container errors
- Infrastructure verified working
- Reward signal needs fixing (Loss = 0)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   vLLM Server   │────▶│   Orchestrator   │────▶│     Trainer     │
│    (GPU 0)      │     │   (generates     │     │   (FSDP, GPUs   │
│                 │◀────│    rollouts)     │◀────│     2-6)        │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  MLGym Docker    │
                        │  Container       │
                        │  (GPU 7)         │
                        │                  │
                        │  - titanic       │
                        │  - prisonersDil  │
                        │  - fmnist        │
                        └──────────────────┘
```

## License

MIT
