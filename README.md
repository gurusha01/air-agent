# air-agent

Training AI Research Agents with Reinforcement Learning and Inference-Time Search using MLGym + prime-rl. Maybe we will do some test time training. 

For my analysis and thourhg peocess, please look at the experiment_logs folder

## Overview

This project trains and evaluates language models on ML research tasks (data science, game theory, regression) via two approaches:

1. **RL Training (Exp 1)**: GRPO-based training with prime-rl on MLGym environments
2. **Tree Search with Verbalized Sampling (Exp 2)**: Inference-time BFS tree search where the model proposes diverse strategies at each branch point
3. **Adaptive Explore-Exploit Tree Search (Exp 3)**: Extends Exp 2 with per-node explore/exploit decisions and past-attempt-aware prompting

## Project Structure

```
air-agent/
├── air/
│   ├── mlgym_env.py                 # MLGym environment wrapper (Exp 1)
│   ├── tree_search.py               # Tree search with verbalized sampling (Exp 2, 3)
│   ├── tree_viewer.py               # Streamlit tree viewer
│   ├── compare_trajectories.py      # Streamlit trajectory comparator (Exp 1)
│   ├── view_trajectory.py           # Terminal trajectory viewer (Exp 1)
│   ├── wandb_logging.py             # W&B logging utilities
│   ├── run_multitask_experiments.sh  # Batch runner for multi-task tree search
│   └── __init__.py
├── configs/mlgym/
│   ├── rl_full.toml                 # Main RL training config (Exp 1.6)
│   ├── rl.toml                      # Base RL config
│   ├── rl_debug.toml                # Debug config (10 steps)
│   ├── infer.toml                   # vLLM inference server config
│   ├── orch.toml                    # Orchestrator config
│   └── train.toml                   # Trainer config
├── experiment_logs/
│   ├── experiments.md               # Exp 1 (RL training) logs
│   ├── experiments2.md              # Exp 2 (tree search) logs and results
│   ├── experiments3.md              # Exp 3 (adaptive search) design doc
│   ├── log_book.md                  # W&B run tracking
│   └── PROJECT_NOTES.md             # Troubleshooting guide (12+ issues)
├── air_mlgym.py                     # Entry point for prime-rl (env_id="air_mlgym")
├── tests/
│   ├── test_mlgym_env.py
│   └── test_integration.py
└── pyproject.toml
```

## Quick Start

### Install

```bash
cd air-agent
uv sync
docker pull aigym/mlgym-agent:latest
```

### Exp 2: Tree Search (recommended starting point)

```bash
# Terminal 1: Start vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 32768

# Terminal 2: Run tree search (from MLGym dir)
cd /path/to/MLGym
uv run --project /path/to/air-agent \
    python /path/to/air-agent/air/tree_search.py \
    --branching-factor 3 --max-depth 2 --max-actions 15 \
    --sampling-mode tail --verbose \
    --task-config tasks/titanic.yaml
```

### Exp 2: Batch runs (multi-task, multi-mode)

```bash
cd /path/to/MLGym
# Run all 4 modes x 5 runs x 3 tasks (or filter by task)
bash /path/to/air-agent/air/run_multitask_experiments.sh
bash /path/to/air-agent/air/run_multitask_experiments.sh houseprice  # single task
```

### Exp 1: RL Training

```bash
cd /path/to/MLGym
uv run --project /path/to/air-agent rl @ /path/to/air-agent/configs/mlgym/rl_full.toml
```

## Key Results

### Tree Search (Exp 2) — Cross-Task Ranking

| Sampling Mode | Titanic (acc) | Battle of Sexes (score) | House Price (R²) | Avg Rank |
|---|---|---|---|---|
| **Tail VS** | **0.943** | **1.422** | 0.899 | **1.33** |
| Uniform VS | 0.941 | 1.368 | 0.892 | 2.33 |
| No VS | 0.884 | 1.230 | **0.902** | 3.00 |
| Local VS | 0.897 | 1.327 | 0.891 | 3.33 |

Tail verbalized sampling (asking for low-probability, unusual strategies) wins on 2/3 tasks and is the overall best mode. See `experiment_logs/experiments2.md` for full analysis.

### RL Training (Exp 1)

- Peak accuracy: 98.6% on titanic (Exp 1.6, GRPO with continuous rewards)
- Policy collapsed after step 80-100. See `experiment_logs/experiments.md`.

## GPU Allocation

| GPU | Usage |
|-----|-------|
| 0 | vLLM inference server |
| 2-6 | FSDP trainer (Exp 1 only) |
| 7 | MLGym Docker container |

## Monitoring

```bash
# Tree search viewer
uv run streamlit run air/tree_viewer.py --server.port 8502

# RL training logs
tail -f outputs/logs/trainer.stdout
tail -f outputs/logs/orchestrator.stdout

# Trajectory comparator
uv run streamlit run air/compare_trajectories.py --server.port 8501
```

## Troubleshooting

See `experiment_logs/PROJECT_NOTES.md` for 12+ documented issues and fixes.

## License

MIT
