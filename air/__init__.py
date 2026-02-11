"""air-agent: AI Research Agents with Reinforcement Learning.

This package provides tools for training AI research agents using
reinforcement learning on MLGym environments with prime-rl.
"""

from air.mlgym_env import (
    MLGymEnvironment,
    compute_delta_reward,
    metric_final_accuracy,
    metric_improvement,
    load_environment,
)
from air.wandb_logging import (
    init_wandb_run,
    log_batch_metrics,
    log_trajectory_table,
    log_trajectory_to_wandb,
)

__all__ = [
    # Environment
    "MLGymEnvironment",
    "compute_delta_reward",
    "metric_final_accuracy",
    "metric_improvement",
    "load_environment",
    # W&B logging
    "init_wandb_run",
    "log_batch_metrics",
    "log_trajectory_table",
    "log_trajectory_to_wandb",
]

__version__ = "0.0.1"
