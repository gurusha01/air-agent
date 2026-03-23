"""air-agent: AI Research Agents with Reinforcement Learning.

This package provides tools for training AI research agents using
reinforcement learning on MLGym environments with prime-rl.
"""

def __getattr__(name):
    """Lazy imports to avoid requiring verifiers/prime-rl for all submodules."""
    _mlgym_names = {"MLGymEnvironment", "compute_delta_reward", "metric_final_accuracy",
                    "metric_improvement", "load_environment"}
    _wandb_names = {"init_wandb_run", "log_batch_metrics", "log_trajectory_table",
                    "log_trajectory_to_wandb"}
    if name in _mlgym_names:
        from air.mlgym_env import (MLGymEnvironment, compute_delta_reward,
                                    metric_final_accuracy, metric_improvement, load_environment)
        return locals()[name]
    if name in _wandb_names:
        from air.wandb_logging import (init_wandb_run, log_batch_metrics,
                                        log_trajectory_table, log_trajectory_to_wandb)
        return locals()[name]
    raise AttributeError(f"module 'air' has no attribute {name!r}")

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
