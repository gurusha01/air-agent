"""
W&B logging utilities for MLGym RL training.

This module provides helper functions for logging training metrics,
trajectories, and visualizations to Weights & Biases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def log_trajectory_to_wandb(
    trajectory: dict,
    step: int | None = None,
    prefix: str = "trajectory",
) -> None:
    """
    Log a single trajectory to W&B.

    Args:
        trajectory: Trajectory dict with metrics, steps, etc.
        step: Training step (optional)
        prefix: Metric prefix
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    task = trajectory.get("task", "unknown")
    metrics = trajectory.get("metrics", {})

    log_data = {
        f"{prefix}/{task}/improvement": metrics.get("improvement", 0),
        f"{prefix}/{task}/final_score": metrics.get("final_score", 0),
        f"{prefix}/{task}/baseline_score": metrics.get("baseline_score", 0),
        f"{prefix}/{task}/num_steps": len(trajectory.get("trajectory", [])),
    }

    if step is not None:
        wandb.log(log_data, step=step)
    else:
        wandb.log(log_data)


def log_batch_metrics(
    trajectories: list[dict],
    step: int,
    prefix: str = "batch",
) -> None:
    """
    Log aggregated metrics for a batch of trajectories.

    Args:
        trajectories: List of trajectory dicts
        step: Training step
        prefix: Metric prefix
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    if not trajectories:
        return

    # Aggregate metrics
    improvements = [t.get("metrics", {}).get("improvement", 0) for t in trajectories]
    final_scores = [t.get("metrics", {}).get("final_score", 0) for t in trajectories]

    log_data = {
        f"{prefix}/mean_improvement": sum(improvements) / len(improvements),
        f"{prefix}/max_improvement": max(improvements),
        f"{prefix}/min_improvement": min(improvements),
        f"{prefix}/mean_final_score": sum(final_scores) / len(final_scores),
        f"{prefix}/num_trajectories": len(trajectories),
    }

    # Per-task breakdown
    task_groups: dict[str, list] = {}
    for t in trajectories:
        task = t.get("task", "unknown")
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(t.get("metrics", {}).get("improvement", 0))

    for task, task_improvements in task_groups.items():
        log_data[f"{prefix}/{task}/mean_improvement"] = sum(task_improvements) / len(task_improvements)

    wandb.log(log_data, step=step)


def create_trajectory_table(trajectories: list[dict]) -> "wandb.Table":
    """
    Create a W&B Table from trajectories for detailed inspection.

    Args:
        trajectories: List of trajectory dicts

    Returns:
        wandb.Table with trajectory data
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not available")

    columns = ["task", "improvement", "final_score", "baseline_score", "num_steps", "trajectory_preview"]
    data = []

    for traj in trajectories:
        metrics = traj.get("metrics", {})
        steps = traj.get("trajectory", [])

        # Create preview of first few actions
        preview = ""
        for step in steps[:3]:
            completion = step.get("completion", "")[:100]
            preview += f"• {completion}...\n"

        data.append([
            traj.get("task", "unknown"),
            metrics.get("improvement", 0),
            metrics.get("final_score", 0),
            metrics.get("baseline_score", 0),
            len(steps),
            preview,
        ])

    return wandb.Table(columns=columns, data=data)


def log_trajectory_table(
    trajectories: list[dict],
    step: int,
    table_name: str = "trajectories",
) -> None:
    """
    Log a trajectory table to W&B.

    Args:
        trajectories: List of trajectory dicts
        step: Training step
        table_name: Name for the table artifact
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    table = create_trajectory_table(trajectories)
    wandb.log({table_name: table}, step=step)


def init_wandb_run(
    project: str,
    name: str,
    config: dict | None = None,
    tags: list[str] | None = None,
) -> "wandb.sdk.wandb_run.Run | None":
    """
    Initialize a W&B run for training.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dict
        tags: Optional tags

    Returns:
        W&B run object or None if wandb not available
    """
    if not WANDB_AVAILABLE:
        return None

    return wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags or ["mlgym", "rl-training"],
    )
