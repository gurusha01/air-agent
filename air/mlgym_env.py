"""
MLGym Environment for prime-rl training.

This module provides a verifiers-compatible environment that wraps MLGym tasks
for multi-turn RL training with branching trajectory strategy.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset
from loguru import logger

import verifiers as vf

# Add MLGym to path - use absolute path to work in spawned subprocesses
MLGYM_PATH = Path("/home/ubuntu/MLScientist/MLGym").resolve()
if str(MLGYM_PATH) not in sys.path:
    sys.path.insert(0, str(MLGYM_PATH))

from mlgym.environment.env import EnvironmentArguments, MLGymEnv

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass
class TaskMetrics:
    """Tracks metrics for a task during an episode."""

    baseline_score: float = 0.0
    current_score: float = 0.0
    previous_score: float = 0.0
    validation_scores: list[float] = field(default_factory=list)
    validation_steps: list[int] = field(default_factory=list)


def normalize_task_config_path(task_config: str) -> str:
    """
    Normalize task config path to be relative to MLGym/configs/.

    Examples:
        "titanic.yaml" -> "tasks/titanic.yaml"
        "tasks/titanic.yaml" -> "tasks/titanic.yaml"
    """
    if not task_config.startswith("tasks/"):
        return f"tasks/{task_config}"
    return task_config


def get_dataset_builder(
    task_configs: list[str],
    num_examples_per_task: int = 100,
    seed: int = 42,
):
    """
    Build dataset of MLGym task prompts.

    Args:
        task_configs: List of task config YAML filenames (e.g., "titanic.yaml")
                     Will be normalized to "tasks/titanic.yaml"
        num_examples_per_task: Number of training examples per task
        seed: Random seed for reproducibility

    Returns:
        Callable that builds the dataset
    """
    import random

    def build() -> Dataset:
        random.seed(seed)
        data = []
        example_id_counter = 0  # Unique ID for each example

        for task_config in task_configs:
            # Normalize path
            task_config_normalized = normalize_task_config_path(task_config)
            task_path = Path(task_config)
            task_name = task_path.stem

            # Load task config to get description and baseline
            config_path = MLGYM_PATH / "configs" / task_config_normalized
            if not config_path.exists():
                logger.warning(f"Task config not found: {config_path}")
                continue

            import yaml

            with open(config_path) as f:
                task_yaml = yaml.safe_load(f)

            task_description = task_yaml.get("description", f"Complete the {task_name} task.")
            baseline_scores = task_yaml.get("baseline_scores", [{}])
            baseline_info = baseline_scores[0] if baseline_scores else {}

            # Format baseline info for prompt
            baseline_str = ", ".join(f"{k}: {v}" for k, v in baseline_info.items()) if baseline_info else "Not available"

            # Create multiple examples per task with different seeds
            for i in range(num_examples_per_task):
                example_seed = seed + i

                system_prompt = """You are an ML research agent that outputs ONLY executable commands. No explanations, no thinking, just the command.

AVAILABLE COMMANDS:
- open <file> - View file contents
- edit <start_line>:<end_line>
<new_content>
end_of_edit - Edit file lines
- python <script.py> - Run a Python script
- validate - Evaluate current solution and get score
- submit - Submit final solution

RULES:
1. Output ONLY the command, nothing else
2. No explanations, no comments, no thinking
3. One command per response
4. Use validate frequently to check progress

STRATEGY:
1. open files to explore
2. Make incremental improvements
3. validate after changes
4. submit when done"""

                user_prompt = f"""Task: {task_yaml.get('name', task_name)}

{task_description}

Baseline: {baseline_str}

Output your first command (e.g., open train_and_predict.py):"""

                data.append({
                    "example_id": example_id_counter,  # Required by prime-rl buffer
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "task": task_name,
                    "task_config": task_config_normalized,  # Use normalized path for MLGym
                    "example_seed": example_seed,
                    "info": {
                        "task_name": task_name,
                        "task_config": task_config_normalized,  # Use normalized path
                        "baseline_scores": baseline_info,
                        "task_description": task_description,
                    },
                })
                example_id_counter += 1

        random.shuffle(data)
        return Dataset.from_list(data)

    return build


class MLGymEnvironment(vf.MultiTurnEnv):
    """
    MLGym environment compatible with verifiers/prime-rl.

    This environment wraps MLGym tasks for multi-turn RL training.
    It supports:
    - Branching trajectory strategy (each turn = separate training sample)
    - Delta improvement rewards at validate steps
    - Multiple task configurations
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 50,
        env_gpu: str = "0",
        container_type: str = "docker",
        image_name: str = "aigym/mlgym-agent:latest",
        training_timeout: int = 1800,
        save_trajectories: bool = True,
        trajectory_dir: str = "trajectories",
        **kwargs,
    ):
        """
        Initialize MLGym environment.

        Args:
            dataset: Dataset of task prompts
            rubric: Scoring rubric for rewards
            max_turns: Maximum turns per episode
            env_gpu: GPU ID for MLGym container (ML training)
            container_type: "docker" or "podman"
            image_name: Docker image for MLGym
            training_timeout: Timeout for training commands (seconds)
            save_trajectories: Whether to save trajectories for visualization
            trajectory_dir: Directory to save trajectories
        """
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns, **kwargs)
        self.env_gpu = env_gpu
        self.container_type = container_type
        self.image_name = image_name
        self.training_timeout = training_timeout
        self.save_trajectories = save_trajectories
        self.trajectory_dir = Path(trajectory_dir)

        # Container management
        self._envs: dict[str, MLGymEnv] = {}
        self._metrics: dict[str, TaskMetrics] = {}

        # Create trajectory directory
        if save_trajectories:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def _is_container_alive(self, env: MLGymEnv) -> bool:
        """Check if the container is still running."""
        try:
            if env.container_obj is None:
                return False
            # Refresh container status from Docker
            import docker
            client = docker.from_env()
            container = client.containers.get(env.container_obj.id)
            return container.status == "running"
        except Exception as e:
            logger.debug(f"Container check failed: {e}")
            return False

    def _create_new_env(self, task_config: str) -> MLGymEnv:
        """Create a new MLGym environment."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(MLGYM_PATH)
            logger.debug(f"Changed to MLGym directory: {MLGYM_PATH}")

            env_args = EnvironmentArguments(
                image_name=self.image_name,
                max_steps=self.max_turns * 2,  # Some buffer
                task_config_path=task_config,
                container_type=self.container_type,
                verbose=False,
            )
            env = MLGymEnv(
                args=env_args,
                devices=[self.env_gpu],
            )
            env.reset()
            return env
        finally:
            os.chdir(original_cwd)

    def _get_or_create_env(self, state: vf.State) -> MLGymEnv:
        """
        Get or create MLGym environment for a task.

        Container reuse strategy:
        - Each task_config gets its own container (reused across episodes)
        - Container is reset() between episodes, not recreated
        - If container is dead, recreate it
        """
        import os
        task_config = state.get("info", {}).get("task_config", "tasks/titanic.yaml")
        example_id = state.get("example_id", "default")
        logger.debug(f"[DEBUG] _get_or_create_env called (pid={os.getpid()}), task={task_config}, example={example_id}")

        # Use task_config as container key (not example_id) for container reuse
        container_key = task_config

        # Check if we have an existing container and if it's still alive
        if container_key in self._envs:
            env = self._envs[container_key]
            if not self._is_container_alive(env):
                logger.warning(f"Container for {task_config} is dead, recreating...")
                try:
                    env.close()
                except Exception:
                    pass
                del self._envs[container_key]

        if container_key not in self._envs:
            logger.info(f"Creating new container for task: {task_config}")
            env = self._create_new_env(task_config)
            self._envs[container_key] = env
        else:
            # Reuse existing container, but reset for new episode
            env = self._envs[container_key]
            # Only reset if this is a new example (not continuation of same episode)
            if example_id not in self._metrics:
                logger.debug(f"Resetting container for new episode: {example_id}")
                try:
                    # WORKAROUND: Change to home directory before reset to avoid
                    # "pip folder not found" error. The reset() does rm -rf on
                    # the workspace which is the shell's cwd, breaking pip.
                    env.communicate("cd /home/agent")
                    env.reset()
                except RuntimeError as e:
                    # Container died during reset, recreate it
                    logger.warning(f"Reset failed, recreating container: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = self._create_new_env(task_config)
                    self._envs[container_key] = env

        # Initialize metrics for this example if new
        if example_id not in self._metrics:
            baseline_scores = state.get("info", {}).get("baseline_scores", {})
            baseline_value = list(baseline_scores.values())[0] if baseline_scores else 0.0
            self._metrics[example_id] = TaskMetrics(
                baseline_score=baseline_value,
                previous_score=baseline_value,
                current_score=baseline_value,
            )

        return env

    def _cleanup_env(self, example_id: str) -> None:
        """
        Clean up after episode ends.

        Note: We don't close the Docker container here because containers
        are shared across episodes (keyed by task_config). We only clean up
        the metrics for this specific example_id.
        """
        # Only clean up metrics, not containers (containers are reused)
        if example_id in self._metrics:
            del self._metrics[example_id]

    def close_all_containers(self) -> None:
        """Close all Docker containers. Call this at the end of training."""
        for task_config, env in list(self._envs.items()):
            try:
                logger.info(f"Closing container for task: {task_config}")
                env.close()
            except Exception as e:
                logger.warning(f"Error closing container for {task_config}: {e}")
        self._envs.clear()
        self._metrics.clear()

    def _extract_command(self, raw_output: str) -> str:
        """
        Extract executable command from model output.

        Handles:
        - Qwen3 thinking tags: <think>...</think>
        - Code blocks: ```command```
        - Extra explanatory text before/after commands
        """
        import re

        text = raw_output.strip()

        # Remove Qwen3 thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove code block markers
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = text.replace('```', '')

        # Look for known command patterns
        lines = text.strip().split('\n')
        command_patterns = [
            r'^(open\s+\S+)',
            r'^(edit\s+\d+:\d+)',
            r'^(python\s+\S+)',
            r'^(validate)\s*$',
            r'^(submit)\s*$',
            r'^(exit_forfeit)\s*$',
            r'^(skip)\s*$',
        ]

        for line in lines:
            line = line.strip()
            for pattern in command_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Found a command, return the rest of the text starting from this line
                    # This handles multi-line edit commands
                    idx = text.find(line)
                    if idx >= 0:
                        return text[idx:].strip()

        # If no known command found, return cleaned text as-is
        # (might be an edit command or something else)
        return text.strip()

    @vf.stop
    async def check_done(self, state: vf.State) -> bool:
        """Check if episode should end."""
        if not state.get("trajectory"):
            return False

        # Get last assistant response - completion is a LIST of messages, not a string!
        last_step = state["trajectory"][-1]
        completion = last_step.get("completion", [])

        # Extract the last assistant message content
        last_response = ""
        if completion:
            assistant_msgs = [m for m in completion if m.get("role") == "assistant"]
            if assistant_msgs:
                last_response = assistant_msgs[-1].get("content", "")

        # Check for submit action
        if "<<SUBMISSION||" in last_response or "submit" in last_response.lower().split()[-1:]:
            return True

        # Check for exit conditions
        exit_keywords = ["exit_forfeit", "exit_context", "exit_cost", "exit_error"]
        if any(kw in last_response.lower() for kw in exit_keywords):
            return True

        # Check max turns
        return len(state["trajectory"]) >= self.max_turns

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> vf.Messages:
        """
        Execute agent action in MLGym container and return observation.

        This is where the actual MLGym interaction happens.
        """
        import os
        logger.info(f"[DEBUG] env_response called (pid={os.getpid()}), num_messages={len(messages)}")

        # Get the last assistant message (agent's action)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs:
            logger.info(f"[DEBUG] No assistant messages, returning early")
            return [{"role": "user", "content": "Please provide an action."}]

        raw_action = assistant_msgs[-1]["content"]
        example_id = state.get("example_id", "default")

        # Extract command from model output (handle thinking tags, explanations, etc.)
        action = self._extract_command(raw_action)
        logger.info(f"[DEBUG] Processing action for example_id={example_id}, action={action[:100]}...")

        max_retries = 2
        task_config = state.get("info", {}).get("task_config", "tasks/titanic.yaml")

        for attempt in range(max_retries):
            try:
                # Get or create environment
                logger.info(f"[DEBUG] Getting or creating env (attempt {attempt + 1})...")
                env = self._get_or_create_env(state)
                logger.info(f"[DEBUG] Got env, calling step()...")

                # Execute action in MLGym
                observation, reward, done, info = env.step(action)
                logger.info(f"[DEBUG] step() returned: done={done}, obs_len={len(observation) if observation else 0}")

                # Track validation scores for delta rewards
                if info.get("score"):
                    metrics = self._metrics.get(example_id)
                    if metrics:
                        # Get the score value (handle dict format)
                        score_data = info["score"][-1] if info.get("score") else {}
                        score_value = list(score_data.values())[0] if isinstance(score_data, dict) else score_data

                        metrics.previous_score = metrics.current_score
                        metrics.current_score = score_value
                        metrics.validation_scores.append(score_value)
                        metrics.validation_steps.append(len(state.get("trajectory", [])))

                        # Store delta reward in state for the rubric to use
                        delta_reward = score_value - metrics.previous_score
                        state["last_delta_reward"] = delta_reward
                        state["last_validation_score"] = score_value
                        state["baseline_score"] = metrics.baseline_score

                # Handle episode end
                if done:
                    self._save_trajectory(state, info)
                    self._cleanup_env(example_id)

                return [{"role": "user", "content": observation or "Action executed."}]

            except RuntimeError as e:
                if "Failed to communicate with container" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Container communication failed, forcing recreation (attempt {attempt + 1})")
                    # Force container recreation by removing from cache
                    if task_config in self._envs:
                        try:
                            self._envs[task_config].close()
                        except Exception:
                            pass
                        del self._envs[task_config]
                    continue
                else:
                    logger.exception(f"Error executing action: {e}")
                    return [{"role": "user", "content": f"Error: {e}"}]
            except Exception as e:
                logger.exception(f"Error executing action: {e}")
                return [{"role": "user", "content": f"Error: {e}"}]

        return [{"role": "user", "content": "Error: Max retries exceeded"}]

    def _save_trajectory(self, state: vf.State, info: dict) -> None:
        """Save trajectory for visualization."""
        if not self.save_trajectories:
            return

        example_id = state.get("example_id", "default")
        task_name = state.get("info", {}).get("task_name", "unknown")
        metrics = self._metrics.get(example_id, TaskMetrics())

        trajectory_data = {
            "task": task_name,
            "example_id": example_id,
            "trajectory": state.get("trajectory", []),
            "metrics": {
                "baseline_score": metrics.baseline_score,
                "final_score": metrics.current_score,
                "improvement": metrics.current_score - metrics.baseline_score,
                "validation_history": list(zip(metrics.validation_steps, metrics.validation_scores)),
            },
            "info": info,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }

        # Save to file
        filename = f"{task_name}_{example_id}_{trajectory_data['timestamp']}.json"
        filepath = self.trajectory_dir / filename
        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2, default=str)

        logger.info(f"Saved trajectory to {filepath}")


def compute_delta_reward(completion: list[dict], state: vf.State, **kwargs) -> float:
    """
    Compute reward as delta improvement since last validation.

    This implements the delta improvement reward strategy:
    reward = current_score - previous_score

    If no validation has occurred yet, reward is 0.
    """
    # Get delta reward stored by env_response
    delta_reward = state.get("last_delta_reward", 0.0)

    # Normalize to reasonable range [-1, 1]
    # Most ML tasks have scores in [0, 1] or similar ranges
    # Delta is typically small, so we scale it up
    normalized_reward = max(-1.0, min(1.0, delta_reward * 5))

    return normalized_reward


def compute_final_improvement_reward(completion: list[dict], state: vf.State, **kwargs) -> float:
    """
    Compute reward as total improvement over baseline.

    This can be used as an additional reward signal at episode end.
    """
    final_score = state.get("last_validation_score", 0.0)
    baseline_score = state.get("baseline_score", 0.0)

    improvement = final_score - baseline_score
    return max(-1.0, min(1.0, improvement * 5))


def load_environment(
    task_configs: list[str] | None = None,
    max_turns: int = 50,
    num_examples_per_task: int = 100,
    env_gpu: str = "0",
    container_type: str = "docker",
    image_name: str = "aigym/mlgym-agent:latest",
    training_timeout: int = 1800,
    save_trajectories: bool = True,
    trajectory_dir: str = "trajectories",
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """
    Load MLGym environment for prime-rl training.

    Args:
        task_configs: List of task config YAML filenames (e.g., ["titanic.yaml", "prisonersDilemma.yaml"])
        max_turns: Maximum turns per episode
        num_examples_per_task: Number of training examples per task
        env_gpu: GPU ID for MLGym container
        container_type: "docker" or "podman"
        image_name: Docker image for MLGym
        training_timeout: Timeout for training commands
        save_trajectories: Whether to save trajectories
        trajectory_dir: Directory for trajectories
        seed: Random seed

    Returns:
        vf.Environment configured for MLGym tasks
    """
    import traceback as tb

    _debug_log_path = "/tmp/air_mlgym_import.log"

    try:
        with open(_debug_log_path, "a") as f:
            f.write(f"\nload_environment called (pid={os.getpid()})\n")
            f.write(f"  task_configs={task_configs}\n")
            f.write(f"  num_examples_per_task={num_examples_per_task}\n")
            f.flush()

        if task_configs is None:
            task_configs = ["titanic.yaml"]

        # Build dataset
        with open(_debug_log_path, "a") as f:
            f.write("Building dataset...\n")
            f.flush()

        dataset_builder = get_dataset_builder(
            task_configs=task_configs,
            num_examples_per_task=num_examples_per_task,
            seed=seed,
        )
        dataset = dataset_builder()

        with open(_debug_log_path, "a") as f:
            f.write(f"Dataset built: {len(dataset)} examples\n")
            f.write(f"Dataset columns: {dataset.column_names}\n")
            f.flush()

        logger.info(f"Loaded {len(dataset)} examples from {len(task_configs)} tasks")

    except Exception as e:
        with open(_debug_log_path, "a") as f:
            f.write(f"\nload_environment FAILED: {e}\n")
            f.write(tb.format_exc())
            f.flush()
        raise

    # Create rubric with delta reward
    # Using delta reward as the main signal for credit assignment
    rubric = vf.Rubric(
        funcs=[compute_delta_reward],
        weights=[1.0],
    )

    return MLGymEnvironment(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        env_gpu=env_gpu,
        container_type=container_type,
        image_name=image_name,
        training_timeout=training_timeout,
        save_trajectories=save_trajectories,
        trajectory_dir=trajectory_dir,
    )
