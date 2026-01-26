"""MLGym environment wrapper compatible with prime-rl's verifiers interface.

This module provides a verifiers-compatible Environment class that wraps MLGym tasks,
allowing them to be used with prime-rl's training infrastructure.
"""

from __future__ import annotations, print_function

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import yaml
from datasets import Dataset

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass
class RolloutInput:
    """Input for a single rollout."""

    prompt: list[dict[str, str]]  # Chat messages format
    task: str
    example_id: int
    extra: dict = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    """A single step in the trajectory."""

    prompt: list[dict[str, str]]
    completion: str
    tokens: dict | None = None


@dataclass
class State:
    """State of a rollout, compatible with prime-rl's expected format."""

    example_id: int
    task: str
    prompt: list[dict[str, str]]
    completion: str
    trajectory: list[TrajectoryStep]
    reward: float
    is_truncated: bool = False
    error: Exception | None = None
    timing: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for compatibility."""
        return getattr(self, key, default)


class MLGymEnvironment:
    """Verifiers-compatible environment wrapper for MLGym tasks.

    This class adapts MLGym's gymnasium-based environments to work with
    prime-rl's training infrastructure.
    """

    def __init__(
        self,
        task: str,
        task_config_path: str | Path,
        agent_config_path: str | Path,
        container_type: str = "docker",
        max_steps: int = 50,
        seed: int = 42,
        verbose: bool = False,
        devices: list[str] | None = None,
    ):
        """Initialize the MLGym environment wrapper.

        Args:
            task: Name of the MLGym task (e.g., "battleOfSexes")
            task_config_path: Path to the task configuration YAML
            agent_config_path: Path to the agent configuration YAML
            container_type: Container type for MLGym ("docker" or "local")
            max_steps: Maximum number of steps per episode
            seed: Random seed for reproducibility
            verbose: Whether to enable verbose logging
            devices: List of GPU devices to use
        """
        self.task = task
        self.task_config_path = Path(task_config_path)
        self.agent_config_path = Path(agent_config_path)
        self.container_type = container_type
        self.max_steps = max_steps
        self.seed = seed
        self.verbose = verbose
        self.devices = devices or ["cuda:0"]

        # Load agent config for prompt templates
        self.agent_config = self._load_agent_config()

        # Register the task with MLGym
        self._register_task()

        # Settings for prime-rl compatibility
        self._max_seq_len = 8192
        self._interleaved_rollouts = False
        self._score_rollouts = True

        # Environment names for multi-env support
        self.env_names = [task]

    def _load_agent_config(self) -> dict:
        """Load the agent configuration YAML."""
        with open(self.agent_config_path) as f:
            return yaml.safe_load(f)

    def _register_task(self) -> None:
        """Register the MLGym task."""
        from mlgym.environment.env import EnvironmentArguments
        from mlgym.environment.registration import register_task

        env_args = EnvironmentArguments(
            task_config_path=self.task_config_path,
            container_type=self.container_type,
            max_steps=self.max_steps,
            seed=self.seed,
            verbose=self.verbose,
        )
        register_task(env_args)

    def set_max_seq_len(self, max_seq_len: int) -> None:
        """Set maximum sequence length for tokenization."""
        self._max_seq_len = max_seq_len

    def set_interleaved_rollouts(self, enabled: bool) -> None:
        """Enable/disable interleaved rollout mode."""
        self._interleaved_rollouts = enabled

    def set_score_rollouts(self, enabled: bool) -> None:
        """Enable/disable scoring of rollouts."""
        self._score_rollouts = enabled

    def get_dataset(self, seed: int | None = None) -> Dataset:
        """Get the training dataset for this environment.

        For MLGym, we create a synthetic dataset of task prompts since
        MLGym environments don't have a predefined dataset of inputs.
        """
        # Create prompts for the task
        examples = []
        for i in range(100):  # Generate 100 example prompts
            prompt = self._create_initial_prompt(example_id=i)
            examples.append({
                "prompt": prompt,
                "task": self.task,
                "example_id": i,
            })

        return Dataset.from_list(examples)

    def get_eval_dataset(self, seed: int | None = None) -> Dataset:
        """Get the evaluation dataset."""
        return self.get_dataset(seed=seed)

    def _create_initial_prompt(self, example_id: int = 0) -> list[dict[str, str]]:
        """Create the initial prompt for an episode."""
        system_template = self.agent_config.get(
            "system_template", "You are an ML research agent. Complete the given task by executing commands."
        )

        task_template = self.agent_config.get(
            "task_template", "Task: {task_description}\n\nYou can execute bash commands to complete this task."
        )

        # Load task description from config
        with open(self.task_config_path) as f:
            task_config = yaml.safe_load(f)
        task_description = task_config.get("description", f"Complete the {self.task} task.")

        # (Current Step: {current_step}, Remaining Steps: {remaining_steps})
        # (Open file: {open_file})
        # (Current directory: {working_dir})

        return [
            {"role": "system", "content": system_template},
            {"role": "user", "content": task_template.replace("{description}", task_description)},
        ]

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        gen_sampling_args: dict,
        gen_sem: asyncio.Semaphore | None = None,
        score_sem: asyncio.Semaphore | None = None,
    ) -> list[State]:
        """Run rollouts for a group of inputs.

        This is the main interface expected by prime-rl's orchestrator.
        """
        tasks = [
            self._run_single_rollout(
                rollout_input=inp,
                client=client,
                model=model,
                sampling_args=gen_sampling_args,
                semaphore=gen_sem,
            )
            for inp in group_inputs
        ]
        return await asyncio.gather(*tasks)

    async def _run_single_rollout(
        self,
        rollout_input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict,
        semaphore: asyncio.Semaphore | None = None,
    ) -> State:
        """Run a single rollout in the MLGym environment."""
        start_time = time.perf_counter()
        generation_ms = 0
        scoring_ms = 0

        trajectory: list[TrajectoryStep] = []
        error: Exception | None = None
        reward = 0.0
        is_truncated = False
        full_completion = ""

        try:
            # Create the gymnasium environment
            env = gym.make(f"mlgym/{self.task}", devices=self.devices).unwrapped

            assert env.container is not None

            # Reset environment
            obs_dict, info = env.reset()
            observation = obs_dict.get("observation", "")

            # Build conversation history
            messages = list(rollout_input.prompt)
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nRespond with THOUGHT: <your reasoning> and ACTION: <command to execute>",
            })

            done = False
            step_count = 0

            while not done and step_count < self.max_steps:
                # Generate action from LLM
                gen_start = time.perf_counter()

                if semaphore:
                    async with semaphore:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            **sampling_args,
                        )
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **sampling_args,
                    )

                generation_ms += (time.perf_counter() - gen_start) * 1000

                # Extract completion
                completion_text = response.choices[0].message.content or ""
                full_completion += completion_text + "\n"

                # Record trajectory step
                trajectory.append(
                    TrajectoryStep(
                        prompt=messages.copy(),
                        completion=completion_text,
                        tokens=self._extract_tokens(response) if hasattr(response, "usage") else None,
                    )
                )

                # Parse action from completion
                action = self._parse_action(completion_text)

                # Execute action in environment
                obs_dict, env_reward, terminated, truncated, info = env.step(action)
                observation = obs_dict.get("observation", "")
                done = terminated or truncated
                is_truncated = truncated

                # Update messages for next turn
                messages.append({"role": "assistant", "content": completion_text})
                if not done:
                    messages.append({
                        "role": "user",
                        "content": f"OBSERVATION:\n{observation}\n\nContinue with THOUGHT and ACTION.",
                    })

                step_count += 1

            # Get final reward
            if self._score_rollouts:
                score_start = time.perf_counter()
                reward = self._get_trajectory_reward(env)
                scoring_ms = (time.perf_counter() - score_start) * 1000

            env.close()

        except Exception as e:
            print(e)
            error = e
            reward = 0.0

        total_time = time.perf_counter() - start_time

        return State(
            example_id=rollout_input.example_id,
            task=rollout_input.task,
            prompt=rollout_input.prompt,
            completion=full_completion,
            trajectory=trajectory,
            reward=reward,
            is_truncated=is_truncated,
            error=error,
            timing={
                "generation_ms": generation_ms,
                "scoring_ms": scoring_ms,
                "total_ms": total_time * 1000,
            },
            metrics={
                "num_steps": len(trajectory),
            },
        )

    def _parse_action(self, text: str) -> str:
        """Extract action command from LLM output."""
        if "ACTION:" in text:
            action = text.split("ACTION:")[1].split("\n")[0].strip()
            return action
        # Fallback: return the entire text
        return text.strip()

    def _extract_tokens(self, response: Any) -> dict | None:
        """Extract token information from OpenAI response."""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return None

    def _get_trajectory_reward(self, env: Any) -> float:
        """Get reward from MLGym evaluation."""
        try:
            # Trigger evaluation if available
            if hasattr(env, "task") and env.task:
                env.task.evaluate()

            # Look for results.json in workspace
            workspace_path = Path(env.task_workspace) if hasattr(env, "task_workspace") else None
            if workspace_path:
                results_file = workspace_path / "results.json"
                if results_file.exists():
                    results = json.loads(results_file.read_text())
                    agent_score = results.get("agent", [])
                    if agent_score and isinstance(agent_score, list):
                        return float(agent_score[0].get("Score", 0.0))
        except Exception as e:
            print(f"Warning: Could not get reward: {e}")

        return 0.0


def load_environment(
    task: str = "battleOfSexes",
    task_config_path: str | None = None,
    agent_config_path: str | None = None,
    **kwargs,
) -> MLGymEnvironment:
    """Load an MLGym environment (verifiers-compatible interface).

    This function provides the standard verifiers `load_environment` interface
    for loading MLGym environments.

    Args:
        task: Name of the MLGym task
        task_config_path: Path to task config (defaults to MLGym config location)
        agent_config_path: Path to agent config (defaults to MLGym config location)
        **kwargs: Additional arguments passed to MLGymEnvironment

    Returns:
        MLGymEnvironment instance
    """
    import os

    # Default paths based on environment variables or common locations
    mlgym_config_root = os.environ.get("MLGYM_CONFIG_ROOT", "/home/ubuntu/MLScientist/MLGym/configs")

    if task_config_path is None:
        task_config_path = f"{mlgym_config_root}/tasks/{task}.yaml"

    if agent_config_path is None:
        agent_config_path = f"{mlgym_config_root}/agents/default.yaml"

    return MLGymEnvironment(
        task=task,
        task_config_path=task_config_path,
        agent_config_path=agent_config_path,
        **kwargs,
    )


# Registry for environment discovery
ENVIRONMENT_REGISTRY = {
    "mlgym": load_environment,
}
