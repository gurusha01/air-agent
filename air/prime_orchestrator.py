"""Prime-RL orchestrator integration for MLGym environments.

This module provides the main entry point for running RL training on MLGym
tasks using prime-rl's infrastructure.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from air.mlgym_env import MLGymEnvironment, RolloutInput, load_environment

if TYPE_CHECKING:
    pass


class MLGymOrchestrator:
    """Orchestrator for running RL training on MLGym environments.

    This class coordinates rollout collection from MLGym environments and
    prepares training batches for the prime-rl trainer.
    """

    def __init__(
        self,
        task: str,
        task_config_path: str | Path,
        agent_config_path: str | Path,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        batch_size: int = 16,
        rollouts_per_example: int = 4,
        max_steps: int = 50,
        seq_len: int = 4096,
        output_dir: str | Path = "outputs",
        sampling_temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        """Initialize the MLGym orchestrator.

        Args:
            task: MLGym task name
            task_config_path: Path to task configuration
            agent_config_path: Path to agent configuration
            model_name: Model name for inference
            base_url: vLLM server base URL
            api_key: API key for vLLM server
            batch_size: Number of samples per training batch
            rollouts_per_example: Number of rollouts per problem
            max_steps: Maximum steps per episode
            seq_len: Maximum sequence length
            output_dir: Directory for outputs
            sampling_temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
        """
        self.task = task
        self.task_config_path = Path(task_config_path)
        self.agent_config_path = Path(agent_config_path)
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.rollouts_per_example = rollouts_per_example
        self.max_steps = max_steps
        self.seq_len = seq_len
        self.output_dir = Path(output_dir)
        self.sampling_temperature = sampling_temperature
        self.max_tokens = max_tokens

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "rollouts").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Initialize environment
        self.env = load_environment(
            task=task,
            task_config_path=str(task_config_path),
            agent_config_path=str(agent_config_path),
            max_steps=max_steps,
        )
        self.env.set_max_seq_len(seq_len)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Initialize client
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # Sampling args
        self.sampling_args = {
            "temperature": sampling_temperature,
            "max_tokens": max_tokens,
        }

        # Progress tracking
        self.step = 0
        self.total_tokens = 0
        self.total_samples = 0

    async def generate_batch(self, step: int | None = None) -> list[dict]:
        """Generate a batch of rollouts.

        Args:
            step: Current training step

        Returns:
            List of rollout results
        """
        if step is not None:
            self.step = step

        logger.info(f"Generating batch for step {self.step}")

        # Get examples from dataset
        dataset = self.env.get_dataset()
        num_problems = self.batch_size // self.rollouts_per_example

        # Sample problems
        problem_indices = random.sample(range(len(dataset)), min(num_problems, len(dataset)))

        # Create rollout inputs
        all_inputs = []
        for idx in problem_indices:
            example = dataset[idx]
            for _ in range(self.rollouts_per_example):
                all_inputs.append(
                    RolloutInput(
                        prompt=example["prompt"],
                        task=example["task"],
                        example_id=example["example_id"],
                    )
                )

        # Run rollouts
        logger.info(f"Running {len(all_inputs)} rollouts...")
        start_time = time.perf_counter()

        states = await self.env.run_group(
            group_inputs=all_inputs,
            client=self.client,
            model=self.model_name,
            gen_sampling_args=self.sampling_args,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(f"Completed {len(states)} rollouts in {elapsed:.2f}s")

        # Convert states to result dicts
        results = []
        for state in states:
            result = {
                "example_id": state.example_id,
                "task": state.task,
                "prompt": state.prompt,
                "completion": state.completion,
                "reward": state.reward,
                "is_truncated": state.is_truncated,
                "error": type(state.error).__name__ if state.error else None,
                "timing": state.timing,
                "metrics": state.metrics,
                "trajectory": [
                    {
                        "prompt": step.prompt,
                        "completion": step.completion,
                        "tokens": step.tokens,
                    }
                    for step in state.trajectory
                ],
            }
            results.append(result)

        # Update progress
        self.total_samples += len(results)

        return results

    def compute_advantages(self, rewards: list[float], group_size: int = 4) -> list[float]:
        """Compute GRPO-style advantages within groups.

        Args:
            rewards: List of rewards
            group_size: Size of each group for normalization

        Returns:
            List of advantages
        """
        advantages = []
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i : i + group_size]
            mean_reward = sum(group_rewards) / len(group_rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            std_reward = max(std_reward, 1e-8)  # Prevent division by zero

            for r in group_rewards:
                advantages.append((r - mean_reward) / std_reward)

        return advantages

    def prepare_training_batch(self, rollouts: list[dict]) -> dict:
        """Prepare a training batch from rollouts.

        This formats the rollouts for prime-rl's trainer.

        Args:
            rollouts: List of rollout results

        Returns:
            Training batch dictionary
        """
        # Compute advantages
        rewards = [r["reward"] for r in rollouts]
        advantages = self.compute_advantages(rewards, self.rollouts_per_example)

        # Tokenize and prepare sequences
        examples = []
        for rollout, advantage in zip(rollouts, advantages):
            # Build full sequence from trajectory
            input_ids = []
            loss_mask = []

            for step in rollout["trajectory"]:
                # Tokenize prompt (no gradient)
                prompt_text = self._format_messages(step["prompt"])
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                input_ids.extend(prompt_ids)
                loss_mask.extend([0] * len(prompt_ids))

                # Tokenize completion (gradient)
                completion_ids = self.tokenizer.encode(step["completion"], add_special_tokens=False)
                input_ids.extend(completion_ids)
                loss_mask.extend([1] * len(completion_ids))

            # Truncate to max length
            if len(input_ids) > self.seq_len:
                input_ids = input_ids[: self.seq_len]
                loss_mask = loss_mask[: self.seq_len]

            examples.append({
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "advantage": advantage,
                "reward": rollout["reward"],
            })

        return {
            "examples": examples,
            "temperature": self.sampling_temperature,
            "step": self.step,
        }

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages to string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    def save_rollouts(self, rollouts: list[dict], step: int) -> Path:
        """Save rollouts to disk.

        Args:
            rollouts: List of rollout results
            step: Training step number

        Returns:
            Path to saved file
        """
        output_file = self.output_dir / "rollouts" / f"step_{step:06d}.json"

        import json

        with open(output_file, "w") as f:
            json.dump(rollouts, f, indent=2, default=str)

        logger.info(f"Saved rollouts to {output_file}")
        return output_file

    def get_metrics(self, rollouts: list[dict]) -> dict:
        """Compute metrics from rollouts.

        Args:
            rollouts: List of rollout results

        Returns:
            Dictionary of metrics
        """
        df = pd.DataFrame([
            {
                "example_id": r["example_id"],
                "task": r["task"],
                "reward": r["reward"],
                "is_truncated": r["is_truncated"],
                "error": r["error"],
                "num_steps": r["metrics"].get("num_steps", 0),
                "generation_ms": r["timing"].get("generation_ms", 0),
            }
            for r in rollouts
        ])

        # Compute solve rates
        solve_all = (
            df
            .groupby("example_id")
            .apply(lambda x: x.reward.sum() == self.rollouts_per_example, include_groups=False)
            .mean()
        )
        solve_none = df.groupby("example_id").apply(lambda x: x.reward.sum() == 0, include_groups=False).mean()

        return {
            "reward/mean": df.reward.mean(),
            "reward/std": df.reward.std(),
            "num_steps/mean": df.num_steps.mean(),
            "generation_ms/mean": df.generation_ms.mean(),
            "batch/solve_all": solve_all,
            "batch/solve_none": solve_none,
            "batch/effective": 1 - solve_all - solve_none,
            "error/rate": (~df.error.isna()).mean(),
        }


async def run_orchestrator(
    task: str = "battleOfSexes",
    task_config_path: str | None = None,
    agent_config_path: str | None = None,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    base_url: str = "http://localhost:8000/v1",
    num_steps: int = 10,
    batch_size: int = 16,
    rollouts_per_example: int = 4,
    output_dir: str = "outputs",
    **kwargs,
) -> None:
    """Run the MLGym orchestrator.

    Args:
        task: MLGym task name
        task_config_path: Path to task config
        agent_config_path: Path to agent config
        model_name: Model name for inference
        base_url: vLLM server URL
        num_steps: Number of training steps
        batch_size: Batch size
        rollouts_per_example: Rollouts per example
        output_dir: Output directory
        **kwargs: Additional arguments
    """
    # Set up paths
    mlgym_config_root = Path(os.environ.get("MLGYM_CONFIG_ROOT", "./../MLGym/configs")).resolve()

    if task_config_path is None:
        task_config_path = (mlgym_config_root / "tasks" / f"{task}.yaml").as_posix()
    if agent_config_path is None:
        agent_config_path = (mlgym_config_root / "agents" / "default.yaml").as_posix()

    logger.info(f"Starting MLGym orchestrator for task: {task}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")

    orchestrator = MLGymOrchestrator(
        task=task,
        task_config_path=task_config_path,
        agent_config_path=agent_config_path,
        model_name=model_name,
        base_url=base_url,
        batch_size=batch_size,
        rollouts_per_example=rollouts_per_example,
        output_dir=output_dir,
        **kwargs,
    )

    for step in range(num_steps):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step {step + 1}/{num_steps}")
        logger.info(f"{'=' * 60}")

        # Generate rollouts
        rollouts = await orchestrator.generate_batch(step=step)

        # Compute metrics
        metrics = orchestrator.get_metrics(rollouts)
        logger.info(f"Metrics: {metrics}")

        # Save rollouts
        orchestrator.save_rollouts(rollouts, step)

        # Prepare training batch
        batch = orchestrator.prepare_training_batch(rollouts)
        logger.info(f"Prepared training batch with {len(batch['examples'])} examples")

    logger.success("Orchestrator finished!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MLGym Prime-RL Orchestrator")
    parser.add_argument("--task", default="battleOfSexes", help="MLGym task name")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")

    args = parser.parse_args()

    asyncio.run(
        run_orchestrator(
            task=args.task,
            model_name=args.model,
            base_url=args.base_url,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        ),
    )


if __name__ == "__main__":
    main()
