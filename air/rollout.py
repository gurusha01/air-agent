"""Rollout collection for MLGym with veRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import yaml
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Delimiters for parsing
OBS_START = "\n\nOBSERVATION:\n"
OBS_END = "\n\n"
ACTION_MARKER = "ACTION:"


def load_agent_config(config_path: Path) -> dict:
    """Load agent config YAML for prompt templates."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class MLGymDataset(IterableDataset):
    """Dataset that generates episodes from MLGym environments."""

    def __init__(self, task: str, agent_config: dict, tokenizer, num_episodes: int = 16):
        self.task = task
        self.agent_config = agent_config
        self.tokenizer = tokenizer
        self.num_episodes = num_episodes
        self.episode_count = 0

    def __iter__(self):
        """Generate episodes."""
        while self.episode_count < self.num_episodes:
            # Create environment
            env = gym.make(f"mlgym/{self.task}", devices=["cuda:0"]).unwrapped

            # Collect episode
            episode_data = collect_episode_as_sequence(
                env=env,
                agent_config=self.agent_config,
                tokenizer=self.tokenizer,
                max_steps=50,
            )

            env.close()
            self.episode_count += 1

            # Yield in veRL format
            yield {
                "prompt": episode_data["prompt_ids"],
                "response": episode_data["response_ids"],
                "response_mask": episode_data["response_mask"],
                "reward": episode_data["reward"],
            }

    def __len__(self):
        return self.num_episodes


def format_initial_prompt(agent_config: dict, task_desc: str, initial_obs: str) -> str:
    """Format the initial prompt for the episode."""
    system = agent_config.get("system_template", "You are an ML research agent.")
    task_template = agent_config.get("task_template", "Task: {task_description}")

    prompt = f"{system}\n\n{task_template.format(task_description=task_desc)}\n\n"
    prompt += f"{OBS_START}{initial_obs}{OBS_END}"
    prompt += "Think step-by-step. Respond with:\nTHOUGHT: <reasoning>\nACTION: <command>\n\n"

    return prompt


def parse_action(text: str) -> str:
    """Extract action command from LLM output."""
    if ACTION_MARKER in text:
        action = text.split(ACTION_MARKER)[1].split("\n")[0].strip()
        return action
    # Fallback: return full text
    return text.strip()


def collect_episode_as_sequence(
    env,
    agent_config: dict,
    tokenizer: PreTrainedTokenizer,
    max_steps: int = 50,
    max_response_tokens: int = 4096,
) -> dict:
    """Collect one episode as a token sequence with masks using veRL's rollout worker.

    Note: veRL will handle generation via its rollout workers.
    This function just builds the prompt and collects environment interactions.

    Returns:
        dict with keys: prompt_ids, response_ids, response_mask, reward, num_steps
    """
    # Reset environment
    obs_dict, _info = env.reset()
    initial_obs = obs_dict.get("observation", "")

    # Build initial prompt
    task_desc = env.task_args.description if hasattr(env, "task_args") else "Complete the task."
    initial_prompt = format_initial_prompt(agent_config, task_desc, initial_obs)

    # Tokenize prompt
    prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)

    # For now, return simple format
    # veRL's PPO will handle the actual rollout with its workers
    # This is a simplified version for initial integration

    return {
        "prompt_ids": prompt_ids,
        "response_ids": [],  # Will be filled by veRL's rollout
        "response_mask": [],
        "reward": 0.0,
        "num_steps": 0,
    }


def get_trajectory_reward_from_env(env) -> float:
    """Get reward from MLGym results.json."""
    try:
        # Trigger evaluation if not done
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



