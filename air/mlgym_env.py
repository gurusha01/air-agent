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

                system_prompt = """You are an ML research agent. Output ONLY ONE command per response. No explanations.

To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > train_and_predict.py
import pandas as pd
# your code here
ENDOFFILE

COMMANDS:
- cat << 'ENDOFFILE' > filename.py ... ENDOFFILE - Write a file
- python <script.py> - Run Python script
- validate - Check your solution score
- ls, cat, head - View files

CRITICAL RULES:
1. ONE command per response
2. Use cat << 'ENDOFFILE' > file to write files
3. After writing, run 'python train_and_predict.py'
4. Then run 'validate' to check score

WORKSPACE:
- data/train.csv, data/test.csv - Input data
- Output: submission.csv with PassengerId and Survived columns

WORKFLOW:
1. cat << 'ENDOFFILE' > train_and_predict.py
<complete python script>
ENDOFFILE
2. python train_and_predict.py
3. validate"""

                user_prompt = f"""Task: {task_yaml.get('name', task_name)}

{task_description}

Baseline: {baseline_str}

Output your first command (start with: ls data/):"""

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
        logger.info(f"[DEBUG] MLGymEnvironment initialized with self.max_turns={self.max_turns}")
        self.env_gpu = env_gpu
        self.container_type = container_type
        self.image_name = image_name
        self.training_timeout = training_timeout
        self.save_trajectories = save_trajectories
        self.trajectory_dir = Path(trajectory_dir)

        # Container management
        self._envs: dict[str, MLGymEnv] = {}
        self._metrics: dict[str, TaskMetrics] = {}

        # Policy version tracking (based on weight broadcasts)
        self._last_known_step = 0
        self._broadcasts_dir = Path("outputs/run_default/broadcasts")

        # Create trajectory directory
        if save_trajectories:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_policy_step(self) -> int:
        """
        Get the current policy version by checking weight broadcast directories.

        The trainer broadcasts weights to outputs/run_default/broadcasts/step_N/
        after each training step. The highest N is the current policy version.

        Returns:
            Current policy step (0 = initial policy, 1 = after step 0 update, etc.)
        """
        try:
            if not self._broadcasts_dir.exists():
                return 0

            step_dirs = [d for d in self._broadcasts_dir.iterdir()
                        if d.is_dir() and d.name.startswith("step_")]

            if not step_dirs:
                return 0

            # Extract step numbers and find max
            steps = []
            for d in step_dirs:
                try:
                    step_num = int(d.name.split("_")[1])
                    steps.append(step_num)
                except (IndexError, ValueError):
                    continue

            if steps:
                current_step = max(steps)
                if current_step != self._last_known_step:
                    logger.info(f"[Policy] Detected new policy version: π_{current_step}")
                    self._last_known_step = current_step
                return current_step

            return 0
        except Exception as e:
            logger.debug(f"Error getting policy step: {e}")
            return self._last_known_step

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
                # max_steps needs to account for parallel rollouts sharing the container
                # With rollouts_per_example=8 and branching, the same container may receive
                # many more step() calls than max_turns. Use generous buffer.
                max_steps=self.max_turns * 20,
                task_config_path=task_config,
                container_type=self.container_type,
                verbose=False,
            )
            env = MLGymEnv(
                args=env_args,
                devices=[self.env_gpu],
            )
            env.reset()

            # Load MLGym's custom commands (open, edit, validate, etc.)
            self._load_commands(env)

            return env
        finally:
            os.chdir(original_cwd)

    def _load_commands(self, env: MLGymEnv) -> None:
        """Load MLGym's custom shell commands into the container."""
        # Set required environment variables (from MLGym's default agent config)
        env_vars = {
            "WINDOW": "100",       # Lines to show in open command
            "OVERLAP": "2",
            "CURRENT_LINE": "0",
            "CURRENT_FILE": "",
            "SEARCH_RESULTS": "()",
            "SEARCH_FILES": "()",
            "SEARCH_INDEX": "0",
        }
        for var, value in env_vars.items():
            env.communicate(f"export {var}={value}")

        # Command files from MLGym's default agent config
        command_files_paths = [
            "tools/defaults.sh",      # open, goto, scroll, edit, create
            "tools/search.sh",        # search, find_file
            "tools/edit_linting.sh",  # edit with linting
            "tools/validate.sh",      # validate - evaluate solution
            "tools/submit.sh",        # submit - submit final solution
        ]

        command_files = []
        for file_path in command_files_paths:
            full_path = MLGYM_PATH / file_path
            if not full_path.exists():
                logger.warning(f"Command file not found: {full_path}")
                continue

            contents = full_path.read_text()
            name = full_path.name

            # Shell scripts are sourced
            datum = {
                "name": name,
                "contents": contents,
                "type": "source_file",
            }
            command_files.append(datum)

        if command_files:
            env.add_commands(command_files)
            logger.info(f"Loaded {len(command_files)} command files into container")

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
                    # Reload commands after reset (reset clears shell functions)
                    self._load_commands(env)
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

    def _extract_command(self, raw_output: str) -> tuple[str | None, bool]:
        """
        Extract the FIRST complete command from model output.

        Returns:
            Tuple of (command, is_multi_command)
            - Returns (first_command, False) - always extracts first command
            - is_multi_command is True if trailing commands were ignored (for logging)

        Strategy: Extract and execute the first complete command, ignore the rest.
        This allows edit commands to succeed even if model adds trailing commands.

        Handles:
        - Qwen3 thinking tags: <think>...</think>
        - Code blocks: ```command```
        - Edit commands (multi-line with end_of_edit)
        """
        import re

        text = raw_output.strip()

        # Remove Qwen3 thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove code block markers
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = text.replace('```', '')

        text = text.strip()
        lines = text.split('\n')

        # Command patterns
        simple_command_pattern = r'^(open|python|validate|submit|exit_forfeit|skip|ls|head|tail|cd|create|goto|scroll|search|find_file)\b'
        edit_pattern = r'^edit\s+\d+:\d+'
        heredoc_pattern = r'^cat\s+<<\s*[\'"]?(\w+)[\'"]?\s*>\s*\S+'  # cat << 'EOF' > file

        first_command = None
        first_command_end_line = 0
        has_trailing_commands = False

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for heredoc command (cat << 'MARKER' > file)
            heredoc_match = re.match(heredoc_pattern, line_stripped, re.IGNORECASE)
            if heredoc_match:
                if first_command is None:
                    marker = heredoc_match.group(1)  # e.g., 'ENDOFFILE'
                    heredoc_lines = [line_stripped]
                    for j in range(i + 1, len(lines)):
                        heredoc_lines.append(lines[j])
                        if lines[j].strip() == marker:
                            first_command = '\n'.join(heredoc_lines)
                            first_command_end_line = j
                            break
                    else:
                        # No end marker found, include everything
                        first_command = '\n'.join(heredoc_lines)
                        first_command_end_line = len(lines) - 1
                    break
                continue

            # Check for edit command
            if re.match(edit_pattern, line_stripped, re.IGNORECASE):
                if first_command is None:
                    # Find end_of_edit
                    edit_lines = [line_stripped]
                    for j in range(i + 1, len(lines)):
                        line_content = lines[j]
                        # Skip markdown code block markers (```python, ```, etc.)
                        if line_content.strip().startswith('```'):
                            continue
                        edit_lines.append(line_content)
                        if 'end_of_edit' in line_content.lower():
                            first_command = '\n'.join(edit_lines)
                            first_command_end_line = j
                            break
                    else:
                        # No end_of_edit found
                        first_command = '\n'.join(edit_lines)
                        first_command_end_line = len(lines) - 1
                    break  # Found first command, stop looking
                continue

            # Check for simple commands (note: 'cat' alone is not here, it's handled by heredoc)
            if re.match(simple_command_pattern, line_stripped, re.IGNORECASE):
                if first_command is None:
                    first_command = line_stripped
                    first_command_end_line = i
                    break  # Found first command, stop looking

        # Check for trailing commands after the first one (for logging only)
        if first_command is not None:
            for i in range(first_command_end_line + 1, len(lines)):
                line_stripped = lines[i].strip()
                if not line_stripped:
                    continue
                if re.match(simple_command_pattern, line_stripped, re.IGNORECASE):
                    has_trailing_commands = True
                    logger.warning(f"[TRAILING-CMD] Ignoring trailing command: {line_stripped[:50]}")
                    break
                elif re.match(edit_pattern, line_stripped, re.IGNORECASE):
                    has_trailing_commands = True
                    logger.warning(f"[TRAILING-CMD] Ignoring trailing edit command")
                    break

        # Return first command
        if first_command:
            return first_command, has_trailing_commands

        # No known command found, return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip(), False

        return text.strip(), False

    @vf.stop
    async def check_done(self, state: vf.State) -> bool:
        """Check if episode should end."""
        traj_len = len(state.get("trajectory", []))

        if not state.get("trajectory"):
            logger.debug(f"[check_done] No trajectory, returning False")
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
            logger.info(f"[check_done] Submit detected, returning True (traj_len={traj_len})")
            self._save_trajectory(state, {"exit_reason": "submit"})
            return True

        # Check for exit conditions
        exit_keywords = ["exit_forfeit", "exit_context", "exit_cost", "exit_error"]
        if any(kw in last_response.lower() for kw in exit_keywords):
            logger.info(f"[check_done] Exit keyword detected, returning True (traj_len={traj_len})")
            self._save_trajectory(state, {"exit_reason": "exit_keyword"})
            return True

        # Check max turns
        done = traj_len >= self.max_turns
        if done:
            logger.info(f"[check_done] max_turns reached, traj_len={traj_len}, max_turns={self.max_turns}")
            self._save_trajectory(state, {"exit_reason": "max_turns"})
        elif traj_len % 5 == 0:  # Log every 5 turns
            logger.info(f"[check_done] traj_len={traj_len}, max_turns={self.max_turns}, done={done}")
        return done

    def _step_with_timeout(self, env: MLGymEnv, action: str, timeout: float = 60.0) -> tuple:
        """
        Execute step() with a timeout to prevent infinite hangs.

        Args:
            env: MLGym environment
            action: Command to execute
            timeout: Timeout in seconds (default 60s)

        Returns:
            Tuple of (observation, reward, done, info)

        Raises:
            TimeoutError: If step() takes longer than timeout
        """
        import concurrent.futures
        import threading

        result = [None]
        exception = [None]

        def run_step():
            try:
                result[0] = env.step(action)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_step)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running after timeout - this is a hang
            logger.warning(f"[TIMEOUT] step() timed out after {timeout}s for action: {action[:50]}...")
            # We can't forcefully kill the thread, but we can mark this as an error
            # The container may be in a bad state
            raise TimeoutError(f"step() timed out after {timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

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

        # Record policy step at rollout START (first env_response call for this example)
        # This ensures we track which policy GENERATED the trajectory, not when it ended
        if "_policy_step_at_start" not in state:
            state["_policy_step_at_start"] = self._get_current_policy_step()
            logger.info(f"[DEBUG] Rollout starting with policy π_{state['_policy_step_at_start']}")

        # Get the last assistant message (agent's action)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs:
            logger.info(f"[DEBUG] No assistant messages, returning early")
            return [{"role": "user", "content": "Please provide an action."}]

        raw_action = assistant_msgs[-1]["content"]
        example_id = state.get("example_id", "default")

        # Extract command from model output (handle thinking tags, explanations, etc.)
        action, has_trailing = self._extract_command(raw_action)

        # Log if trailing commands were ignored (but don't reject)
        if has_trailing:
            logger.info(f"[TRAILING-CMD] Extracted first command, ignored trailing for example_id={example_id}")

        # Reject empty/None action
        if not action or not action.strip():
            logger.warning(f"[EMPTY-CMD] Empty action for example_id={example_id}")
            state["last_tool_success"] = False  # Triggers -0.5 reward
            return [{"role": "user", "content": "Error: No command detected. Please output a valid command."}]

        logger.info(f"[DEBUG] Processing action for example_id={example_id}, action={action[:100]}...")

        max_retries = 2
        task_config = state.get("info", {}).get("task_config", "tasks/titanic.yaml")

        for attempt in range(max_retries):
            try:
                # Get or create environment
                logger.info(f"[DEBUG] Getting or creating env (attempt {attempt + 1})...")
                env = self._get_or_create_env(state)
                logger.info(f"[DEBUG] Got env, calling step()...")

                # Execute action in MLGym with timeout to prevent hangs
                observation, reward, done, info = self._step_with_timeout(env, action, timeout=60.0)
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

                # Track tool call success for reward computation
                # A successful tool call: no error in observation
                is_error = observation and ("error" in observation.lower()[:100] or
                                           "traceback" in observation.lower()[:200] or
                                           "command not found" in observation.lower())
                state["last_tool_success"] = not is_error

                # Track if this was a validation call and episode is ending
                is_validate = "validate" in action.lower()
                state["last_action_was_validate"] = is_validate
                state["episode_done"] = done

                # Handle episode end
                if done:
                    self._save_trajectory(state, info)
                    self._cleanup_env(example_id)

                return [{"role": "user", "content": observation or "Action executed."}]

            except TimeoutError as e:
                logger.warning(f"Step timed out, forcing container recreation (attempt {attempt + 1})")
                # Container is likely in a bad state, force recreation
                if task_config in self._envs:
                    try:
                        self._envs[task_config].close()
                    except Exception:
                        pass
                    del self._envs[task_config]
                if attempt < max_retries - 1:
                    continue
                else:
                    return [{"role": "user", "content": f"Error: Command timed out after 60s"}]
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

        # Prevent saving the same trajectory twice
        if state.get("_trajectory_saved"):
            logger.debug(f"[_save_trajectory] Already saved, skipping")
            return
        state["_trajectory_saved"] = True

        example_id = state.get("example_id", "default")
        task_name = state.get("info", {}).get("task_name", "unknown")
        metrics = self._metrics.get(example_id, TaskMetrics())

        # Get policy version from rollout START (not current step)
        # This ensures we track which policy GENERATED the trajectory
        policy_step = state.get("_policy_step_at_start", self._get_current_policy_step())

        # Compute improvement safely (handle None values)
        improvement = None
        if metrics.current_score is not None and metrics.baseline_score is not None:
            improvement = metrics.current_score - metrics.baseline_score

        trajectory_data = {
            "task": task_name,
            "example_id": example_id,
            "policy_step": policy_step,  # Which policy (π_N) generated this trajectory
            "trajectory": state.get("trajectory", []),
            "metrics": {
                "baseline_score": metrics.baseline_score,
                "final_score": metrics.current_score,
                "improvement": improvement,
                "validation_history": list(zip(metrics.validation_steps, metrics.validation_scores)),
            },
            "info": info,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }

        # Save to file with policy step in filename: {task}_pi{step}_{timestamp}.json
        filename = f"{task_name}_pi{policy_step}_{trajectory_data['timestamp']}.json"
        filepath = self.trajectory_dir / filename
        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2, default=str)

        logger.info(f"Saved trajectory to {filepath} (policy=π_{policy_step})")


def compute_delta_reward(completion: list[dict], state: vf.State, **kwargs) -> float:
    """
    Compute reward based on tool call success and validation improvement.

    Reward scheme:
    - +0.5 for correct tool call (no error in observation)
    - -0.5 for incorrect tool call (error/exception in observation)
    - +10 * improvement at final validate (episode end)
    """
    reward = 0.0

    # Tool call success/failure reward
    tool_success = state.get("last_tool_success", True)
    if tool_success:
        reward += 0.5
    else:
        reward -= 0.5

    # Final validation improvement bonus (only at episode end)
    episode_done = state.get("episode_done", False)
    if episode_done:
        final_score = state.get("last_validation_score", 0.0)
        baseline_score = state.get("baseline_score", 0.0)
        if final_score is not None and baseline_score is not None:
            improvement = final_score - baseline_score
            reward += 10.0 * improvement

    return reward


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
        logger.info(f"[DEBUG] max_turns={max_turns} (passed to MLGymEnvironment)")

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
