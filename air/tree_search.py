"""
Experiment 2: Tree Search with Verbalized Sampling.

Inference-time tree search over MLGym tasks.
Each node = sequence of actions ending at `validate`.
At each node, branch into N children with diverse strategies
using verbalized sampling.

Usage:
    # Terminal 1: Start vLLM
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 32768

    # Terminal 2: Run tree search
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/air-agent \
        python /home/ubuntu/MLScientist/air-agent/air/tree_search.py \
        --branching-factor 3 --max-depth 2 --task-config configs/tasks/titanic.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

from mlgym.environment.env import EnvironmentArguments, MLGymEnv

MLGYM_PATH = Path(__file__).resolve().parents[2] / "MLGym"


# ---------------------------------------------------------------------------
# Task Profiles — task-specific prompts and config
# ---------------------------------------------------------------------------

@dataclass
class TaskProfile:
    name: str
    system_prompt: str
    primary_metric: str        # key in score dict to optimize
    higher_is_better: bool
    script_name: str           # file the model writes
    submission_file: str | None  # file to delete on snapshot restore (None = no CSV)
    data_head_cmd: str | None    # command to show data to model (None = no data)
    root_task_desc: str        # template with {baseline_score} and optionally {data_head}
    strategy_topic: str        # used in strategy prompts
    branch_write_instruction: str  # what to tell child nodes to write
    use_generic_conda: bool = True  # False for RL tasks that install their own deps
    needs_gpu: bool = False         # True for RL tasks that need GPU for training
    step_timeout: float = 120.0    # seconds; RL tasks need 1800+ for training
    task_type: str = "classification"  # classification, regression, rl, game_theory
    target_column: str = ""            # e.g., "Survived", "SalePrice"
    id_column: str = ""                # e.g., "PassengerId", "Id"


TASK_PROFILES: dict[str, TaskProfile] = {
    "titanic": TaskProfile(
        name="Titanic Survival Prediction",
        primary_metric="accuracy",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -5 /home/agent/workspace/data/train.csv && echo '---' && head -3 /home/agent/workspace/data/test.csv",
        strategy_topic="the Titanic survival prediction task",
        branch_write_instruction="Write a complete new train_and_predict.py, then run 'python train_and_predict.py', then 'validate'.\nOutput your first command (write the file with cat << 'ENDOFFILE' > train_and_predict.py):",
        root_task_desc=(
            "Titanic Survival Prediction.\n"
            "Baseline accuracy: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a model on data/train.csv, predict 'Survived' for data/test.csv, "
            "save as submission.csv with columns: PassengerId,Survived\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML research agent. Output ONLY ONE command per response. No explanations.

To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > train_and_predict.py
import pandas as pd
# your code here
ENDOFFILE

COMMANDS:
- cat << 'ENDOFFILE' > filename.py ... ENDOFFILE - Write a file
- python <script.py> - Run Python script
- validate - Check your solution score (ONLY works after submission.csv exists)
- ls, cat, head - View files

CRITICAL RULES:
1. ONE command per response
2. Use cat << 'ENDOFFILE' > file to write files
3. ALWAYS run 'python train_and_predict.py' BEFORE 'validate'. validate only checks submission.csv — it does NOT run your script
4. Handle NaN values BEFORE passing to sklearn (use fillna/dropna). Handle categorical variables with pd.get_dummies() or manual mapping
5. Try different models and feature engineering to maximize accuracy

AVAILABLE PACKAGES: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, torch, transformers, scipy. Do NOT pip install anything.

WORKSPACE:
- data/train.csv, data/test.csv - Input data
- Output: submission.csv with PassengerId and Survived columns

MANDATORY WORKFLOW (follow this EXACT order):
1. cat << 'ENDOFFILE' > train_and_predict.py
<complete python script that handles NaN, encodes categoricals, trains, predicts, and saves submission.csv>
ENDOFFILE
2. python train_and_predict.py
3. validate""",
        task_type="classification",
        target_column="Survived",
        id_column="PassengerId",
    ),

    "regressionKaggleHousePrice": TaskProfile(
        name="House Price Prediction (Kaggle)",
        primary_metric="r2",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -3 /home/agent/workspace/data/train.csv && echo '---' && head -3 /home/agent/workspace/data/test.csv",
        strategy_topic="the Kaggle House Price prediction task (regression, optimize R2 score)",
        branch_write_instruction="Write a complete new train_and_predict.py, then run 'python train_and_predict.py', then 'validate'.\nOutput your first command (write the file with cat << 'ENDOFFILE' > train_and_predict.py):",
        root_task_desc=(
            "House Price Prediction (Kaggle Ames Housing).\n"
            "Baseline R2: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a regression model on data/train.csv (target: SalePrice), "
            "predict SalePrice for data/test.csv, save as submission.csv with columns: Id,SalePrice\n"
            "Validation data is also available at data/validation.csv.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML research agent. Output ONLY ONE command per response. No explanations.

To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > train_and_predict.py
import pandas as pd
# your code here
ENDOFFILE

COMMANDS:
- cat << 'ENDOFFILE' > filename.py ... ENDOFFILE - Write a file
- python <script.py> - Run Python script
- validate - Check your solution score (ONLY works after submission.csv exists)
- ls, cat, head - View files

CRITICAL RULES:
1. ONE command per response
2. Use cat << 'ENDOFFILE' > file to write files
3. ALWAYS run 'python train_and_predict.py' BEFORE 'validate'. validate only checks submission.csv — it does NOT run your script
4. Handle NaN values BEFORE passing to sklearn (use fillna/dropna). Handle categorical variables with pd.get_dummies() or OneHotEncoder
5. Try different models and feature engineering to maximize R2 score

AVAILABLE PACKAGES: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, torch, transformers, scipy. Do NOT pip install anything.

WORKSPACE:
- data/train.csv, data/validation.csv, data/test.csv - Input data (target column: SalePrice)
- Output: submission.csv with Id and SalePrice columns

MANDATORY WORKFLOW (follow this EXACT order):
1. cat << 'ENDOFFILE' > train_and_predict.py
<complete python script that handles NaN, encodes categoricals, trains regression model, predicts, saves submission.csv>
ENDOFFILE
2. python train_and_predict.py
3. validate""",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
    ),

    "battleOfSexes": TaskProfile(
        name="Battle of Sexes",
        primary_metric="Score",
        higher_is_better=True,
        script_name="strategy.py",
        submission_file=None,  # no CSV — validate imports strategy.py directly
        data_head_cmd=None,
        strategy_topic="the Battle of Sexes game theory task (maximize average payoff as row player)",
        branch_write_instruction="Write a new strategy.py with a different row_strategy(history) function, then 'validate'.\nOutput your first command (write the file with cat << 'ENDOFFILE' > strategy.py):",
        root_task_desc=(
            "Battle of Sexes — Iterated Game Theory.\n"
            "Baseline Score: {baseline_score:.4f}\n\n"
            "You are the ROW player in a 2-player, 2-strategy, 10-round iterated game.\n"
            "Payoffs: Both choose 0 → (2,1). Both choose 1 → (1,2). Different → (0,0).\n"
            "You prefer strategy 0. Your partner prefers strategy 1.\n\n"
            "Your goal: maximize your average payoff across 10,000 Monte Carlo simulations.\n"
            "The opponent (column player) uses a strategy that tends to copy your last move (80% chance).\n\n"
            "Write strategy.py with a row_strategy(history) function. "
            "history is a list of (your_move, their_move) tuples. Return 0 or 1.\n"
            "After writing, call 'validate' to test your strategy.\n\n"
            "Write strategy.py now (use cat << 'ENDOFFILE' > strategy.py):"
        ),
        system_prompt="""You are a game theory research agent. Output ONLY ONE command per response. No explanations.

To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > strategy.py
import random
def row_strategy(history):
    # your strategy here
    return 0
ENDOFFILE

COMMANDS:
- cat << 'ENDOFFILE' > strategy.py ... ENDOFFILE - Write strategy file
- validate - Test your strategy (runs 10,000 Monte Carlo simulations)
- ls, cat, head - View files

CRITICAL RULES:
1. ONE command per response
2. Use cat << 'ENDOFFILE' > strategy.py to write the strategy file
3. DO NOT change the function signature: def row_strategy(history)
4. history is a list of (your_move, their_move) tuples. Return 0 or 1
5. Call 'validate' AFTER writing strategy.py — it imports your function directly
6. You prefer strategy 0 (payoff 2 when both pick 0). Partner prefers 1 (payoff 2 when both pick 1)
7. Opponent tends to copy your last move with ~80% probability

AVAILABLE PACKAGES: random, numpy (import in function if needed). Do NOT pip install anything.

MANDATORY WORKFLOW:
1. cat << 'ENDOFFILE' > strategy.py
<strategy function>
ENDOFFILE
2. validate""",
        task_type="game_theory",
    ),

    "rlMountainCarContinuous": TaskProfile(
        name="Mountain Car Continuous (RL)",
        primary_metric="Reward Mean",
        higher_is_better=True,
        script_name="src/train.py",
        submission_file=None,  # checkpoints, not CSV
        data_head_cmd="cat src/config.yaml",
        strategy_topic="the MountainCarContinuous RL task (improve PPO training to maximize mean reward)",
        branch_write_instruction=(
            "Modify the code and/or config to improve the PPO agent's performance, then run 'python src/train.py', then 'validate'.\n"
            "You can modify any file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py.\n"
            "IMPORTANT: You MUST read the existing code first before writing ANY modifications.\n"
            "IMPORTANT: Do NOT rewrite entire files. Make TARGETED edits using sed -i.\n"
            "Output your first command (cat src/networks.py):"
        ),
        root_task_desc=(
            "MountainCarContinuous-v0 — RL PPO Training.\n"
            "Baseline Reward Mean: {baseline_score:.4f}\n\n"
            "Environment: MountainCarContinuous-v0 (gymnax). Car must reach hilltop (pos >= 0.45).\n"
            "Reward: -0.1*action^2 per step, +100 on goal. Episode: 999 steps.\n\n"
            "Current config:\n{data_head}\n\n"
            "Source files: src/train.py, src/networks.py, src/policy.py, src/helpers.py, src/config.yaml\n"
            "You can modify any of these files.\n\n"
            "IMPORTANT: You MUST read the existing source code BEFORE making any changes.\n"
            "Your first 3 commands MUST be:\n"
            "  1. cat src/networks.py\n"
            "  2. cat src/policy.py\n"
            "  3. cat src/train.py\n"
            "Only AFTER reading all 3 files should you start modifying code.\n\n"
            "Goal: Maximize mean reward. Output your first command (cat src/networks.py):"
        ),
        system_prompt="""You are an RL research agent. Output ONLY ONE command per response. No explanations.

To edit files, use one of these approaches:
- For simple substitutions: sed -i 's/old/new/g' src/filename.py
- To rewrite a file: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  (Read the file first, then rewrite it with your modifications)

COMMANDS:
- sed -i 's/old/new/g' file - Text substitutions
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Rewrite a file with modifications
- python src/train.py - Train the PPO agent
- validate - Evaluate checkpoints (ONLY after training completes)
- cat, ls, head - View files

CRITICAL RULES:
1. ONE command per response
2. ALWAYS run 'python src/train.py' BEFORE 'validate'
3. You can modify ANY file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py
4. The class name 'Model' in networks.py must NOT be changed (evaluation depends on it)
5. You MUST read existing source files BEFORE writing any modifications.
6. rm -rf checkpoints before re-training

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability (tfp). Do NOT pip install anything.

WORKSPACE:
- src/config.yaml - Hyperparameters (train_config with nested keys)
- src/networks.py - Actor-Critic model (class Model, get_model_ready function)
- src/policy.py - PPO training loop, rollout manager, loss functions
- src/train.py - Entry point (loads config, calls train_ppo)
- src/helpers.py - Config loading, pickle save/load

MANDATORY WORKFLOW (follow this EXACT order):
1. cat src/networks.py  (READ existing code first)
2. cat src/policy.py    (READ existing code)
3. cat src/train.py     (READ existing code)
4. Make modifications (sed -i or cat << 'ENDOFFILE')
5. rm -rf checkpoints
6. python src/train.py
7. validate""",
        use_generic_conda=False,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="rl",
    ),

    "rlMetaMaze": TaskProfile(
        name="MetaMaze Navigation (RL)",
        primary_metric="Reward Mean",
        higher_is_better=True,
        script_name="src/train.py",
        submission_file=None,
        data_head_cmd="cat src/config.yaml",
        strategy_topic="the MetaMaze RL task (improve PPO training for grid navigation, maximize mean reward)",
        branch_write_instruction=(
            "Modify the code and/or config to improve the PPO agent's performance, then run 'python src/train.py', then 'validate'.\n"
            "You can modify any file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py.\n"
            "IMPORTANT: You MUST read the existing code first before writing ANY modifications.\n"
            "IMPORTANT: Do NOT rewrite entire files. Make TARGETED edits using sed -i.\n"
            "Output your first command (cat src/networks.py):"
        ),
        root_task_desc=(
            "MetaMaze-misc — RL PPO Training.\n"
            "Baseline Reward Mean: {baseline_score:.4f}\n\n"
            "Environment: MetaMaze-misc (gymnax). Agent navigates grid maze to reach goal.\n"
            "Obs: local receptive field + last action + last reward + timestep.\n"
            "Actions: 4 discrete (up/right/down/left). Reward: +10 on goal. Episode: 200 steps.\n\n"
            "Current config:\n{data_head}\n\n"
            "Source files: src/train.py, src/networks.py, src/policy.py, src/helpers.py, src/config.yaml\n"
            "You can modify any of these files.\n\n"
            "IMPORTANT: You MUST read the existing source code BEFORE making any changes.\n"
            "Your first 3 commands MUST be:\n"
            "  1. cat src/networks.py\n"
            "  2. cat src/policy.py\n"
            "  3. cat src/train.py\n"
            "Only AFTER reading all 3 files should you start modifying code.\n\n"
            "Goal: Maximize mean reward. Output your first command (cat src/networks.py):"
        ),
        system_prompt="""You are an RL research agent. Output ONLY ONE command per response. No explanations.

To edit files, use one of these approaches:
- For simple substitutions: sed -i 's/old/new/g' src/filename.py
- To rewrite a file: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  (Read the file first, then rewrite it with your modifications)

COMMANDS:
- sed -i 's/old/new/g' file - Text substitutions
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Rewrite a file with modifications
- python src/train.py - Train the PPO agent
- validate - Evaluate checkpoints (ONLY after training completes)
- cat, ls, head - View files

CRITICAL RULES:
1. ONE command per response
2. ALWAYS run 'python src/train.py' BEFORE 'validate'
3. You can modify ANY file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py
4. The class name 'Model' in networks.py must NOT be changed (evaluation depends on it)
5. You MUST read existing source files BEFORE writing any modifications.
6. rm -rf checkpoints before re-training

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability (tfp). Do NOT pip install anything.

WORKSPACE:
- src/config.yaml - Hyperparameters (train_config with nested keys)
- src/networks.py - Actor-Critic model (class Model, get_model_ready function)
- src/policy.py - PPO training loop, rollout manager, loss functions
- src/train.py - Entry point (loads config, calls train_ppo)
- src/helpers.py - Config loading, pickle save/load

MANDATORY WORKFLOW (follow this EXACT order):
1. cat src/networks.py  (READ existing code first)
2. cat src/policy.py    (READ existing code)
3. cat src/train.py     (READ existing code)
4. Make modifications (sed -i or cat << 'ENDOFFILE')
5. rm -rf checkpoints
6. python src/train.py
7. validate""",
        use_generic_conda=False,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="rl",
    ),
}


def get_task_profile(task_config: str) -> TaskProfile:
    """Auto-detect task from config path and return its profile."""
    for key in TASK_PROFILES:
        if key.lower() in task_config.lower():
            return TASK_PROFILES[key]
    # Fallback to titanic
    print(f"WARNING: Unknown task config '{task_config}', falling back to titanic profile")
    return TASK_PROFILES["titanic"]


# ---------------------------------------------------------------------------
# Strategy prompt templates (task-agnostic via {task_topic})
# ---------------------------------------------------------------------------

STRATEGY_PROMPT_TAIL = """You are helping plan experiments for {task_topic}.

Current situation:
- Current score: {current_score:.4f}
- Baseline score: {baseline_score:.4f}
- Previous approach: {previous_approach}

<instructions>
Generate {n_strategies} responses to improve the score. Each response must be within a separate <response> tag. Each <response> must include a <text> describing a concrete strategy and a numeric <probability>. Please sample at random from the the distribution, such that the probability of each response is less than 0.20. Each strategy should be FUNDAMENTALLY DIFFERENT from the others (different model families, different feature engineering, different preprocessing).
</instructions>"""

STRATEGY_PROMPT_UNIFORM = """You are helping plan experiments for {task_topic}.

Current situation:
- Current score: {current_score:.4f}
- Baseline score: {baseline_score:.4f}
- Previous approach: {previous_approach}

<instructions>
Generate {n_strategies} different strategies to improve the score. Each response must be within a separate <response> tag. Each <response> must include a <text> describing a concrete strategy and a numeric <probability>. Each strategy should be a reasonable, well-known approach. Assign equal probability to each strategy.
</instructions>"""

STRATEGY_PROMPT_LOCAL = """You are helping plan experiments for {task_topic}.

Current situation:
- Current score: {current_score:.4f}
- Baseline score: {baseline_score:.4f}
- Previous approach: {previous_approach}

<instructions>
The previous approach is showing promise. Generate {n_strategies} VARIATIONS of the same general approach to try to squeeze out more performance. Each response must be within a separate <response> tag. Each <response> must include a <text> describing a concrete variation and a numeric <probability>.

IMPORTANT: Stay within the same methodology as the previous approach. Do NOT propose a completely different approach. Refine and improve what's already working — try different hyperparameters, preprocessing tweaks, or small algorithmic variations.
</instructions>"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    strategy: str
    score: float | None = None
    actions: list[dict] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    error: str | None = None
    snapshot_path: str = ""
    reflection: str = ""  # tree-level reflection injected before this node


# ---------------------------------------------------------------------------
# Command extraction (adapted from mlgym_env.py)
# ---------------------------------------------------------------------------

def extract_command(raw_output: str) -> tuple[str | None, bool]:
    """Extract the first complete command from model output."""
    text = raw_output.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```[a-z]*\n?', '', text)
    text = text.replace('```', '')
    text = text.strip()
    lines = text.split('\n')

    simple_cmd = r'^(open|python|validate|submit|exit_forfeit|skip|ls|head|tail|cd|create|goto|scroll|search|find_file)\b'
    edit_pat = r'^edit\s+\d+:\d+'
    heredoc_pat = r'^cat\s+<<\s*[\'"]?(\w+)[\'"]?\s*>\s*\S+'

    first_command = None
    first_end = 0

    for i, line in enumerate(lines):
        ls = line.strip()
        if not ls:
            continue

        hm = re.match(heredoc_pat, ls, re.IGNORECASE)
        if hm and first_command is None:
            marker = hm.group(1)
            hlines = [ls]
            for j in range(i + 1, len(lines)):
                hlines.append(lines[j])
                if lines[j].strip() == marker:
                    first_command = '\n'.join(hlines)
                    first_end = j
                    break
            else:
                first_command = '\n'.join(hlines)
                first_end = len(lines) - 1
            break

        if re.match(edit_pat, ls, re.IGNORECASE) and first_command is None:
            elines = [ls]
            for j in range(i + 1, len(lines)):
                lc = lines[j]
                if lc.strip().startswith('```'):
                    continue
                elines.append(lc)
                if 'end_of_edit' in lc.lower():
                    first_command = '\n'.join(elines)
                    first_end = j
                    break
            else:
                first_command = '\n'.join(elines)
                first_end = len(lines) - 1
            break

        if re.match(simple_cmd, ls, re.IGNORECASE) and first_command is None:
            first_command = ls
            first_end = i
            break

    has_trailing = False
    if first_command is not None:
        for i in range(first_end + 1, len(lines)):
            ls = lines[i].strip()
            if not ls:
                continue
            if re.match(simple_cmd, ls, re.IGNORECASE) or re.match(edit_pat, ls, re.IGNORECASE):
                has_trailing = True
                break

    if first_command:
        return first_command, has_trailing

    for line in lines:
        if line.strip():
            return line.strip(), False
    return text.strip() if text.strip() else None, False


# ---------------------------------------------------------------------------
# Container Manager
# ---------------------------------------------------------------------------

class ContainerManager:
    """Manages a single MLGym container with workspace snapshots."""

    def __init__(self, task_config: str, env_gpu: str, image_name: str,
                 task_profile: TaskProfile | None = None):
        self.task_config = task_config
        self.env_gpu = env_gpu
        self.image_name = image_name
        self.task_profile = task_profile
        self.env: MLGymEnv | None = None
        self.baseline_score: float = 0.0
        self.baseline_scores_dict: dict = {}

    def create(self):
        original_cwd = os.getcwd()
        try:
            os.chdir(MLGYM_PATH)
            env_args = EnvironmentArguments(
                image_name=self.image_name,
                max_steps=99999,
                task_config_path=self.task_config,
                container_type="docker",
                verbose=False,
            )
            self.env = MLGymEnv(args=env_args, devices=[self.env_gpu])
            self.env.reset()
            self._load_commands()

            # Extract baseline scores (try multiple paths)
            scores = None
            if hasattr(self.env, 'task') and hasattr(self.env.task, 'baseline_scores'):
                scores = self.env.task.baseline_scores
            elif hasattr(self.env, 'task_args') and hasattr(self.env.task_args, 'baseline_scores'):
                scores = self.env.task_args.baseline_scores
            elif hasattr(self.env, 'task') and hasattr(self.env.task, 'args') and hasattr(self.env.task.args, 'baseline_scores'):
                scores = self.env.task.args.baseline_scores
            if scores:
                if isinstance(scores, dict):
                    self.baseline_scores_dict = scores
                elif isinstance(scores, list) and scores:
                    self.baseline_scores_dict = scores[0] if isinstance(scores[0], dict) else {"score": scores[0]}
                # Pick primary metric for baseline_score
                if self.task_profile and self.task_profile.primary_metric in self.baseline_scores_dict:
                    self.baseline_score = self.baseline_scores_dict[self.task_profile.primary_metric]
                elif self.baseline_scores_dict:
                    self.baseline_score = list(self.baseline_scores_dict.values())[0]

            # Activate conda env for tabular ML tasks (has sklearn, torch, etc.)
            # RL tasks (use_generic_conda=False) install their own deps via MLGym.
            if self.task_profile and self.task_profile.use_generic_conda:
                print("Activating generic conda env...")
                self.env.communicate(
                    "export PATH=/home/agent/miniconda3/envs/mlgym_generic/bin:$PATH",
                    timeout_duration=10,
                )
                self.env.communicate(
                    "pip install -q xgboost lightgbm catboost > /dev/null 2>&1",
                    timeout_duration=300,
                )
                check = self.env.communicate(
                    "python -c 'import torch, sklearn, xgboost; "
                    "print(f\"torch={torch.__version__}, sklearn={sklearn.__version__}, xgb={xgboost.__version__}\")' 2>&1"
                )
                print(f"  Package check: {check.strip()}")
            else:
                print("RL task — skipping generic conda (task installs own deps)")

            # cd into workspace
            self.env.communicate("cd /home/agent/workspace")
            print(f"Container created. Baseline scores: {self.baseline_scores_dict}")
            print(f"Primary baseline ({self.task_profile.primary_metric if self.task_profile else '?'}): {self.baseline_score:.4f}")
        finally:
            os.chdir(original_cwd)

    def _load_commands(self):
        env = self.env
        for var, val in {"WINDOW": "100", "OVERLAP": "2", "CURRENT_LINE": "0",
                         "CURRENT_FILE": "", "SEARCH_RESULTS": "()",
                         "SEARCH_FILES": "()", "SEARCH_INDEX": "0"}.items():
            env.communicate(f"export {var}={val}")

        for sh in ["tools/defaults.sh", "tools/search.sh", "tools/edit_linting.sh",
                    "tools/validate.sh", "tools/submit.sh"]:
            full = MLGYM_PATH / sh
            if full.exists():
                contents = full.read_text()
                datum = {"name": full.name, "contents": contents, "type": "source_file"}
                env.add_commands([datum])

    def step(self, action: str, timeout: float | None = None) -> tuple[str, dict]:
        """Execute action, return (observation, info)."""
        if timeout is None:
            timeout = self.task_profile.step_timeout if self.task_profile else 120.0
        obs, reward, done, info = self.env.step(action)
        obs = obs or "Action executed."
        # If the container restarted (timeout/crash), shell functions are lost.
        # Reload commands and restore working directory.
        if "RESTARTING PROCESS" in obs:
            self._load_commands()
            self.env.communicate("cd /home/agent/workspace")
        return obs, info

    def communicate(self, cmd: str, timeout: float = 30.0) -> str:
        return self.env.communicate(cmd, timeout_duration=timeout) or ""

    def save_snapshot(self, node_id: str) -> str:
        safe_id = node_id.replace("/", "_").replace(" ", "_")
        snap = f"/tmp/snap_{safe_id}.tar"
        self.communicate(f"cd /home/agent && tar cf {snap} workspace")
        return snap

    def restore_snapshot(self, snap_path: str):
        self.communicate("cd /home/agent && rm -rf workspace")
        self.communicate(f"cd /home/agent && tar xf {snap_path}")
        self.communicate("cd /home/agent/workspace")
        self.env.current_step = 0  # reset step counter

    def close(self):
        if self.env:
            self.env.close()


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float,
                 thinking_budget: int = 0):
        import os
        # Detect Claude models → use ANTHROPIC_API_KEY and Anthropic's
        # OpenAI-compatible endpoint
        if "claude" in model.lower():
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = base_url or "https://api.anthropic.com/v1/"
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "local")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        is_reasoning = any(t in self.model for t in ("o1", "o3", "o4"))
        token_key = "max_completion_tokens" if is_reasoning or "gpt-5" in self.model else "max_tokens"
        max_tokens = 16384 if is_reasoning else 4096
        # When thinking is enabled, budget is in addition to output tokens
        if self.thinking_budget > 0:
            max_tokens = max(max_tokens, self.thinking_budget + 4096)
        kwargs = {
            "model": self.model,
            "messages": messages,
            token_key: max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = temperature or self.temperature

        # Add thinking budget for Claude via extra_body
        if self.thinking_budget > 0:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}
            }
            # Anthropic requires temperature=1 (or unset) when thinking is enabled
            kwargs.pop("temperature", None)

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"LLM failed after 3 attempts: {e}")

    def generate_strategies(self, current_score: float, baseline_score: float,
                            previous_approach: str, n: int,
                            sampling_mode: str = "tail",
                            task_topic: str = "this task",
                            sibling_info: list[str] | None = None,
                            ) -> list[tuple[str, float]]:
        prompt_templates = {
            "tail": STRATEGY_PROMPT_TAIL,
            "uniform": STRATEGY_PROMPT_UNIFORM,
            "local": STRATEGY_PROMPT_LOCAL,
        }
        template = prompt_templates.get(sampling_mode, STRATEGY_PROMPT_TAIL)
        prompt = template.format(
            current_score=current_score,
            baseline_score=baseline_score,
            previous_approach=previous_approach,
            n_strategies=n,
            task_topic=task_topic,
        )
        if sibling_info:
            prompt += (
                "\n\nStrategies already tried from this same parent "
                "(DO NOT repeat or closely resemble any of these):\n"
                + "\n".join(sibling_info)
            )
        resp = self.chat([{"role": "user", "content": prompt}], temperature=1.0)
        return self._parse_strategies(resp)

    @staticmethod
    def _parse_strategies(text: str) -> list[tuple[str, float]]:
        strategies = []
        for m in re.finditer(r'<response>(.*?)</response>', text, re.DOTALL):
            block = m.group(1)
            tm = re.search(r'<text>(.*?)</text>', block, re.DOTALL)
            pm = re.search(r'<probability>(.*?)</probability>', block, re.DOTALL)
            if tm:
                t = tm.group(1).strip()
                p = float(pm.group(1).strip()) if pm else 0.05
                strategies.append((t, p))
        return strategies


# ---------------------------------------------------------------------------
# Tree Search
# ---------------------------------------------------------------------------

class TreeSearch:
    def __init__(self, llm: LLMClient, container: ContainerManager,
                 task_profile: TaskProfile,
                 branching_factor: int = 3, max_depth: int = 2,
                 max_actions: int = 15, output_dir: str = "outputs/tree_search",
                 verbose: bool = False, verbalized_sampling: bool = True,
                 sampling_mode: str = "tail"):
        self.llm = llm
        self.container = container
        self.task = task_profile
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.max_actions = max_actions
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.verbalized_sampling = verbalized_sampling
        self.sampling_mode = sampling_mode
        self.nodes: dict[str, TreeNode] = {}

    def run(self) -> dict:
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "nodes").mkdir(exist_ok=True)

        # Execute root
        print("\n" + "=" * 60)
        print("TREE SEARCH - Root Node")
        print("=" * 60)
        root = self._execute_root()
        self.nodes[root.node_id] = root
        self._save_node(root)

        # BFS expansion
        frontier = [root.node_id]
        for depth in range(1, self.max_depth + 1):
            print(f"\n{'=' * 60}")
            print(f"TREE SEARCH - Depth {depth} ({len(frontier)} nodes to expand)")
            print("=" * 60)
            next_frontier = []
            for nid in frontier:
                node = self.nodes[nid]
                if node.score is None:
                    print(f"  Skipping {nid} (no score)")
                    continue
                children = self._branch(nid, depth)
                next_frontier.extend(children)
            frontier = next_frontier

        # Results
        scored_nodes = [(nid, n.score) for nid, n in self.nodes.items() if n.score is not None]
        if not scored_nodes:
            print("\nWARNING: No nodes achieved a valid score!")
            best_id, best_score = "root", 0.0
        else:
            if self.task.higher_is_better:
                best_id, best_score = max(scored_nodes, key=lambda x: x[1])
            else:
                best_id, best_score = min(scored_nodes, key=lambda x: x[1])
        elapsed = time.time() - start

        result = {
            "task": self.task.name,
            "primary_metric": self.task.primary_metric,
            "higher_is_better": self.task.higher_is_better,
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "elapsed_seconds": round(elapsed, 1),
            "nodes": {nid: {
                "node_id": n.node_id,
                "parent_id": n.parent_id,
                "depth": n.depth,
                "strategy": n.strategy[:100],
                "score": n.score,
                "actions_count": len(n.actions),
                "children": n.children,
                "error": n.error,
            } for nid, n in self.nodes.items()},
        }

        # Save result
        with open(self.output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        # Print tree
        if scored_nodes:
            self._print_tree(best_id)
        else:
            print("\nNo successful nodes to display.")

        return result

    def _execute_root(self) -> TreeNode:
        """Root = baseline. No model execution — just snapshot the clean workspace."""
        data_head = ""
        if self.task.data_head_cmd:
            data_head = self.container.communicate(self.task.data_head_cmd)

        task_desc = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head=data_head,
        )
        # Root conversation = system prompt + task description (no model response yet)
        messages = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        snap = self.container.save_snapshot("root")
        baseline = self.container.baseline_score

        print(f"  [root] Baseline node (score={baseline:.4f}), branching immediately")

        return TreeNode(
            node_id="root", parent_id=None, depth=0,
            strategy="Baseline (no model execution)",
            score=baseline, actions=[],
            conversation_history=messages,
            snapshot_path=snap,
        )

    def _branch(self, parent_id: str, depth: int) -> list[str]:
        parent = self.nodes[parent_id]

        # Summarize previous approach
        is_from_baseline = len(parent.actions) == 0
        if is_from_baseline:
            approach_summary = "No solution yet (baseline only)"
        else:
            approach_summary = parent.strategy
            cmds = [a.get("action", "")[:80] for a in parent.actions[-5:]]
            approach_summary += " | Recent: " + ", ".join(cmds)

        # Generate diverse strategies
        if self.verbalized_sampling:
            mode_label = {"tail": "tail-sampling", "uniform": "uniform-sampling", "local": "local-refinement"}
            print(f"\n  Generating strategies for {parent_id} (score={parent.score:.4f}) [{mode_label.get(self.sampling_mode, self.sampling_mode)}]...")
            strategies = self.llm.generate_strategies(
                current_score=parent.score or 0.0,
                baseline_score=self.container.baseline_score,
                previous_approach=approach_summary,
                n=self.branching_factor + 2,  # generate extra in case some fail to parse
                sampling_mode=self.sampling_mode,
                task_topic=self.task.strategy_topic,
            )

            if not strategies:
                print(f"  WARNING: No strategies parsed, using fallback")
                strategies = [
                    ("Try XGBoost with extensive feature engineering", 0.05),
                    ("Try stacking ensemble with RF + LR + SVM", 0.05),
                    ("Try LightGBM with Bayesian hyperparameter tuning", 0.05),
                ]
        else:
            # No verbalized sampling — use generic prompts, rely on temperature for diversity
            print(f"\n  Branching {parent_id} (score={parent.score:.4f}) without verbalized sampling...")
            strategies = [
                (f"Temperature sample {i}", 0.0)
                for i in range(self.branching_factor)
            ]

        child_ids = []
        for i, (strategy_text, prob) in enumerate(strategies[:self.branching_factor]):
            child_id = f"{parent_id}_{i}"
            print(f"\n  --- Branch {child_id} (p={prob:.2f}) ---")
            print(f"  Strategy: {strategy_text[:100]}")

            # Restore parent workspace and remove old submission so child must generate its own
            self.container.restore_snapshot(parent.snapshot_path)
            if self.task.submission_file:
                self.container.communicate(f"rm -f /home/agent/workspace/{self.task.submission_file}")

            # Fork conversation and inject strategy
            child_msgs = copy.deepcopy(parent.conversation_history)
            write_instr = self.task.branch_write_instruction
            is_from_baseline = len(parent.actions) == 0  # parent is baseline root

            if is_from_baseline:
                # First generation — parent is baseline, children start fresh
                if self.verbalized_sampling and strategy_text:
                    child_msgs.append({
                        "role": "user",
                        "content": (
                            f"Strategy to try: {strategy_text}\n\n"
                            f"{write_instr}"
                        ),
                    })
                # else: child_msgs already has task_desc from root, model will respond directly
            elif self.verbalized_sampling and self.sampling_mode == "local":
                child_msgs.append({
                    "role": "user",
                    "content": (
                        f"Your current score is {parent.score:.4f}. "
                        f"Refine your current approach to improve it.\n"
                        f"Variation to try: {strategy_text}\n\n"
                        f"IMPORTANT: Stay within the same approach as before. "
                        f"Do NOT switch to a completely different algorithm. "
                        f"Just tune or tweak the existing approach.\n\n"
                        f"{write_instr}"
                    ),
                })
            elif self.verbalized_sampling:
                child_msgs.append({
                    "role": "user",
                    "content": (
                        f"Your current score is {parent.score:.4f}. "
                        f"Try a DIFFERENT approach to improve it.\n"
                        f"Strategy: {strategy_text}\n\n"
                        f"{write_instr}"
                    ),
                })
            else:
                child_msgs.append({
                    "role": "user",
                    "content": (
                        f"Your current score is {parent.score:.4f}. "
                        f"Try a DIFFERENT approach to improve it.\n\n"
                        f"{write_instr}"
                    ),
                })

            try:
                score, actions, final_msgs = self._execute_until_validate(child_msgs, child_id)
                snap = self.container.save_snapshot(child_id)
                error = None
            except Exception as e:
                print(f"  ERROR: {e}")
                score, actions, final_msgs = None, [], child_msgs
                snap = ""
                error = str(e)

            child = TreeNode(
                node_id=child_id, parent_id=parent_id, depth=depth,
                strategy=strategy_text, score=score, actions=actions,
                conversation_history=final_msgs, snapshot_path=snap,
                error=error,
            )
            self.nodes[child_id] = child
            child_ids.append(child_id)
            parent.children.append(child_id)
            self._save_node(child)

        return child_ids

    def _execute_until_validate(self, messages: list[dict], node_id: str
                                ) -> tuple[float | None, list[dict], list[dict]]:
        """Execute actions until validate is called. Returns (score, action_log, messages)."""
        action_log = []
        score = None

        for step in range(self.max_actions):
            # Get model response
            try:
                raw = self.llm.chat(messages)
            except Exception as e:
                # Retry once
                time.sleep(2)
                try:
                    raw = self.llm.chat(messages)
                except Exception:
                    raise RuntimeError(f"LLM failed: {e}")

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "No command detected. Output a valid command."})
                action_log.append({"action": raw[:100], "observation": "No command", "step": step})
                continue

            # Block submit, replace with validate
            if action.strip().lower() == "submit":
                action = "validate"

            if self.verbose:
                print(f"    [{node_id}] step {step}: {action[:80]}")

            # Execute
            obs, info = self.container.step(action)

            # Log validate/python output when verbose
            if self.verbose:
                if "validate" in action.strip().lower():
                    print(f"    [{node_id}] validate score={info.get('score')}")
                elif action.strip().startswith("python"):
                    obs_tail = (obs or '')[-200:] if obs and len(obs) > 200 else (obs or '')
                    has_error = "Traceback" in (obs or "") or "Error" in (obs or "")
                    if has_error:
                        print(f"    [{node_id}] python ERROR: {obs_tail[:150]}")

            action_log.append({
                "action": action[:2000],
                "observation": obs[:2000] if obs else "",
                "step": step,
            })

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            # Check for validate result — also parse score from obs text
            score_found = None
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    # Use primary metric if available, else first value
                    metric = self.task.primary_metric
                    score_found = score_data.get(metric, list(score_data.values())[0])
                else:
                    score_found = score_data
            elif obs and "Evaluation Score" in obs:
                # Fallback: parse score from observation text
                import ast
                m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
                if m:
                    try:
                        score_dict = ast.literal_eval(m.group(1))
                        score_found = list(score_dict.values())[0]
                    except Exception:
                        pass

            if score_found is not None:
                score = score_found
                print(f"  [{node_id}] validate → {score:.4f}")
                break
        else:
            # Max actions without validate — force it
            print(f"  [{node_id}] Max actions reached, forcing validate")
            obs, info = self.container.step("validate")
            messages.append({"role": "assistant", "content": "validate"})
            messages.append({"role": "user", "content": obs})
            action_log.append({"action": "validate (forced)", "observation": obs[:500], "step": self.max_actions})
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    metric = self.task.primary_metric
                    score = score_data.get(metric, list(score_data.values())[0])
                else:
                    score = score_data
                print(f"  [{node_id}] forced validate → {score:.4f}")
            elif obs and "Evaluation Score" in obs:
                import ast
                m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
                if m:
                    try:
                        score_dict = ast.literal_eval(m.group(1))
                        score = list(score_dict.values())[0]
                        print(f"  [{node_id}] forced validate (from obs) → {score:.4f}")
                    except Exception:
                        pass

        return score, action_log, messages

    def _save_node(self, node: TreeNode):
        path = self.output_dir / "nodes" / f"{node.node_id}.json"
        data = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "strategy": node.strategy,
            "score": node.score,
            "error": node.error,
            "actions": node.actions,
            "conversation_length": len(node.conversation_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _print_tree(self, best_id: str):
        print(f"\n{'=' * 70}")
        print("TREE SEARCH RESULTS")
        print(f"{'=' * 70}")

        best = self.nodes[best_id]
        print(f"Baseline: {self.container.baseline_score:.4f} | "
              f"Best: {best.score:.4f} (node: {best_id}) | "
              f"Improvement: {best.score - self.container.baseline_score:+.4f}")
        print(f"Nodes explored: {len(self.nodes)}")
        print(f"{'=' * 70}\n")

        def _print_node(nid: str, prefix: str = "", is_last: bool = True):
            n = self.nodes[nid]
            connector = "└── " if is_last else "├── "
            marker = " *** BEST ***" if nid == best_id else ""
            score_str = f"{n.score:.4f}" if n.score is not None else "FAIL"
            strategy_short = n.strategy[:50]
            print(f"{prefix}{connector}{nid} [{score_str}] {strategy_short}{marker}")

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, cid in enumerate(n.children):
                _print_node(cid, child_prefix, i == len(n.children) - 1)

        _print_node("root")

        # Print best path
        path = []
        nid = best_id
        while nid:
            path.append(nid)
            nid = self.nodes[nid].parent_id
        path.reverse()
        print(f"\nBest path: {' -> '.join(path)}")
        for p in path:
            n = self.nodes[p]
            print(f"  {p}: [{n.score:.4f}] {n.strategy[:80]}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tree search with verbalized sampling over MLGym")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--task-config", default="tasks/titanic.yaml")
    parser.add_argument("--output-dir", default="outputs/tree_search")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-verbalized-sampling", action="store_true",
                        help="Disable verbalized sampling; rely on temperature for diversity")
    parser.add_argument("--sampling-mode", default="tail", choices=["tail", "uniform", "local"],
                        help="Verbalized sampling mode: tail (sample unusual strategies), "
                             "uniform (sample normally), local (refine parent strategy)")
    args = parser.parse_args()

    use_vs = not args.no_verbalized_sampling
    task_profile = get_task_profile(args.task_config)
    mode_desc = {"tail": "tail-distribution", "uniform": "uniform/normal", "local": "local-refinement"}
    print("=" * 60)
    print(f"Task: {task_profile.name}")
    if use_vs:
        print(f"Tree Search with VS ({mode_desc[args.sampling_mode]})")
    else:
        print(f"Tree Search WITHOUT Verbalized Sampling")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Branching: {args.branching_factor}, Depth: {args.max_depth}")
    print(f"Max actions/node: {args.max_actions}")
    print(f"Temperature: {args.temperature}")
    print(f"Verbalized sampling: {use_vs}" + (f" (mode: {args.sampling_mode})" if use_vs else ""))
    print(f"Primary metric: {task_profile.primary_metric} ({'higher' if task_profile.higher_is_better else 'lower'} is better)")
    print()

    llm = LLMClient(args.vllm_url, args.model, args.temperature)
    container = ContainerManager(args.task_config, args.env_gpu, args.image_name,
                                 task_profile=task_profile)

    print("Creating MLGym container...")
    container.create()

    search = TreeSearch(
        llm=llm, container=container,
        task_profile=task_profile,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        max_actions=args.max_actions,
        output_dir=args.output_dir,
        verbose=args.verbose,
        verbalized_sampling=use_vs,
        sampling_mode=args.sampling_mode,
    )

    try:
        result = search.run()
        print(f"Results saved to {args.output_dir}/result.json")
    finally:
        container.close()


if __name__ == "__main__":
    main()
