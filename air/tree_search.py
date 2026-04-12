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
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

from mlgym.environment.env import EnvironmentArguments, MLGymEnv


def generate_code_outline(host_path: str, max_lines: int = 200) -> str:
    """Return a compact outline of a Python file: class/def lines with line numbers.

    Used to inject a table-of-contents for large baseline scripts so the executor
    can navigate via `goto <line>` without cat-ing the whole file.
    Returns empty string if the file is missing or smaller than 50 lines (outline
    adds no value for tiny files).
    """
    try:
        if not host_path or not Path(host_path).is_file():
            return ""
        src = Path(host_path).read_text()
        lines = src.splitlines()
        if len(lines) < 50:
            return ""
        outline_lines = []
        for i, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if stripped.startswith("class ") or stripped.startswith("def "):
                indent = len(line) - len(stripped)
                prefix = "  " + " " * indent
                # strip trailing : and function body
                sig = stripped.rstrip()
                if sig.endswith(":"):
                    sig = sig[:-1]
                outline_lines.append(f"{prefix}L{i:<5d} {sig}")
            if len(outline_lines) >= max_lines:
                outline_lines.append(f"  ... (outline truncated at {max_lines} entries)")
                break
        if not outline_lines:
            return ""
        total = len(lines)
        header = f"=== {Path(host_path).name} outline ({total} lines total — use `goto <line>` to jump) ===\n"
        footer = (
            "\n=== end outline ===\n"
            "After editing, regenerate with: grep -nE '^class |^def |^    def ' " + Path(host_path).name
        )
        return header + "\n".join(outline_lines) + footer
    except Exception:
        return ""

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
    starter_code_host_path: str = ""  # optional host path for generating {code_outline}
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

EDITING FILES:
- Config changes: sed -i 's/old/new/g' src/config.yaml
- Code changes: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  WARNING: python -c "..." DOES NOT WORK for multi-line code in this shell. Use heredoc instead.
- Single-line substitutions: sed -i 's/old_line/new_line/g' src/filename.py

COMMANDS:
- cat src/file - Read a file
- sed -i 's/old/new/g' file - In-place substitution
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Write/rewrite a file
- python src/train.py - Train the PPO agent (takes ~20-30 min, produces checkpoints/)
- validate - Score the checkpoints (ONLY after training completes)

CRITICAL RULES:
1. ONE command per response. No explanations.
2. ALWAYS run 'python src/train.py' BEFORE 'validate'. validate reads checkpoints/ — if you haven't trained, it will fail.
3. The class name 'Model' in networks.py MUST NOT be changed.
4. Do NOT rm -rf checkpoints unless you are about to retrain immediately after.
5. Do NOT use python -c "..." for multi-line code — it fails in this shell. Use heredoc.

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability. Do NOT pip install anything.
NOTE: distrax is NOT installed. Use tensorflow_probability.substrates.jax (already imported in networks.py).

MANDATORY WORKFLOW — follow this EXACT order:
1. cat src/config.yaml       (understand current hyperparameters)
2. cat src/networks.py       (understand the model)
3. sed -i / heredoc edits    (make your improvements — keep it to 1-3 changes max)
4. python src/train.py       (MUST train before validate — takes ~20-30 min)
5. validate

YOU MUST RUN python src/train.py BY ACTION 8 AT THE LATEST. Do not spend more than 5 actions on reading/editing.""",
        use_generic_conda=False,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="rl",
    ),

    "rlMountainCarContinuousReinforce": TaskProfile(
        name="Mountain Car Continuous REINFORCE (RL)",
        primary_metric="Reward Mean",
        higher_is_better=True,
        script_name="src/train.py",
        submission_file=None,
        data_head_cmd="cat src/config.yaml",
        strategy_topic="the MountainCarContinuous REINFORCE task (improve REINFORCE training to maximize mean reward)",
        branch_write_instruction=(
            "Modify the code and/or config to improve the agent's performance, then run 'python src/train.py', then 'validate'.\n"
            "You can modify any file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py.\n"
            "IMPORTANT: You MUST read the existing code first before writing ANY modifications.\n"
            "IMPORTANT: Do NOT rewrite entire files. Make TARGETED edits using sed -i.\n"
            "Output your first command (cat src/networks.py):"
        ),
        root_task_desc=(
            "MountainCarContinuous-v0 — RL REINFORCE Training.\n"
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

EDITING FILES:
- Config changes: sed -i 's/old/new/g' src/config.yaml
- Code changes: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  WARNING: python -c "..." DOES NOT WORK for multi-line code in this shell. Use heredoc instead.
- Single-line substitutions: sed -i 's/old_line/new_line/g' src/filename.py

COMMANDS:
- cat src/file - Read a file
- sed -i 's/old/new/g' file - In-place substitution
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Write/rewrite a file
- python src/train.py - Train the agent (takes ~20-30 min, produces checkpoints/)
- validate - Score the checkpoints (ONLY after training completes)

CRITICAL RULES:
1. ONE command per response. No explanations.
2. ALWAYS run 'python src/train.py' BEFORE 'validate'. validate reads checkpoints/ — if you haven't trained, it will fail.
3. The class name 'Model' in networks.py MUST NOT be changed.
4. Do NOT rm -rf checkpoints unless you are about to retrain immediately after.
5. Do NOT use python -c "..." for multi-line code — it fails in this shell. Use heredoc.

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability. Do NOT pip install anything.
NOTE: distrax is NOT installed. Use tensorflow_probability.substrates.jax (already imported in networks.py).

MANDATORY WORKFLOW — follow this EXACT order:
1. cat src/config.yaml       (understand current hyperparameters)
2. cat src/networks.py       (understand the model)
3. sed -i / heredoc edits    (make your improvements — keep it to 1-3 changes max)
4. python src/train.py       (MUST train before validate — takes ~20-30 min)
5. validate

YOU MUST RUN python src/train.py BY ACTION 8 AT THE LATEST. Do not spend more than 5 actions on reading/editing.""",
        use_generic_conda=False,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="rl",
    ),

    "rlBreakoutMinAtar": TaskProfile(
        name="Breakout MinAtar (RL)",
        primary_metric="Reward Mean",
        higher_is_better=True,
        script_name="src/train.py",
        submission_file=None,
        data_head_cmd="cat src/config.yaml",
        strategy_topic="the Breakout MinAtar RL task (improve PPO training for Breakout, maximize mean reward)",
        branch_write_instruction=(
            "Modify the code and/or config to improve the PPO agent's performance, then run 'python src/train.py', then 'validate'.\n"
            "You can modify any file: src/config.yaml, src/networks.py, src/policy.py, src/train.py, src/helpers.py.\n"
            "IMPORTANT: You MUST read the existing code first before writing ANY modifications.\n"
            "IMPORTANT: Do NOT rewrite entire files. Make TARGETED edits using sed -i.\n"
            "Output your first command (cat src/networks.py):"
        ),
        root_task_desc=(
            "Breakout MinAtar — RL PPO Training.\n"
            "Baseline Reward Mean: {baseline_score:.4f}\n\n"
            "Environment: Breakout-MinAtar (gymnax). Control paddle to bounce ball and break bricks.\n"
            "Ball moves diagonally, bounces off paddle/walls. Game ends when ball hits bottom.\n\n"
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

EDITING FILES:
- Config changes: sed -i 's/old/new/g' src/config.yaml
- Code changes: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  WARNING: python -c "..." DOES NOT WORK for multi-line code in this shell. Use heredoc instead.
- Single-line code substitutions: sed -i 's/old_line/new_line/g' src/filename.py

COMMANDS:
- cat src/file - Read a file
- sed -i 's/old/new/g' file - In-place substitution
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Write/rewrite a file
- python src/train.py - Train the PPO agent (takes ~20-30 min, produces checkpoints/)
- validate - Score the checkpoints (ONLY after training completes)

CRITICAL RULES:
1. ONE command per response. No explanations.
2. ALWAYS run 'python src/train.py' BEFORE 'validate'. validate reads checkpoints/ — if you haven't trained, it will fail.
3. The class name 'Model' in networks.py MUST NOT be changed.
4. Do NOT rm -rf checkpoints unless you are about to retrain immediately after.
5. Do NOT use python -c "..." for multi-line code — it fails in this shell. Use heredoc.

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability. Do NOT pip install anything.
NOTE: distrax is NOT installed. Use tensorflow_probability.substrates.jax (already imported in networks.py).

MANDATORY WORKFLOW — follow this EXACT order:
1. cat src/config.yaml       (understand current hyperparameters)
2. cat src/networks.py       (understand the model)
3. sed -i / heredoc edits    (make your improvements — keep it to 1-3 changes max)
4. python src/train.py       (MUST train before validate — takes ~20-30 min)
5. validate

YOU MUST RUN python src/train.py BY ACTION 8 AT THE LATEST. Do not spend more than 5 actions on reading/editing.""",
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
            "Output your first command (cat src/config.yaml):"
        ),
        root_task_desc=(
            "MetaMaze-misc — RL PPO Training.\n"
            "Baseline Reward Mean: {baseline_score:.4f}\n\n"
            "Environment: MetaMaze-misc (gymnax). Agent navigates grid maze to reach goal.\n"
            "Obs: local receptive field + last action + last reward + timestep.\n"
            "Actions: 4 discrete (up/right/down/left). Reward: +10 on goal. Episode: 200 steps.\n\n"
            "Current config:\n{data_head}\n\n"
            "Source files: src/train.py, src/networks.py, src/policy.py, src/helpers.py, src/config.yaml\n\n"
            "MANDATORY: Your first command MUST be cat src/config.yaml.\n"
            "Then make 1-3 targeted changes, then run python src/train.py.\n"
            "YOU MUST run python src/train.py by action 8 at the latest.\n\n"
            "Goal: Maximize mean reward. Output your first command (cat src/config.yaml):"
        ),
        system_prompt="""You are an RL research agent. Output ONLY ONE command per response. No explanations.

EDITING FILES:
- Config changes: sed -i 's/old/new/g' src/config.yaml
- Code changes: cat << 'ENDOFFILE' > src/filename.py ... ENDOFFILE
  WARNING: python -c "..." DOES NOT WORK for multi-line code in this shell. Use heredoc instead.
- Single-line code substitutions: sed -i 's/old_line/new_line/g' src/filename.py

COMMANDS:
- cat src/file - Read a file
- sed -i 's/old/new/g' file - In-place substitution
- cat << 'ENDOFFILE' > file ... ENDOFFILE - Write/rewrite a file
- python src/train.py - Train the PPO agent (takes ~20-30 min, produces checkpoints/)
- validate - Score the checkpoints (ONLY after training completes)

CRITICAL RULES:
1. ONE command per response. No explanations.
2. ALWAYS run 'python src/train.py' BEFORE 'validate'. validate reads checkpoints/ — if you haven't trained, it will fail.
3. The class name 'Model' in networks.py MUST NOT be changed.
4. Do NOT rm -rf checkpoints unless you are about to retrain immediately after.
5. Do NOT use python -c "..." for multi-line code — it fails in this shell. Use heredoc.

AVAILABLE PACKAGES: jax, flax, gymnax, optax, numpy, tensorflow_probability. Do NOT pip install anything.
NOTE: distrax is NOT installed. Use tensorflow_probability.substrates.jax (already imported in networks.py).

MANDATORY WORKFLOW — follow this EXACT order:
1. cat src/config.yaml       (understand current hyperparameters)
2. cat src/networks.py       (understand the model)
3. sed -i / heredoc edits    (make your improvements — keep it to 1-3 changes max)
4. python src/train.py       (MUST train before validate — takes ~20-30 min)
5. validate

YOU MUST RUN python src/train.py BY ACTION 8 AT THE LATEST. Do not spend more than 5 actions on reading/editing.""",
        use_generic_conda=False,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="rl",
    ),
    "imageClassificationFMnist": TaskProfile(
        name="Fashion MNIST Image Classification",
        primary_metric="accuracy",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="python -c \"from datasets import load_dataset; ds=load_dataset('zalando-datasets/fashion_mnist',split='train[:3]'); print('Columns:', ds.column_names); print('Labels:', ds['label']); print('Image shape:', ds[0]['image'].size)\"",
        strategy_topic="Fashion MNIST image classification (10-class clothing classification, maximize accuracy)",
        branch_write_instruction=(
            "Write a complete train_and_predict.py that loads Fashion MNIST from HuggingFace, "
            "trains a classifier, predicts on the test set, and saves submission.csv with a single 'label' column.\n"
            "Then run 'python train_and_predict.py', then 'validate'.\n"
            "Output your first command (write the file with cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        root_task_desc=(
            "Fashion MNIST Image Classification.\n"
            "Baseline accuracy: {baseline_score:.4f}\n\n"
            "Dataset: zalando-datasets/fashion_mnist from HuggingFace (28x28 grayscale images, 10 classes).\n"
            "Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.\n"
            "Train: 60,000 images. Test: 10,000 images.\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a classifier, predict labels for the test set, save as submission.csv with a single 'label' column.\n\n"
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
3. ALWAYS run 'python train_and_predict.py' BEFORE 'validate'
4. Load data from HuggingFace: load_dataset("zalando-datasets/fashion_mnist")
5. submission.csv must have a single 'label' column with predicted class indices (0-9)

AVAILABLE PACKAGES: pandas, numpy, scikit-learn, xgboost, lightgbm, torch, transformers, scipy, datasets. Do NOT pip install anything.

PERFORMANCE TIPS (CRITICAL — code will be killed after 120s):
- Load data: `ds = load_dataset("zalando-datasets/fashion_mnist"); X = np.array(ds['train']['image']).reshape(-1, 784) / 255.0`
- Do NOT iterate samples one by one — use batch numpy operations
- For sklearn: use 10000 training samples max. Use max_iter=100.
- For neural nets: 10 epochs max.
- Run: timeout 120 python train_and_predict.py

WORKSPACE:
- baseline.py - Reference baseline (PyTorch CNN via HuggingFace Trainer)
- evaluate.py - Evaluation script (read-only)
- sample_submission.csv - Submission format example
- Output: submission.csv with predicted labels

DO NOT modify evaluate.py — it is read-only and will be restored before evaluation. Your submission.csv must contain predictions from an actual trained model on the test data.

MANDATORY WORKFLOW:
1. cat << 'ENDOFFILE' > train_and_predict.py
<complete python script>
ENDOFFILE
2. python train_and_predict.py
3. validate""",
        task_type="classification",
    ),

    "languageModelingFineWeb": TaskProfile(
        name="Language Modeling (FineWeb GPT-2)",
        primary_metric="val_loss",
        higher_is_better=False,
        script_name="baseline.py",
        submission_file=None,  # model.pt + model_config.pt, not CSV
        starter_code_host_path="/home/jarnav/MLScientist/MLGym/data/languageModelingFineWeb/baseline.py",
        data_head_cmd=None,
        strategy_topic="language model training (GPT-2 style on FineWeb, minimize validation loss)",
        branch_write_instruction=(
            "Modify baseline.py to improve the model, then run "
            "'torchrun --nproc_per_node=1 --standalone baseline.py' to train, "
            "then type the literal word 'validate' (NOT 'torchrun evaluate.py') to evaluate.\n"
            "You MUST read baseline.py first before making any changes. Use `open baseline.py` — do NOT use `cat`.\n"
            "Output your first command (suggest: open baseline.py):"
        ),
        root_task_desc=(
            "Language Modeling — GPT-2 on FineWeb.\n"
            "Baseline val_loss: {baseline_score:.4f}\n\n"
            "This is a modded-nanogpt GPT-2 style model trained on FineWeb (2.4B tokens).\n"
            "Architecture: 12 transformer layers, 6 heads, 768 embedding dim, RoPE, RMSNorm, ReLU^2.\n"
            "Three optimizers: Adam for embeddings (lr=0.3), Adam for LM head (lr=0.002), Muon for transformer blocks (lr=0.02).\n"
            "Training: 500 iterations, batch_size=512, seq_len=1024, bfloat16, torch.compile, DDP.\n\n"
            "Files: baseline.py (511 lines, 22KB), evaluate.py (read-only).\n"
            "Submission: model.pt (state_dict) + model_config.pt (pickle config).\n\n"
            "{code_outline}\n\n"
            "Goal: Minimize val_loss. Use `open baseline.py <line>` + `goto` to read specific sections, then `edit` to modify.\n\n"
            "Output your first command:"
        ),
        system_prompt="""You are an ML research agent. Output ONLY ONE shell command per response.

=== RESPONSE FORMAT RULES (CRITICAL — violations waste action budget) ===
- Output EXACTLY ONE valid shell command. Nothing else. No prose. No commentary. No "Now I will...", "Let me...", "Perfect!", "Excellent!", "Based on...", "The results show...".
- The first character of your response must be the start of a runnable shell command.
- Do NOT include reasoning, analysis, or explanations — those go in your internal thought, NOT in the output.
- Bad examples (these all get sent to bash and fail):
  ❌ "Now let me train the model: torchrun ..."  → bash sees "Now" as a command
  ❌ "Perfect! The edit is applied. Now running training."  → bash sees "Perfect!" as a command
  ❌ "Based on the results, I'll try..."  → bash sees "Based" as a command
- Good examples:
  ✅ open baseline.py 193
  ✅ torchrun --nproc_per_node=1 --standalone baseline.py
  ✅ validate

TASK: Train a GPT-2 style language model on FineWeb. Minimize validation loss.

WORKSPACE FILES:
- baseline.py — Full training script (modded-nanogpt, distributed PyTorch)
- evaluate.py — Evaluation script (READ-ONLY, do not modify, do not run directly)

COMMANDS (use these windowed tools — do NOT cat the whole 22KB baseline.py, it will blow context):
- open baseline.py [line_number] — Opens file in ~100-line window. Preferred over cat.
- goto <line> — Jump the open window to a specific line
- scroll_down, scroll_up — Move the window
- search_file <term> — Grep inside the currently open file (shows matching lines)
- search_dir <term> — Grep a directory
- find_file <name> — Locate a file
- edit <start>:<end>\\n<new_code>\\nend_of_edit — Replace line range with new code (lint-checked)
- sed -i 's/old/new/g' baseline.py — Simple targeted edits (for one-liners)
- torchrun --nproc_per_node=1 --standalone baseline.py — Train
- validate — Evaluate (ONLY after training produces model_config.pt)
- ls — List directory

SCORING: Your final score is recorded ONLY by the literal `validate` command — other evaluation attempts will not be recorded.

AVOID: `cat baseline.py` (22KB, fills context), `python -c "..."` for reading (same issue).
PREFER: `open baseline.py` + `goto 200` + `scroll_down` to navigate. `search_file 'class Block'` to jump to a class.

TUNABLE KNOBS IN baseline.py:
- Hyperparameters dataclass: num_iterations, warmup_iters, warmdown_iters, weight_decay, device_batch_size, sequence_length
- Optimizer LRs: embedding lr=0.3, lm_head lr=0.002, transformer lr=0.02
- Optimizer params: betas, momentum
- Architecture: n_layer, n_head, n_embd (in Model constructor call)
- Activation function (currently relu^2), normalization, attention config

CRITICAL RULES:
1. ONE shell command per response. NEVER prose.
2. ALWAYS read baseline.py BEFORE modifying it (use `open`, not `cat`)
3. ALWAYS run training BEFORE validate
4. Training command: torchrun --nproc_per_node=1 --standalone baseline.py
5. Do NOT modify evaluate.py, do NOT run `torchrun evaluate.py` or `python evaluate.py`
6. Scoring: the ONLY way to record a score is the literal `validate` command. Manual evaluate.py runs are NOT counted.
7. When modifying code with python -c, always compile() before writing to catch syntax errors
8. Commit early: as soon as `validate` returns a valid score (even if worse than baseline), the node is done. Do not try to beat baseline within a single node — that's the scientist's job across nodes.

TIMING:
- The baseline configuration (500 iterations, 12 layers, 768 dim, batch 512) trains in ~4 minutes on this hardware.
- torchrun has a 40-minute hard timeout. If training exceeds 40 min, it gets killed and model_config.pt is NOT saved → FAILED node.
- After EVERY torchrun, run `ls model_config.pt` to verify the file exists BEFORE validating.

MANDATORY WORKFLOW:
1. open baseline.py  (READ first, windowed)
2. Navigate with goto/scroll_up/scroll_down/search_file to understand relevant sections
3. Make targeted modifications (edit <start>:<end>, sed -i, or python -c)
4. torchrun --nproc_per_node=1 --standalone baseline.py
5. ls model_config.pt  (verify file was saved)
6. validate""",
        use_generic_conda=True,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="language_modeling",
    ),

    "naturalLanguageInferenceMNLI": TaskProfile(
        name="Natural Language Inference (BERT on MNLI)",
        primary_metric="validation_accuracy",
        higher_is_better=True,
        script_name="baseline.py",
        submission_file=None,  # bert_mnli_model/ folder
        data_head_cmd=None,
        strategy_topic="BERT fine-tuning for natural language inference (MNLI 3-class, maximize validation accuracy)",
        branch_write_instruction=(
            "Modify baseline.py to improve BERT fine-tuning, then run 'python baseline.py', "
            "then 'validate'.\n"
            "You MUST read baseline.py first before making any changes.\n"
            "Output your first command (cat baseline.py):"
        ),
        root_task_desc=(
            "Natural Language Inference — BERT on MNLI.\n"
            "Baseline validation_accuracy: {baseline_score:.4f}\n\n"
            "Fine-tuning bert-base-uncased on MNLI (3 classes: entailment, contradiction, neutral).\n"
            "Current config: EPOCHS=1, BATCH_SIZE=32, LEARNING_RATE=1e-7, MAX_LENGTH=128.\n"
            "Uses mixed-precision training (GradScaler + autocast).\n\n"
            "Files: baseline.py (training script), evaluate.py (read-only evaluation).\n"
            "Submission: bert_mnli_model/ folder (model.save_pretrained).\n\n"
            "Goal: Maximize validation accuracy. Read baseline.py first, then modify and train.\n\n"
            "Output your first command (cat baseline.py):"
        ),
        system_prompt="""You are an ML research agent. Output ONLY ONE command per response. No explanations.

TASK: Fine-tune BERT for natural language inference on MNLI. Maximize validation accuracy.

WORKSPACE FILES:
- baseline.py — Training script (BERT fine-tuning with HuggingFace transformers)
- evaluate.py — Evaluation script (READ-ONLY, do not modify)

COMMANDS:
- cat baseline.py — Read the training script
- sed -i 's/old/new/g' baseline.py — Make targeted edits
- python -c "..." — Programmatic edits (read, modify, compile-check, write)
- cat << 'ENDOFFILE' > baseline.py ... ENDOFFILE — Full file rewrite
- python baseline.py — Train
- validate — Evaluate (ONLY after training produces bert_mnli_model/)
- ls, head — View files

TUNABLE KNOBS IN baseline.py:
- EPOCHS (currently 1 — very low, likely the biggest lever)
- BATCH_SIZE (currently 32)
- LEARNING_RATE (currently 1e-7 — very low)
- MAX_LENGTH (currently 128)
- Model choice (currently bert-base-uncased, could try bert-large or other models)
- Optimizer (currently AdamW)
- Learning rate scheduling (currently none — add warmup/decay)
- Weight decay, gradient clipping

CRITICAL RULES:
1. ONE command per response
2. ALWAYS read baseline.py BEFORE modifying it
3. ALWAYS run training BEFORE validate
4. Do NOT modify evaluate.py
5. When modifying code with python -c, always compile() before writing to catch syntax errors
6. Submission is saved by model.save_pretrained('bert_mnli_model')

MANDATORY WORKFLOW:
1. cat baseline.py  (READ first)
2. Make modifications (sed -i or python -c or full rewrite)
3. python baseline.py
4. validate""",
        use_generic_conda=True,
        needs_gpu=True,
        step_timeout=2400.0,
        task_type="nlp",
    ),
    # ---- Additional tasks (game theory, vision, NLP, optimization) ----
    "blotto": TaskProfile(
        name="Colonel Blotto Game",
        primary_metric="Score",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls /home/agent/workspace/",
        strategy_topic="the Colonel Blotto game (allocate soldiers across battlefields to maximize expected score)",
        branch_write_instruction="Write a complete strategy script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "Colonel Blotto Game.\n"
            "Baseline Score: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Design a strategy to allocate soldiers across battlefields. Write and run your strategy.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are a game theory agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="game_theory",
    ),
    "prisonersDilemma": TaskProfile(
        name="Iterated Prisoner's Dilemma",
        primary_metric="Score",
        higher_is_better=True,
        script_name="strategy.py",
        submission_file=None,  # no CSV — validate imports strategy.py directly
        data_head_cmd=None,
        strategy_topic="the Iterated Prisoner's Dilemma (design a strategy to maximize cumulative score against unknown opponents)",
        branch_write_instruction="Write a complete strategy script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "Iterated Prisoner's Dilemma.\n"
            "Baseline Score: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Design a strategy for iterated PD (cooperate=0, defect=1). Maximize total score.\n\n"
            "Write strategy.py now (use cat << 'ENDOFFILE' > strategy.py):"
        ),
        system_prompt="You are a game theory agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.\nDO NOT modify evaluate.py or target.py — they are read-only and will be restored before evaluation. ONLY modify strategy.py.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="game_theory",
    ),
    "3SATTime": TaskProfile(
        name="3-SAT Solver Heuristic Optimization",
        primary_metric="Time",
        higher_is_better=False,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls /home/agent/workspace/",
        strategy_topic="the 3-SAT heuristic optimization task (design variable selection heuristics to minimize solving time)",
        branch_write_instruction="Write a solver heuristic script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "3-SAT Solver Heuristic Optimization.\n"
            "Baseline Time: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Design a DPLL variable selection heuristic to solve 3-SAT instances faster. Minimize time.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an optimization agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="optimization",
    ),
    "imageCaptioningCOCO": TaskProfile(
        name="Image Captioning (MS-COCO)",
        primary_metric="BLEU Score",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls /home/agent/workspace/",
        strategy_topic="the MS-COCO image captioning task (train a model to generate captions for images, maximize BLEU score)",
        branch_write_instruction="Write a training script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "Image Captioning (MS-COCO).\n"
            "Baseline BLEU Score: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train an image captioning model. Maximize BLEU score on test set.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an ML research agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.\nDO NOT modify evaluate.py — it is read-only and will be restored before evaluation.",
        use_generic_conda=True,
        needs_gpu=True,
        task_type="vision_nlp",
    ),
    "imageClassificationCifar10": TaskProfile(
        name="CIFAR-10 Image Classification",
        primary_metric="accuracy",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls /home/agent/workspace/",
        strategy_topic="the CIFAR-10 image classification task (train a classifier on 32x32 color images, 10 classes)",
        branch_write_instruction="Write a training script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "CIFAR-10 Image Classification.\n"
            "Baseline accuracy: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a classifier for CIFAR-10 (10 classes, 32x32 color images). Maximize accuracy.\n\n"
            "IMPORTANT: Load data via HuggingFace `from datasets import load_dataset; ds = load_dataset('uoft-cs/cifar10')`. "
            "The container has NO internet — torchvision.datasets.CIFAR10(download=True) will FAIL.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an ML research agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.\nDO NOT modify evaluate.py — it is read-only and will be restored before evaluation.\nLoad CIFAR-10 from HuggingFace: load_dataset('uoft-cs/cifar10'). Do NOT use torchvision.datasets — no internet access.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="vision",
    ),
    "imageClassificationCifar10L1": TaskProfile(
        name="CIFAR-10 Image Classification (L1 variant)",
        primary_metric="accuracy",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls /home/agent/workspace/",
        strategy_topic="the CIFAR-10 L1 image classification task (classify 32x32 color images with L1 regularization constraint)",
        branch_write_instruction="Write a training script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "CIFAR-10 Image Classification (L1 variant).\n"
            "Baseline accuracy: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Classify CIFAR-10 images. Maximize accuracy.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an ML research agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.\nDO NOT modify evaluate.py — it is read-only and will be restored before evaluation. Your submission.csv must contain predictions from an actual trained model.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="vision",
    ),
    "regressionHousingPrice": TaskProfile(
        name="Housing Price Prediction",
        primary_metric="rmse",
        higher_is_better=False,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -5 /home/agent/workspace/data/train.csv 2>/dev/null || ls /home/agent/workspace/",
        strategy_topic="the housing price prediction task (minimize RMSE on house price regression)",
        branch_write_instruction="Write a training script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "Housing Price Prediction.\n"
            "Baseline RMSE: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a regression model to predict house prices. Minimize RMSE.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an ML research agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="tabular",
    ),
    "regressionHousePrice": TaskProfile(
        name="Housing Price Prediction",
        primary_metric="rmse",
        higher_is_better=False,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -5 /home/agent/workspace/data/train.csv 2>/dev/null || ls /home/agent/workspace/",
        strategy_topic="the housing price prediction task (minimize RMSE on house price regression)",
        branch_write_instruction="Write a training script, then run it, then 'validate'.\nOutput your first command:",
        root_task_desc=(
            "Housing Price Prediction.\n"
            "Baseline RMSE: {baseline_score}\n\n"
            "Data preview:\n{data_head}\n\n"
            "Goal: Train a regression model to predict house prices. Minimize RMSE.\n\n"
            "Write train_and_predict.py now (use cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="You are an ML research agent. Output ONLY ONE command per response. No explanations.\nAvailable: cat, python, validate.",
        use_generic_conda=True,
        needs_gpu=False,
        task_type="tabular",
    ),

    # ---- MLE-bench tasks ----
    "mlebenchJigsawToxic": TaskProfile(
        name="Jigsaw Toxic Comment Classification (MLE-bench)",
        primary_metric="column_wise_roc_auc",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -3 data/train.csv 2>/dev/null | cut -c1-120",
        strategy_topic="multilabel toxic comment classification (predict probability for 6 toxicity categories, maximize mean column-wise ROC AUC)",
        branch_write_instruction=(
            "Write a complete train_and_predict.py, run it, then 'validate'.\n"
            "Read baseline.py first for data layout.\n"
            "Output your first command (cat baseline.py):"
        ),
        root_task_desc=(
            "Jigsaw Toxic Comment Classification (MLE-bench).\n"
            "Baseline column_wise_roc_auc: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "TASK: Predict toxicity probabilities for Wikipedia comments.\n"
            "Files: data/train.csv (labeled), data/test.csv (unlabeled), data/sample_submission.csv\n"
            "Output: submission.csv with columns: id, toxic, severe_toxic, obscene, threat, insult, identity_hate\n\n"
            "Baseline (TF-IDF + LR per class) scores ~0.96 ROC AUC. Top sklearn solutions reach ~0.988.\n"
            "Try: char n-grams (3-6), larger vocab, LR+LinearSVC+NB ensembles, LightGBM on TF-IDF.\n"
            "IMPORTANT: Only sklearn/scipy/numpy/pandas available — no transformers/torch.\n\n"
            "Write train_and_predict.py now (cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML engineering agent. Output ONLY ONE command per response. No explanations.

TASK: Jigsaw Toxic Comment Classification. Maximize column-wise mean ROC AUC.

WORKSPACE:
- data/train.csv    — comment_text + 6 binary label columns (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- data/test.csv     — comment_text only (no labels)
- data/sample_submission.csv — expected submission format
- baseline.py       — reference TF-IDF + LR baseline

COMMANDS:
- cat baseline.py            — read baseline
- head -5 data/train.csv     — preview data
- cat << 'ENDOFFILE' > train_and_predict.py ... ENDOFFILE — write script
- python train_and_predict.py — run training + prediction
- validate                   — score submission.csv (column-wise ROC AUC)
- ls, head, wc -l, sed -i    — file utilities

OUTPUT: submission.csv with columns: id, toxic, severe_toxic, obscene, threat, insult, identity_hate
        Values = probabilities in [0, 1] for each comment.

AVAILABLE PACKAGES (use ONLY these — no internet, no pip install):
- sklearn, scipy, numpy, pandas (standard conda env)
- lightgbm, xgboost (may be present)
DO NOT USE: transformers, torch, tensorflow, keras, sentence_transformers, spacy, nltk (NOT installed)

KEY STRATEGIES (sklearn-only, in order of preference):
1. TF-IDF (word + char n-grams, 50k-200k vocab) + LogisticRegression per class (~0.96)
2. Add char-level n-grams (3-6 chars), increase max_features
3. Ensemble: LR + LinearSVC + MultinomialNB with averaged probabilities
4. LightGBM or XGBoost on TF-IDF features (if available)
5. Better preprocessing: lowercase, strip punctuation, remove numbers

CRITICAL RULES:
1. ONE command per response
2. Read baseline.py before writing new code
3. Always run python train_and_predict.py BEFORE validate
4. submission.csv MUST have exact column order: id, toxic, severe_toxic, obscene, threat, insult, identity_hate
5. NEVER import transformers, torch, tensorflow, or any deep learning library

MANDATORY WORKFLOW:
1. cat baseline.py   (understand data + format)
2. Write train_and_predict.py
3. python train_and_predict.py
4. validate""",
        use_generic_conda=True,
        needs_gpu=False,
        step_timeout=900.0,
        task_type="nlp",
    ),
    "mlebenchVesuvius": TaskProfile(
        name="Vesuvius Challenge Ink Detection (MLE-bench)",
        primary_metric="f05_score",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="ls data/train/1/surface_volume/*.tif 2>/dev/null | head -5; echo '---'; head -3 data/sample_submission.csv",
        strategy_topic="binary segmentation of ink in 3D X-ray CT scans of ancient scrolls (predict RLE-encoded pixel masks, maximize F0.5 score)",
        branch_write_instruction=(
            "Write a complete train_and_predict.py, run it, then 'validate'.\n"
            "Read baseline.py first for data layout.\n"
            "Output your first command (cat baseline.py):"
        ),
        root_task_desc=(
            "Vesuvius Challenge Ink Detection (MLE-bench).\n"
            "Baseline f05_score: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "TASK: Detect ink in 3D X-ray CT scans of ancient Herculaneum scroll fragments.\n"
            "Files (all under data/ directory):\n"
            "  data/train/1/surface_volume/ — 65 TIF slices (3D volume)\n"
            "  data/train/1/inklabels.png  — ground truth binary mask\n"
            "  data/train/1/mask.png       — valid region mask\n"
            "  data/train/2/              — second fragment (same structure)\n"
            "  data/test/a/surface_volume/ — test fragment slices\n"
            "  data/test/a/mask.png       — valid region mask\n"
            "  data/sample_submission.csv — format reference\n\n"
            "Output: submission.csv with columns: Id, Predicted (RLE-encoded binary mask)\n"
            "RLE format: space-delimited pairs of (start_pixel run_length), 1-indexed, left-to-right top-to-bottom\n\n"
            "APPROACHES (in order of expected quality):\n"
            "1. U-Net or ResNet encoder on 2.5D slices (stack of ~10 middle slices as channels)\n"
            "2. 3D CNN on local volume patches\n"
            "3. EfficientNet/ResNet feature extraction + pixel-wise classifier\n"
            "4. Simple threshold on mean intensity (baseline, very poor)\n\n"
            "PyTorch, torchvision are available. Build models from torch.nn (no smp).\n"
            "GPU is available. START SIMPLE (threshold/sklearn), validate, then improve.\n\n"
            "Write train_and_predict.py now (cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML engineering agent. Output ONLY ONE command per response. No explanations.

TASK: Vesuvius Challenge Ink Detection. Maximize F0.5 score on binary ink segmentation.

WORKSPACE:
- data/train/1/surface_volume/  — 65 TIF slices of 3D X-ray CT volume (fragment 1)
- data/train/1/inklabels.png    — ground truth binary ink mask
- data/train/1/mask.png         — valid region mask
- data/train/2/                 — fragment 2 (same structure)
- data/test/a/surface_volume/   — test fragment slices
- data/test/a/mask.png          — valid region mask
- data/sample_submission.csv    — expected submission format
- baseline.py                   — reference baseline

COMMANDS:
- cat baseline.py                    — read baseline
- ls data/train/1/surface_volume/    — see available slices
- cat << 'ENDOFFILE' > train_and_predict.py ... ENDOFFILE — write script
- python train_and_predict.py — run training + prediction
- validate                   — score submission.csv (F0.5)
- ls, head, wc -l, sed -i    — file utilities

OUTPUT: submission.csv with columns: Id, Predicted
        Predicted = RLE-encoded binary mask (1-indexed, space-delimited start-length pairs)

AVAILABLE PACKAGES (use ONLY these — do NOT pip install anything):
- torch, torchvision (GPU available — use CUDA)
- numpy, pandas, scipy, sklearn, PIL/Pillow
DO NOT USE: segmentation_models_pytorch (broken), albumentations, cv2 (not installed)
Build ALL models from scratch using torch.nn and torchvision only.

STRATEGY — START SIMPLE, THEN IMPROVE:
Phase 1 (MUST complete and validate first):
  - Load middle ~10 TIF slices with PIL, compute mean intensity per pixel
  - Simple threshold or sklearn classifier (RandomForest on patch features)
  - Produce submission.csv with RLE encoding and run validate
  - This gets a NON-ZERO baseline score
Phase 2 (only after Phase 1 validates successfully):
  - Replace with CNN using torchvision.models.resnet18(pretrained=True) as encoder
  - Build decoder with torch.nn.ConvTranspose2d layers
  - Train on 256x256 patches, predict on test, threshold, RLE-encode
Phase 3: Data augmentation, longer training, ensemble

CRITICAL RULES:
1. ONE command per response
2. Read baseline.py before writing new code
3. ALWAYS produce submission.csv and validate BEFORE trying complex models
4. submission.csv MUST have columns: Id, Predicted
5. RLE encoding: 1-indexed, left-to-right top-to-bottom, pairs of (start, length)
6. DO NOT pip install anything — use only pre-installed packages
7. If code errors, SIMPLIFY — a working simple model beats a broken complex one
8. A score of 0.01 is infinitely better than no score at all

MANDATORY WORKFLOW:
1. cat baseline.py   (understand data + format)
2. Write SIMPLE train_and_predict.py (threshold or sklearn FIRST)
3. python train_and_predict.py
4. validate — MUST succeed before trying anything else
5. Only THEN write more complex approaches""",
        use_generic_conda=True,
        needs_gpu=True,
        step_timeout=1800.0,
        task_type="vision",
    ),
    "mlebenchBMS": TaskProfile(
        name="BMS Molecular Translation (MLE-bench)",
        primary_metric="mean_levenshtein_distance",
        higher_is_better=False,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -3 data/train_labels.csv 2>/dev/null; echo '---'; head -3 data/sample_submission.csv",
        strategy_topic="image-to-sequence prediction of InChI chemical identifiers from molecule images (minimize mean Levenshtein distance)",
        branch_write_instruction=(
            "Write a complete train_and_predict.py, run it, then 'validate'.\n"
            "Read baseline.py first for data layout.\n"
            "Output your first command (cat baseline.py):"
        ),
        root_task_desc=(
            "BMS Molecular Translation (MLE-bench).\n"
            "Baseline mean_levenshtein_distance: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "TASK: Predict InChI chemical identifier strings from molecule structure images.\n"
            "Files:\n"
            "  data/train_labels.csv          — image_id, InChI columns\n"
            "  data/train/<nested>/image.png  — training images (triple-nested: id[0]/id[1]/id[2]/id.png)\n"
            "  data/test/<nested>/image.png   — test images (same nesting)\n"
            "  sample_submission.csv     — format reference\n"
            "  data/extra_approved_InChIs.csv — reference InChI data\n\n"
            "Output: submission.csv with columns: image_id, InChI\n\n"
            "PROVEN APPROACH (scored 62.5, improve on it):\n"
            "  ResNet50 encoder + LSTM decoder with attention, 150k-300k samples, 8-15 epochs,\n"
            "  beam search decoding (width 5), MAX_SEQ_LENGTH=300, lr=1e-4\n"
            "ALTERNATIVE APPROACHES:\n"
            "  - Transformer decoder (better for long sequences)\n"
            "  - EfficientNet-B3 encoder\n"
            "  - BPE tokenization instead of character-level\n\n"
            "PyTorch, torchvision available. GPU available. Training takes 20-40 min per run.\n\n"
            "Write train_and_predict.py now (cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML engineering agent. Output ONLY ONE command per response. No explanations.

TASK: BMS Molecular Translation. Minimize mean Levenshtein distance between predicted and true InChI strings.

WORKSPACE:
- data/train_labels.csv          — image_id, InChI columns (large dataset)
- data/train/<nested>/image.png  — molecule images (nested: id[0]/id[1]/id[2]/id.png)
- data/test/<nested>/image.png   — test molecule images (same nesting)
- sample_submission.csv     — expected submission format
- data/extra_approved_InChIs.csv — reference InChI data
- baseline.py               — reference baseline

COMMANDS:
- cat baseline.py            — read baseline
- head -5 data/train_labels.csv   — preview data
- wc -l data/train_labels.csv     — count training samples
- cat << 'ENDOFFILE' > train_and_predict.py ... ENDOFFILE — write script
- python train_and_predict.py — run training + prediction
- validate                   — score submission.csv (Levenshtein distance)
- ls, head, wc -l, sed -i    — file utilities

OUTPUT: submission.csv with columns: image_id, InChI
        InChI = International Chemical Identifier string (e.g., "InChI=1S/C6H12O6/...")

AVAILABLE PACKAGES:
- torch, torchvision (GPU available — use CUDA)
- numpy, pandas, scipy, sklearn, PIL/Pillow
- cv2 (OpenCV)

PROVEN APPROACH (ResNet + LSTM + Attention scored 62.5 — improve on this):
1. CNN encoder: ResNet50 (pretrained) — extract 2048-dim features from molecule images
2. Decoder: LSTM with attention over encoder features, character-level output
3. Training data: use 150k-300k samples (more data = much better, 50k is too few)
4. Epochs: train 8-15 epochs (3 epochs is too few — loss is still dropping)
5. Beam search decoding (width 5) for inference — greedy decoding loses ~10-20 points
6. MAX_SEQ_LENGTH = 300 (some InChI strings exceed 200 chars)
7. Image size: 256x256, standard ImageNet normalization
8. Batch size: 32-64 depending on GPU memory
9. Learning rate: 1e-4 with ReduceLROnPlateau scheduler

ALTERNATIVE APPROACHES TO EXPLORE:
- Transformer decoder instead of LSTM (potentially better for long sequences)
- EfficientNet-B3 encoder (more powerful features)
- BPE or InChI-layer tokenization instead of character-level
- Data augmentation: random rotation, contrast adjustment

CRITICAL RULES:
1. ONE command per response
2. Read baseline.py before writing new code
3. Always run python train_and_predict.py BEFORE validate
4. submission.csv MUST have columns: image_id, InChI
5. Predict for ALL test images — missing predictions get max distance
6. Use GPU (torch.cuda) for training — CPU will be too slow
7. DO NOT pip install anything — use only pre-installed packages
8. ALWAYS produce a submission and validate before trying to improve

MANDATORY WORKFLOW:
1. cat baseline.py   (understand data + format)
2. Write train_and_predict.py (use ResNet50 + LSTM + Attention)
3. python train_and_predict.py (will take 20-40 min — this is normal)
4. validate
5. Iterate: more data, more epochs, beam search""",
        use_generic_conda=True,
        needs_gpu=True,
        step_timeout=3600.0,
        task_type="nlp",
    ),
    "mlebench3DDetection": TaskProfile(
        name="3D Object Detection for Autonomous Vehicles (MLE-bench)",
        primary_metric="mean_average_precision",
        higher_is_better=True,
        script_name="train_and_predict.py",
        submission_file="submission.csv",
        data_head_cmd="head -3 data/train.csv 2>/dev/null | cut -c1-120; echo '---'; head -3 data/sample_submission.csv",
        strategy_topic="3D object detection from LiDAR point clouds (predict 3D bounding boxes, maximize mAP at IoU 0.5-0.95)",
        branch_write_instruction=(
            "Write a complete train_and_predict.py, run it, then 'validate'.\n"
            "Read baseline.py first for data layout.\n"
            "Output your first command (cat baseline.py):"
        ),
        root_task_desc=(
            "3D Object Detection for Autonomous Vehicles (MLE-bench).\n"
            "Baseline mean_average_precision: {baseline_score:.4f}\n\n"
            "Data preview:\n{data_head}\n\n"
            "TASK: Detect 3D objects (cars, pedestrians, etc.) from LiDAR point clouds.\n"
            "Files:\n"
            "  data/train.csv                — Id, PredictionString (ground truth bounding boxes)\n"
            "  data/train_data/              — JSON metadata (log.json, sample.json, sample_data.json, etc.)\n"
            "  data/train_images/            — JPEG camera images\n"
            "  data/train_lidar/             — Binary LiDAR point clouds (.bin, 5 floats per point: x,y,z,intensity,ring)\n"
            "  data/train_maps/              — Semantic map PNGs\n"
            "  data/test_data/               — JSON metadata for test (no annotations)\n"
            "  data/test_images/             — Test camera images\n"
            "  data/test_lidar/              — Test LiDAR files\n"
            "  data/test_maps/               — Test semantic maps\n"
            "  data/sample_submission.csv    — format reference\n\n"
            "Output: submission.csv with columns: Id, PredictionString\n"
            "PredictionString format per object: confidence center_x center_y center_z width length height yaw class_name\n"
            "9 object classes: animal, bicycle, bus, car, emergency_vehicle, motorcycle, other_vehicle, pedestrian, truck\n\n"
            "APPROACHES (in order of expected quality):\n"
            "1. PointPillars: voxelize LiDAR → 2D pseudo-image → 2D detection backbone\n"
            "2. BEV (bird's eye view) projection of LiDAR → 2D detection\n"
            "3. Simple heuristic: cluster LiDAR points, fit bounding boxes\n\n"
            "PyTorch is available. GPU is available. Use it.\n\n"
            "Write train_and_predict.py now (cat << 'ENDOFFILE' > train_and_predict.py):"
        ),
        system_prompt="""You are an ML engineering agent. Output ONLY ONE command per response. No explanations.

TASK: 3D Object Detection for Autonomous Vehicles. Maximize mAP at IoU thresholds 0.5-0.95.

WORKSPACE:
- data/train.csv                — Id, PredictionString columns (bounding box annotations)
- data/train_data/              — JSON metadata (log.json, sample.json, sample_data.json, etc.)
- data/train_images/            — JPEG camera images
- data/train_lidar/             — Binary LiDAR point clouds (.bin files)
- data/train_maps/              — Semantic map PNGs
- data/test_data/               — JSON metadata for test
- data/test_images/, data/test_lidar/, data/test_maps/ — test data
- data/sample_submission.csv    — expected submission format
- baseline.py              — reference baseline

COMMANDS:
- cat baseline.py            — read baseline
- head -3 train.csv          — preview annotations
- python -c "import numpy as np; pc=np.fromfile('train_lidar/FILE.bin',dtype=np.float32).reshape(-1,5); print(pc.shape)" — inspect LiDAR
- cat << 'ENDOFFILE' > train_and_predict.py ... ENDOFFILE — write script
- python train_and_predict.py — run training + prediction
- validate                   — score submission.csv (mAP)
- ls, head, wc -l, sed -i    — file utilities

OUTPUT: submission.csv with columns: Id, PredictionString
        PredictionString = space-delimited: confidence center_x center_y center_z width length height yaw class_name
        Multiple objects separated by spaces. Empty string = no detections.

AVAILABLE PACKAGES:
- torch, torchvision (GPU available — use CUDA)
- numpy, pandas, scipy, sklearn
- cv2 (OpenCV)

LiDAR FORMAT:
- Binary .bin files: np.fromfile(path, dtype=np.float32).reshape(-1, 5)
- 5 channels: x, y, z, intensity, ring_index
- Coordinate system: x=forward, y=left, z=up

OBJECT CLASSES: animal, bicycle, bus, car, emergency_vehicle, motorcycle, other_vehicle, pedestrian, truck

KEY STRATEGIES:
1. PointPillars: discretize x-y plane into pillars → extract features → 2D detection head
2. BEV approach: project points to bird's eye view image → 2D object detection
3. Clustering: DBSCAN on LiDAR points → fit oriented bounding boxes per cluster
4. Start with car-only detection, then extend to all classes

CRITICAL RULES:
1. ONE command per response
2. Read baseline.py before writing new code
3. Always run python train_and_predict.py BEFORE validate
4. submission.csv MUST have columns: Id, PredictionString
5. PredictionString: 9 tokens per object (confidence cx cy cz w l h yaw class_name)
6. Use GPU (torch.cuda) for training if using neural approaches
7. Map sample tokens from test_data/sample.json to match sample_submission.csv Ids

MANDATORY WORKFLOW:
1. cat baseline.py   (understand data + format)
2. Write train_and_predict.py
3. python train_and_predict.py
4. validate""",
        use_generic_conda=True,
        needs_gpu=True,
        step_timeout=1800.0,
        task_type="vision",
    ),
}


def _filter_torch_warning_spam(obs: str) -> str:
    """Drop torch._dynamo / _inductor warning lines that drown out useful training output.

    A single failed dynamo compile emits ~50 lines of `[rank0]:W ... _dynamo/convert_frame.py:1125]`
    warnings. Multiple failed compile sites produce hundreds. With observation truncation
    (head 2000 + tail 6000), the actual stdout (step counts, val_loss, save messages) gets
    completely buried. This filter drops the noise so the agent can see what training did.
    """
    if not obs or ("_dynamo" not in obs and "WON'T CONVERT" not in obs):
        return obs
    kept = []
    dropped = 0
    for line in obs.splitlines():
        if "_dynamo/" in line or "WON'T CONVERT" in line or "_inductor/lowering" in line:
            dropped += 1
            continue
        kept.append(line)
    if dropped > 0:
        kept.append(f"[filtered {dropped} torch._dynamo/inductor warning lines]")
    return "\n".join(kept)


def get_task_profile(task_config: str) -> TaskProfile:
    """Auto-detect task from config path and return its profile."""
    # Validate config file exists (resolve relative to MLGYM_PATH)
    # MLGym's EnvironmentArguments prepends CONFIG_DIR (configs/), so
    # "tasks/X.yaml" is valid as a task_config_path even though the file
    # lives at configs/tasks/X.yaml on disk.
    config_path = Path(task_config)
    if not config_path.is_absolute():
        config_path = MLGYM_PATH / task_config
    if not config_path.exists():
        # Also try under configs/ (MLGym convention)
        alt = MLGYM_PATH / "configs" / task_config
        if alt.exists():
            config_path = alt
        else:
            parent = (MLGYM_PATH / "configs" / task_config).parent
            if parent.exists():
                available = sorted(p.stem for p in parent.glob("*.yaml"))
                print(f"ERROR: Task config '{task_config}' not found.")
                print(f"  Available configs: {', '.join(available)}")
            raise FileNotFoundError(f"Task config not found: {config_path}")

    # Sort by key length descending to match longest (most specific) key first
    for key in sorted(TASK_PROFILES, key=len, reverse=True):
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
    # Environment feedback fields (set after execution)
    execution_status: str = ""  # "success" | "training_failed" | "no_validate" | "env_error"
    error_type: str = ""        # Python exception class, e.g. "FileNotFoundError"


def classify_execution(action_log: list[dict], score) -> tuple[str, str]:
    """Classify what happened during an executor run.

    Returns (execution_status, error_type):
      - execution_status: "success" | "training_failed" | "no_validate" | "env_error"
      - error_type: Python exception class name if any, else ""

    "training_failed" means training crashed with a Traceback but validate still
    returned a score (likely from a pre-existing baseline submission file).
    """
    import re as _re

    error_type = ""
    has_training_error = False

    for action in action_log:
        cmd = action.get("action", "")
        obs = action.get("observation", "") or ""

        # Only look for errors in python execution steps
        is_python = cmd.strip().startswith("python") or "python src/" in cmd
        if is_python and "Traceback (most recent call last)" in obs:
            has_training_error = True
            # Extract the last error class name from the traceback
            for line in reversed(obs.strip().split("\n")):
                line = line.strip()
                m = _re.match(r'^([A-Za-z_.]+(?:Error|Exception|Interrupt)):', line)
                if m:
                    error_type = m.group(1).split(".")[-1]
                    break

        # CUDA OOM can appear outside tracebacks
        if "CUDA out of memory" in obs or "OutOfMemoryError" in obs:
            has_training_error = True
            if not error_type:
                error_type = "CUDAOutOfMemory"

    if score is None:
        # Check if a forced validate was attempted
        last_action = action_log[-1].get("action", "") if action_log else ""
        if "validate (forced)" in last_action:
            status = "no_submission_produced"
        else:
            status = "no_validate_called"
    elif has_training_error:
        # Got a score despite training errors → likely baseline fallback file was scored
        status = "training_failed"
    else:
        status = "success"

    return status, error_type


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
        self._eval_readonly_paths: list[tuple[str, Path]] = []  # (rel_path, host_src)

    def create(self):
        original_cwd = os.getcwd()
        try:
            os.chdir(MLGYM_PATH)
            env_args = EnvironmentArguments(
                image_name=self.image_name,
                max_steps=99999,
                task_config_path=self.task_config,
                container_type=os.environ.get("MLGYM_CONTAINER_TYPE", "apptainer"),
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
                # Only install if missing — skip if node has slow/no internet to avoid hanging
                self.env.communicate(
                    "python -c 'import xgboost' 2>/dev/null || "
                    "timeout 30 pip install -q xgboost lightgbm catboost > /dev/null 2>&1 || true",
                    timeout_duration=60,
                )
                check = self.env.communicate(
                    "python -c 'import torch, sklearn, xgboost; "
                    "print(f\"torch={torch.__version__}, sklearn={sklearn.__version__}, xgb={xgboost.__version__}\")' 2>&1"
                )
                print(f"  Package check: {check.strip()}")
            else:
                # RL tasks use mlgym_rl conda env baked into the container.
                # Explicitly set PATH so python resolves to the conda env,
                # even after container process restarts.
                self.env.communicate(
                    "export PATH=/home/agent/miniconda3/envs/mlgym_rl/bin:$PATH",
                    timeout_duration=10,
                )
                print("RL task — activated mlgym_rl conda env")

            # cd into workspace
            self.env.communicate("cd /home/agent/workspace")

            # Back up evaluation files so we can restore them before each validate call.
            # This prevents agents from gaming scores by rewriting evaluate.py.
            self._backup_eval_files()

            print(f"Container created. Baseline scores: {self.baseline_scores_dict}")
            print(f"Primary baseline ({self.task_profile.primary_metric if self.task_profile else '?'}): {self.baseline_score:.4f}")
        finally:
            os.chdir(original_cwd)

    def _backup_eval_files(self):
        """Record host-side source paths for evaluation files so we can overwrite before validate.

        Instead of copying inside the container (vulnerable to symlink attacks), we store the
        original file path on the host and write directly to the workspace bind-mount directory.
        """
        if not self.env:
            return
        try:
            task_args = None
            if hasattr(self.env, 'task') and hasattr(self.env.task, 'args'):
                task_args = self.env.task.args
            if task_args is None or not getattr(task_args, 'evaluation_read_only', False):
                return
            paths = getattr(task_args, 'evaluation_paths', [])
            if not paths:
                return
            starter_code = getattr(task_args, 'starter_code', []) or []
            for rel_path in paths:
                host_src = None
                for sc in starter_code:
                    if str(sc).endswith(str(rel_path)):
                        candidate = MLGYM_PATH / sc
                        if candidate.exists():
                            host_src = candidate
                            break
                if host_src is None:
                    # Fallback: look for the file in the MLGym data directory
                    task_name = getattr(task_args, 'task_name', '')
                    candidate = MLGYM_PATH / "data" / task_name / rel_path
                    if candidate.exists():
                        host_src = candidate
                if host_src is not None:
                    self._eval_readonly_paths.append((str(rel_path), host_src))
                    print(f"  [eval-guard] registered host source for {rel_path}: {host_src}")
                else:
                    print(f"  [eval-guard] WARNING: could not find host source for {rel_path}")
        except Exception as e:
            print(f"  [eval-guard] backup failed (non-fatal): {e}")

    def _restore_eval_files(self):
        """Overwrite evaluation files from the host, bypassing any container-side symlink tricks."""
        if not self._eval_readonly_paths or not self.env:
            return
        try:
            workspace_host_dir = Path(self.env.container_obj.workspace_host_dir)
        except AttributeError:
            return
        for rel_path, host_src in self._eval_readonly_paths:
            dst = workspace_host_dir / rel_path
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Remove symlink or existing file first so we write the real file
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                shutil.copy2(str(host_src), str(dst))
                print(f"  [eval-guard] restored {rel_path} from host")
            except Exception as e:
                print(f"  [eval-guard] restore failed for {rel_path}: {e}")

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
        # Restore evaluation files before validate to prevent gaming.
        if action.strip().lower() in ("validate", "submit"):
            self._restore_eval_files()
        obs, reward, done, info = self.env.step(action)
        obs = obs or "Action executed."
        obs = _filter_torch_warning_spam(obs)
        # If the container restarted (timeout/crash), shell functions are lost.
        # Reload commands, restore working directory, and re-activate conda env.
        if "RESTARTING PROCESS" in obs:
            self._load_commands()
            self.env.communicate("cd /home/agent/workspace")
            if self.task_profile and self.task_profile.use_generic_conda:
                self.env.communicate(
                    "export PATH=/home/agent/miniconda3/envs/mlgym_generic/bin:$PATH",
                    timeout_duration=10,
                )
        return obs, info

    def communicate(self, cmd: str, timeout: float = 30.0) -> str:
        return self.env.communicate(cmd, timeout_duration=timeout) or ""

    def save_snapshot(self, node_id: str) -> str:
        safe_id = node_id.replace("/", "_").replace(" ", "_")
        snap = f"/tmp/snap_{safe_id}.tar"
        # Exclude data/ directory from snapshots — it's immutable input data
        # and can be very large (e.g. 20 GB for Vesuvius).  Also exclude
        # common large cache dirs that don't need to be restored.
        self.communicate(
            f"cd /home/agent && tar cf {snap} --exclude='workspace/data' "
            f"--exclude='workspace/.cache' --exclude='workspace/__pycache__' "
            f"workspace",
            timeout=120.0,
        )
        return snap

    def restore_snapshot(self, snap_path: str):
        # Clear workspace CONTENTS without removing the directory itself.
        # Using rm -rf workspace would delete the apptainer bind-mount point,
        # causing tar to extract into the tmpfs overlay instead of the bound
        # host directory — leaving the agent with an empty visible workspace.
        # Preserve data/ since it was excluded from the snapshot.
        self.communicate(
            "find /home/agent/workspace -mindepth 1 -not -path '*/data/*' "
            "-not -path '*/data' -delete 2>/dev/null || true"
        )
        self.communicate(f"cd /home/agent && tar xf {snap_path}", timeout=120.0)
        self.communicate("cd /home/agent/workspace")
        # Re-activate conda env. After container process restarts (e.g. due to
        # timeout/broken-pipe), the new shell loses the conda env that was set
        # up during initial container creation. For RL tasks this means
        # 'python src/train.py' fails with 'No module named numpy/jax'.
        if self.task_profile and not self.task_profile.use_generic_conda:
            self.communicate(
                "export PATH=/home/agent/miniconda3/envs/mlgym_rl/bin:$PATH",
                timeout=10,
            )
        elif self.task_profile and self.task_profile.use_generic_conda:
            self.communicate(
                "export PATH=/home/agent/miniconda3/envs/mlgym_generic/bin:$PATH",
                timeout=10,
            )
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
        model_lower = model.lower()
        if "claude" in model_lower:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = base_url or "https://api.anthropic.com/v1/"
        elif "gemini" in model_lower:
            # Gemini OpenAI-compatible endpoint
            # Support per-job key via GEMINI_API_KEY_OVERRIDE env (set by SLURM script)
            api_key = (os.environ.get("GEMINI_API_KEY_OVERRIDE")
                       or os.environ.get("GEMINI_API_KEY")
                       or os.environ.get("GOOGLE_API_KEY", ""))
            base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "local")
        kwargs = {"api_key": api_key}
        # Skip explicit base_url for OpenAI (use default — avoids connection issues on some clusters)
        # Always set base_url for Anthropic, Gemini, and local vLLM
        if base_url and "openai.com" not in base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        is_reasoning = any(t in self.model for t in ("o1", "o3", "o4"))
        is_claude = "claude" in self.model.lower()
        token_key = "max_completion_tokens" if is_reasoning or "gpt-5" in self.model else "max_tokens"
        max_tokens = 16384 if is_reasoning else 8192
        # Claude: thinking budget is separate from output tokens — increase max.
        if self.thinking_budget > 0 and is_claude:
            max_tokens = max(max_tokens, self.thinking_budget + 8192)
        kwargs = {
            "model": self.model,
            "messages": messages,
            token_key: max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = temperature or self.temperature

        # o1/o3/o4 with thinking budget: cap total tokens (thinking + output).
        # 6144 output buffer is enough for any single ReAct action or script edit.
        if self.thinking_budget > 0 and is_reasoning and not is_claude:
            max_tokens = self.thinking_budget + 6144
            kwargs[token_key] = max_tokens

        # Claude: enable extended thinking via Anthropic extra_body
        if self.thinking_budget > 0 and is_claude:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}
            }
            kwargs.pop("temperature", None)

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"LLM failed after 3 attempts: {type(e).__name__}: {e}")

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
                 sampling_mode: str = "tail", time_budget: int = 0):
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
        self.time_budget = time_budget  # seconds, 0 = no limit
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
        stopped_by = "node_budget"
        time_expired = False
        for depth in range(1, self.max_depth + 1):
            if time_expired:
                break
            print(f"\n{'=' * 60}")
            print(f"TREE SEARCH - Depth {depth} ({len(frontier)} nodes to expand)")
            print("=" * 60)
            next_frontier = []
            for nid in frontier:
                # Check time budget before each branch expansion
                if self.time_budget > 0:
                    elapsed = time.time() - start
                    if elapsed >= self.time_budget:
                        print(f"\n--- Time budget reached ({elapsed:.0f}s >= {self.time_budget}s). Stopping. ---")
                        stopped_by = "time_budget"
                        time_expired = True
                        break
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
            "node_budget": self.branching_factor * self.max_depth,
            "time_budget_seconds": self.time_budget,
            "stopped_by": stopped_by,
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
    parser.add_argument("--time-budget", type=int, default=0,
                        help="Max seconds for search (0 = no limit). Stops when either node or time budget is exhausted.")
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Run without GPU (CPU only). Passes devices=['cpu'] to MLGym, "
                             "skipping the --gpus flag in the container command.")
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
    time_str = f"{args.time_budget}s" if args.time_budget > 0 else "unlimited"
    print(f"Max actions/node: {args.max_actions}, Time budget: {time_str}")
    print(f"Temperature: {args.temperature}")
    print(f"Verbalized sampling: {use_vs}" + (f" (mode: {args.sampling_mode})" if use_vs else ""))
    print(f"Primary metric: {task_profile.primary_metric} ({'higher' if task_profile.higher_is_better else 'lower'} is better)")
    print()

    env_gpu = "cpu" if args.no_gpu else args.env_gpu

    llm = LLMClient(args.vllm_url, args.model, args.temperature)
    container = ContainerManager(args.task_config, env_gpu, args.image_name,
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
        time_budget=args.time_budget,
    )

    try:
        result = search.run()
        print(f"Results saved to {args.output_dir}/result.json")
    finally:
        container.close()


if __name__ == "__main__":
    main()
