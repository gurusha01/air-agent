"""
MLGym Tree Search environment for PRIME-RL / verifiers.

Multi-turn environment where:
- Each turn: scientist sees tree state → proposes experiment → executor runs it → score returned
- Episode = full tree search (node_budget nodes)
- Reward = real MLGym execution score at each node

Usage with PRIME-RL:
    This file defines a `load_environment()` function that returns a verifiers.MultiTurnEnv.
    Register it as a PRIME-RL environment and use it in rl.toml configs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import copy
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

# Add air-agent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from air.tree_search import (
    ContainerManager,
    LLMClient,
    TaskProfile,
    get_task_profile,
    extract_command,
    MLGYM_PATH,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASKS = {
    "titanic": {
        "task_config": "tasks/titanic.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "battleOfSexes": {
        "task_config": "tasks/battleOfSexes.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "regression": {
        "task_config": "tasks/regressionKaggleHousePrice.yaml",
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
    },
    "mountaincar": {
        "task_config": "tasks/rlMountainCarContinuous.yaml",
        "container_image": os.environ.get("MLGYM_RL_IMAGE", "/scratch/jarnav/mlgym_rl.sif"),
    },
}


# ---------------------------------------------------------------------------
# Single node execution (synchronous, runs in thread)
# ---------------------------------------------------------------------------

def execute_in_container(
    proposal: str,
    task_profile: TaskProfile,
    task_config: str,
    container_image: str,
    executor_url: str,
    executor_model: str,
    max_actions: int = 15,
) -> tuple[float | None, str]:
    """Execute one proposal in an MLGym container. Returns (score, feedback_text)."""
    container = None
    try:
        container = ContainerManager(
            task_config=task_config,
            env_gpu="cpu",
            image_name=container_image,
            task_profile=task_profile,
        )
        container.create()

        executor = LLMClient(
            base_url=executor_url,
            model=executor_model,
            temperature=0.9,
        )

        # Build initial messages for executor
        data_head = ""
        if task_profile.data_head_cmd:
            try:
                data_head = container.env.communicate(task_profile.data_head_cmd, timeout_duration=10)
            except Exception:
                pass

        task_desc = task_profile.root_task_desc.format(
            baseline_score=container.baseline_score,
            data_head=data_head,
        )

        messages = [
            {"role": "system", "content": task_profile.system_prompt},
            {"role": "user", "content": task_desc},
            {"role": "user", "content": f"Strategy to try: {proposal}\n\n{task_profile.branch_write_instruction}"},
        ]

        score = None
        feedback_parts = []

        for step in range(max_actions):
            try:
                raw = executor.chat(messages)
            except Exception as e:
                feedback_parts.append(f"Executor error: {e}")
                break

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "No command detected."})
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            obs, info = container.step(action)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(task_profile.primary_metric, list(score_data.values())[0])
                else:
                    score = score_data
                feedback_parts.append(f"Score: {score}")
                break
        else:
            obs, info = container.step("validate")
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(task_profile.primary_metric, list(score_data.values())[0])
                else:
                    score = score_data
                feedback_parts.append(f"Score (forced validate): {score}")
            else:
                feedback_parts.append("No score produced after max actions.")

        feedback = " | ".join(feedback_parts) if feedback_parts else f"Score: {score}"
        return score, feedback

    except Exception as e:
        return None, f"Execution failed: {e}"
    finally:
        if container and container.env:
            try:
                container.env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def node_reward(parser, completion, answer, state, **kwargs) -> float:
    """Reward based on the score achieved at this node."""
    score = state.get("last_score")
    baseline = state.get("baseline_score", 0)
    higher_is_better = state.get("higher_is_better", True)

    if score is None:
        return -0.5  # penalty for failed execution

    import math
    if higher_is_better:
        delta = (score - baseline) / max(abs(baseline), 1e-6)
    else:
        delta = (baseline - score) / max(abs(baseline), 1e-6)
    return math.tanh(delta)


def best_score_reward(parser, completion, answer, state, **kwargs) -> float:
    """Bonus reward if this node set a new best score."""
    return 0.5 if state.get("is_new_best", False) else 0.0


def format_reward(parser, completion, **kwargs) -> float:
    """Small reward for producing parseable output."""
    text = parser.parse_answer(completion) if hasattr(parser, 'parse_answer') else ""
    if text and len(text) > 10:
        return 0.1
    return 0.0


# ---------------------------------------------------------------------------
# MLGym Tree Search Environment
# ---------------------------------------------------------------------------

class MLGymTreeSearchEnv(vf.MultiTurnEnv):
    """Multi-turn environment for training a scientist via tree search.

    Each turn:
    1. Scientist sees current tree state (nodes, scores, strategies)
    2. Scientist proposes: which node to expand + what experiment to try
    3. Proposal is executed in a real MLGym container
    4. Score is returned as environment feedback
    5. Tree state is updated

    Episode ends after node_budget turns.
    """

    def __init__(
        self,
        task_name: str = "titanic",
        node_budget: int = 5,
        executor_url: str = "http://localhost:8001/v1",
        executor_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_actions: int = 15,
        num_train_examples: int = 200,
        num_eval_examples: int = 20,
        system_prompt: str | None = None,
        parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.task_cfg = TASKS[task_name]
        self.task_profile = get_task_profile(self.task_cfg["task_config"])
        self.node_budget = node_budget
        self.executor_url = executor_url
        self.executor_model = executor_model
        self.max_actions = max_actions

        system_prompt = system_prompt or (
            "You are an ML scientist guiding experiment design. "
            "You will see the current state of an experiment tree with nodes, their scores, and strategies. "
            "Your job is to propose the next experiment to try. "
            "Output your proposal in this format:\n"
            "<proposal>A specific, actionable experiment direction in 2-3 sentences.</proposal>"
        )

        parser = parser or vf.XMLParser(fields=["proposal"], answer_field="proposal")

        rubric = rubric or vf.Rubric(parser=parser)
        rubric.add_reward_func(node_reward, weight=1.0)
        rubric.add_reward_func(best_score_reward, weight=0.5)

        # Build dataset: each example is a task prompt with baseline info
        dataset = self._build_dataset(num_train_examples)
        eval_dataset = self._build_dataset(num_eval_examples) if num_eval_examples > 0 else None

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=node_budget,
            **kwargs,
        )

    def _build_dataset(self, num_examples: int) -> Dataset:
        baseline = self.task_profile.primary_metric
        rows = []
        for _ in range(num_examples):
            rows.append({
                "question": self._format_initial_prompt(),
                "answer": "",  # no fixed answer — reward is from execution
            })
        return Dataset.from_list(rows)

    def _format_initial_prompt(self) -> str:
        tp = self.task_profile
        return (
            f"Task: {tp.name}\n"
            f"Metric: {tp.primary_metric} ({'higher' if tp.higher_is_better else 'lower'} is better)\n"
            f"Baseline score: {self.task_profile.primary_metric}\n\n"
            f"Current experiment tree:\n"
            f"  root: score=baseline (no experiment run yet)\n\n"
            f"Propose the next experiment to try."
        )

    def _format_tree_state(self, tree: list[dict]) -> str:
        lines = ["Current experiment tree:"]
        for node in tree:
            indent = "  " * node.get("depth", 0)
            score_str = f"{node['score']:.4f}" if node.get("score") is not None else "FAILED"
            lines.append(f"{indent}{node['id']}: score={score_str} | strategy: {node.get('strategy', 'baseline')[:100]}")
        best = max((n["score"] for n in tree if n.get("score") is not None), default=0)
        lines.append(f"\nBest score so far: {best:.4f}")
        lines.append(f"Nodes remaining: {self.node_budget - len(tree) + 1}")
        lines.append("\nPropose the next experiment to try.")
        return "\n".join(lines)

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize tree search state for this rollout."""
        tp = self.task_profile
        state["tree"] = [{
            "id": "root",
            "depth": 0,
            "score": getattr(tp, '_baseline_score', None),
            "strategy": "Baseline (no experiment)",
        }]
        state["baseline_score"] = getattr(tp, '_baseline_score', 0)
        state["higher_is_better"] = tp.higher_is_better
        state["best_score"] = state["baseline_score"]
        state["node_counter"] = 0
        state["last_score"] = None
        state["is_new_best"] = False
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        """Execute the scientist's proposal and return the result."""
        # Parse the scientist's proposal
        proposal = self.parser.parse_answer(messages)
        if not proposal or len(proposal.strip()) < 5:
            # Failed to parse — return error feedback
            state["last_score"] = None
            state["is_new_best"] = False
            response = vf.UserMessage(
                content="Invalid proposal. Please output your experiment in <proposal>...</proposal> tags.\n\n"
                + self._format_tree_state(state["tree"])
            )
            return [response]

        logger.info(f"Executing proposal: {proposal[:100]}...")

        # Execute in container (blocking — run in thread)
        score, feedback = await asyncio.to_thread(
            execute_in_container,
            proposal=proposal,
            task_profile=self.task_profile,
            task_config=self.task_cfg["task_config"],
            container_image=self.task_cfg["container_image"],
            executor_url=self.executor_url,
            executor_model=self.executor_model,
            max_actions=self.max_actions,
        )

        # Update tree
        state["node_counter"] += 1
        node_id = f"node_{state['node_counter']}"
        state["tree"].append({
            "id": node_id,
            "depth": 1,
            "score": score,
            "strategy": proposal[:200],
        })

        # Update state for reward computation
        state["last_score"] = score
        state["is_new_best"] = False
        if score is not None:
            higher = state["higher_is_better"]
            if (higher and score > state["best_score"]) or (not higher and score < state["best_score"]):
                state["is_new_best"] = True
                state["best_score"] = score
                logger.info(f"NEW BEST: {score:.4f}")

        # Check if done
        if state["node_counter"] >= self.node_budget - 1:
            # Final turn
            score_str = f"{score:.4f}" if score is not None else "FAILED"
            state["final_env_response"] = [vf.UserMessage(
                content=f"Experiment result: {score_str}. {feedback}\n\nTree search complete. Best score: {state['best_score']:.4f}"
            )]
            return state["final_env_response"]

        # More turns remaining — show updated tree
        score_str = f"{score:.4f}" if score is not None else "FAILED"
        tree_state = self._format_tree_state(state["tree"])
        response = vf.UserMessage(
            content=f"Experiment result: {score_str}. {feedback}\n\n{tree_state}"
        )
        return [response]


# ---------------------------------------------------------------------------
# Entry point for PRIME-RL
# ---------------------------------------------------------------------------

def load_environment(
    task: str = "titanic",
    node_budget: int = 5,
    executor_url: str = "http://localhost:8001/v1",
    executor_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_actions: int = 15,
    num_train_examples: int = 200,
    num_eval_examples: int = 20,
    **kwargs,
) -> MLGymTreeSearchEnv:
    return MLGymTreeSearchEnv(
        task_name=task,
        node_budget=node_budget,
        executor_url=executor_url,
        executor_model=executor_model,
        max_actions=max_actions,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        **kwargs,
    )
