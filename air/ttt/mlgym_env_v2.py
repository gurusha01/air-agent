"""
MLGym Tree Search environment v2 — matches llm_guided_tree_search.py exactly.

The scientist gets the same prompts, context, and decision format as in
the evaluation setting (llm_guided_tree_search.py). This ensures training
and eval use identical interfaces.

Key differences from mlgym_env.py (v1):
- Scientist sees task_details (file names, code structure, baseline info)
- Scientist outputs REASONING + STRATEGIES + DIRECTION (not just <proposal>)
- Scientist has memory across turns
- Reward based on best score in episode (discrete)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from air.tree_search import (
    ContainerManager,
    LLMClient,
    TaskProfile,
    get_task_profile,
    extract_command,
    MLGYM_PATH,
)
from air.ttt.mlgym_env import TASKS, execute_in_container

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scientist prompts (from llm_guided_tree_search.py, simplified for training)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior research scientist mentoring a junior coder.
You guide experiment design by proposing directions for the executor to implement.

The executor is a small language model that writes and runs code in a container.
It can read all source files. Focus on the IDEA, not the code.

WHAT THE EXECUTOR CAN DO WELL:
- Short, self-contained Python scripts (<100 lines)
- For ML tasks: sklearn/XGBoost/LightGBM pipelines, pandas preprocessing
- For RL tasks: modifying config files, hyperparameter tuning
- Hyperparameter changes when you spell out exact values

WHAT THE EXECUTOR CANNOT DO:
- PyTorch/TensorFlow custom models
- Complex multi-file code or algorithms >150 lines
- Debugging subtle errors

For each decision, follow this process:
1. DIAGNOSE: What worked and what failed? Why?
2. BRAINSTORM 3 strategies spanning safe/moderate/bold
3. CHOOSE one and explain why

Respond in this format:

REASONING:
[Your analysis of what worked, what failed, and why]

STRATEGIES:
1. [Safe incremental change] — [risk assessment]
2. [Moderate modification] — [risk assessment]
3. [Bold new direction] — [risk assessment]
CHOSEN: [number] because [reason]

DIRECTION:
[Clear instructions for the executor. Be specific about what to change and to what values.]

MEMORY:
[One sentence about what you learned from the results. Write NONE if first turn.]"""


INITIAL_PROMPT = """## Task

{task_description}

## Task Details (what the executor sees)

{task_details}

Metric: {metric_name} ({direction} is better)
Baseline score: {baseline_score:.4f}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining. Make your decision."""


TURN_PROMPT = """## Result

{result}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining. Make your decision."""


def parse_direction(text: str) -> str:
    """Extract the DIRECTION section from scientist output."""
    m = re.search(r'DIRECTION:\s*\n(.*?)(?:\n(?:MEMORY|EXECUTOR|MODE):|\Z)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: try <proposal> tags
    m = re.search(r'<proposal>(.*?)</proposal>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Last resort: return everything after CHOSEN
    m = re.search(r'CHOSEN:.*?\n(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()[:500]
    return text.strip()[:500]


def parse_memory(text: str) -> str:
    """Extract the MEMORY section from scientist output."""
    m = re.search(r'MEMORY:\s*\n(.*?)(?:\n[A-Z]+:|\Z)', text, re.DOTALL)
    if m:
        mem = m.group(1).strip()
        if mem.upper() != "NONE":
            return mem
    return ""


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _log_reward(scheme: str, reward: float, best, baseline, any_score, state, extra: dict | None = None):
    log_dir = Path("/scratch/jarnav/rollout_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    import time
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scheme": scheme,
        "reward": reward,
        "best_score": best,
        "baseline": baseline,
        "any_score": any_score,
        "tree": [(n.get("id"), n.get("score")) for n in state.get("tree", [])],
    }
    if extra:
        entry.update(extra)
    # Write to a scheme-specific file so parallel runs don't collide
    with open(log_dir / f"rewards_{scheme}.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    # Also append to combined rewards.jsonl for backwards-compat with old tools
    with open(log_dir / "rewards.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def discrete_reward_v2(parser, completion, answer, state, **kwargs) -> float:
    """Scheme C (original, v6): +1 any score above moving best_so_far, +0.2 unused (binary here),
    0 below baseline, -0.5 fault. Used by v6 config.
    NOTE: this is the v6-style binary version — kept for backwards compatibility."""
    best = state.get("best_score")
    baseline = state.get("baseline_score", 0)
    higher = state.get("higher_is_better", True)
    any_score = state.get("any_score_achieved", False)

    if not any_score:
        reward = -0.5
    elif higher:
        reward = 1.0 if best > baseline else 0.0
    else:
        reward = 1.0 if best < baseline else 0.0

    _log_reward("v6_moving_best", reward, best, baseline, any_score, state)
    return reward


# ---------------------------------------------------------------------------
# Scheme A (v7): fixed absolute tiers, no moving baseline.
#   -0.5  executor fault (no score)
#    0.0  score < baseline
#   +0.2  baseline <= score <= 0.9
#   +1.0  score > 0.9
# ---------------------------------------------------------------------------
FIXED_TIER_HIGH = 0.9  # absolute threshold for +1 reward on FashionMNIST

def discrete_reward_v7_fixed_tier(parser, completion, answer, state, **kwargs) -> float:
    best = state.get("best_score")
    baseline = state.get("baseline_score", 0)
    higher = state.get("higher_is_better", True)
    any_score = state.get("any_score_achieved", False)

    if not any_score or best is None:
        reward = -0.5
    elif higher:
        if best > FIXED_TIER_HIGH:
            reward = 1.0
        elif best >= baseline:
            reward = 0.2
        else:
            reward = 0.0
    else:
        # lower-is-better: flip comparisons. "high tier" means below (1 - FIXED_TIER_HIGH).
        low_thr = 1.0 - FIXED_TIER_HIGH
        if best < low_thr:
            reward = 1.0
        elif best <= baseline:
            reward = 0.2
        else:
            reward = 0.0

    _log_reward("v7_fixed_tier", reward, best, baseline, any_score, state,
                extra={"fixed_tier_high": FIXED_TIER_HIGH})
    return reward


# ---------------------------------------------------------------------------
# Scheme B (v8): percentile-moving best.
#   Keep a rolling deque of the last N scores; "best" = 70th percentile.
#   -0.5  executor fault
#    0.0  score <= baseline (but ran fine)
#   +0.2  baseline < score <= p70(recent)
#   +1.0  score > p70(recent)
# ---------------------------------------------------------------------------
PERCENTILE_WINDOW = 500
PERCENTILE_Q = 70
_recent_scores: deque[float] = deque(maxlen=PERCENTILE_WINDOW)

def _percentile_best(baseline: float, higher: bool) -> float:
    if len(_recent_scores) < 8:
        return baseline  # not enough history; fall back to baseline as the +1 threshold
    import numpy as np
    q = PERCENTILE_Q if higher else (100 - PERCENTILE_Q)
    return float(np.percentile(list(_recent_scores), q))

def discrete_reward_v8_percentile(parser, completion, answer, state, **kwargs) -> float:
    best = state.get("best_score")
    baseline = state.get("baseline_score", 0)
    higher = state.get("higher_is_better", True)
    any_score = state.get("any_score_achieved", False)

    pct_best = _percentile_best(baseline, higher)

    if not any_score or best is None:
        reward = -0.5
    elif higher:
        if best > pct_best:
            reward = 1.0
        elif best > baseline:
            reward = 0.2
        else:
            reward = 0.0
    else:
        if best < pct_best:
            reward = 1.0
        elif best < baseline:
            reward = 0.2
        else:
            reward = 0.0

    # Update rolling window ONLY with valid numeric scores, after computing reward
    # so the current score doesn't self-reference.
    if any_score and best is not None:
        _recent_scores.append(float(best))

    _log_reward("v8_percentile", reward, best, baseline, any_score, state,
                extra={"pct_best": pct_best, "window_size": len(_recent_scores)})
    return reward


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MLGymTreeEnvV2(vf.MultiTurnEnv):
    """Training environment that matches llm_guided_tree_search.py prompts."""

    def __init__(
        self,
        task: str = "titanic",
        task_name: str | None = None,
        node_budget: int = 5,
        executor_url: str = "http://localhost:9001/v1",
        executor_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_actions: int = 20,
        num_train_examples: int = 200,
        num_eval_examples: int = 20,
        env_gpu: str = "cpu",
        system_prompt: str | None = None,
        parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        reward_scheme: str = "v6_moving_best",
        **kwargs,
    ):
        self.task_name = task_name or task
        logger.info(f"MLGymTreeEnvV2: task={self.task_name}")
        self.task_cfg = TASKS[self.task_name]
        self.task_profile = get_task_profile(self.task_cfg["task_config"])
        self.node_budget = node_budget
        self.executor_url = executor_url
        self.executor_model = executor_model
        self.max_actions = max_actions
        self.env_gpu = env_gpu

        # Get baseline score from task YAML
        tp = self.task_profile
        self._baseline_score = 0.0
        try:
            import yaml
            yaml_path = Path(__file__).resolve().parents[3] / "MLGym" / "configs" / self.task_cfg["task_config"]
            with open(yaml_path) as f:
                task_yaml = yaml.safe_load(f)
            scores = task_yaml.get("baseline_scores", [])
            if scores and isinstance(scores[0], dict):
                self._baseline_score = scores[0].get(tp.primary_metric, 0.0)
            logger.info(f"Loaded baseline score from YAML: {self._baseline_score}")
        except Exception as e:
            logger.warning(f"Could not load baseline from YAML: {e}")

        # Get task details (what the executor sees)
        self.task_details = tp.root_task_desc.format(
            baseline_score=self._baseline_score,
            data_head="(code files available in workspace)",
        )

        system_prompt = system_prompt or SYSTEM_PROMPT

        # Parser extracts DIRECTION from freeform text
        parser = parser or vf.XMLParser(fields=["proposal"], answer_field="proposal")

        # Select reward function based on reward_scheme
        reward_fn_map = {
            "v6_moving_best": discrete_reward_v2,
            "v7_fixed_tier": discrete_reward_v7_fixed_tier,
            "v8_percentile": discrete_reward_v8_percentile,
        }
        if reward_scheme not in reward_fn_map:
            raise ValueError(f"Unknown reward_scheme={reward_scheme}. Options: {list(reward_fn_map)}")
        self.reward_scheme = reward_scheme
        logger.info(f"Using reward scheme: {reward_scheme}")

        rubric = rubric or vf.Rubric(parser=parser)
        rubric.add_reward_func(reward_fn_map[reward_scheme], weight=1.0)

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
        rows = []
        for _ in range(num_examples):
            rows.append({
                "question": self._format_initial_prompt(),
                "answer": "",
            })
        return Dataset.from_list(rows)

    def _format_initial_prompt(self) -> str:
        tp = self.task_profile
        baseline = self._baseline_score
        return INITIAL_PROMPT.format(
            task_description=tp.name,
            task_details=self.task_details,
            metric_name=tp.primary_metric,
            direction="higher" if tp.higher_is_better else "lower",
            baseline_score=baseline,
            tree_view="  root: score={:.4f} (baseline, no experiment yet)".format(baseline),
            memory="No experiments run yet.",
            budget_left=self.node_budget,
        )

    def _format_tree(self, tree: list[dict]) -> str:
        lines = []
        for node in tree:
            score_str = f"{node['score']:.4f}" if node.get("score") is not None else "FAILED"
            strategy = node.get("strategy", "baseline")[:120]
            lines.append(f"  {node['id']}: score={score_str} | {strategy}")
        best = max((n["score"] for n in tree if n.get("score") is not None), default=0)
        lines.append(f"\nBest score so far: {best:.4f}")
        return "\n".join(lines)

    async def setup_state(self, state: vf.State) -> vf.State:
        tp = self.task_profile
        baseline = self._baseline_score
        state["tree"] = [{
            "id": "root",
            "depth": 0,
            "score": baseline,
            "strategy": "Baseline (no experiment)",
        }]
        state["baseline_score"] = baseline
        state["higher_is_better"] = tp.higher_is_better
        state["best_score"] = baseline
        state["node_counter"] = 0
        state["last_score"] = None
        state["memory_lines"] = []
        state["any_score_achieved"] = False
        state["executor_fault_count"] = 0
        return state

    def _log_rollout(self, state, direction, score, feedback, executor_fault, scientist_output):
        """Log every rollout to a JSONL file for analysis."""
        import time
        log_dir = Path("/scratch/jarnav/rollout_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        # Scheme-tagged log so v6/v7/v8 don't collide when running in parallel
        log_file = log_dir / f"{self.task_name}_{self.reward_scheme}_rollouts.jsonl"

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": self.task_name,
            "scheme": self.reward_scheme,
            "node_counter": state.get("node_counter", 0),
            "baseline_score": state.get("baseline_score", 0),
            "best_so_far": state.get("best_score", 0),
            "score": score,
            "executor_fault": executor_fault,
            "feedback": feedback[:300] if feedback else "",
            "direction": direction[:300],
            "scientist_reasoning": "",
            "scientist_strategies": "",
            "tree_size": len(state.get("tree", [])),
            "memory": state.get("memory_lines", [])[-1] if state.get("memory_lines") else "",
        }

        # Extract reasoning and strategies from scientist output
        reasoning_match = re.search(r'REASONING:\s*\n(.*?)(?:\nSTRATEGIES:|\Z)', scientist_output, re.DOTALL)
        if reasoning_match:
            entry["scientist_reasoning"] = reasoning_match.group(1).strip()[:300]
        strategies_match = re.search(r'STRATEGIES:\s*\n(.*?)(?:\nCHOSEN:|\nDIRECTION:|\Z)', scientist_output, re.DOTALL)
        if strategies_match:
            entry["scientist_strategies"] = strategies_match.group(1).strip()[:500]

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        # Extract direction from scientist's output
        last_msg = messages[-1] if messages else None
        raw_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        direction = parse_direction(raw_text)
        memory_update = parse_memory(raw_text)

        if not direction or len(direction.strip()) < 5:
            state["last_score"] = None
            state["executor_fault_count"] += 1
            tree_view = self._format_tree(state["tree"])
            memory = "\n".join(state["memory_lines"]) or "No observations yet."
            response = vf.UserMessage(
                content=TURN_PROMPT.format(
                    result="Invalid direction. Please provide a DIRECTION section.",
                    tree_view=tree_view,
                    memory=memory,
                    budget_left=self.node_budget - state["node_counter"],
                )
            )
            return [response]

        if memory_update:
            state["memory_lines"].append(memory_update)

        logger.info(f"Executing direction: {direction[:100]}...")

        # Execute in container
        max_retries = 2
        for attempt in range(max_retries):
            score, feedback, executor_fault = await asyncio.to_thread(
                execute_in_container,
                proposal=direction,
                task_profile=self.task_profile,
                task_config=self.task_cfg["task_config"],
                container_image=self.task_cfg["container_image"],
                executor_url=self.executor_url,
                executor_model=self.executor_model,
                max_actions=self.max_actions,
                env_gpu=self.env_gpu,
            )
            if not executor_fault or score is not None:
                break
            logger.info(f"Executor fault on attempt {attempt+1}, retrying...")

        # Log rollout details
        self._log_rollout(state, direction, score, feedback, executor_fault, raw_text)

        # Update state
        state["node_counter"] += 1
        node_id = f"node_{state['node_counter']}"
        state["tree"].append({
            "id": node_id,
            "depth": 1,
            "score": score,
            "strategy": direction[:200],
        })

        if executor_fault:
            state["executor_fault_count"] += 1

        state["last_score"] = score
        if score is not None:
            state["any_score_achieved"] = True
            higher = state["higher_is_better"]
            if (higher and score > state["best_score"]) or \
               (not higher and score < state["best_score"]):
                state["best_score"] = score
                logger.info(f"NEW BEST: {score:.4f}")

        # Format result
        if score is not None:
            result = f"Score: {score:.4f}. {feedback}"
        elif executor_fault:
            result = f"FAILED (executor error). {feedback}"
        else:
            result = f"FAILED. {feedback}"

        tree_view = self._format_tree(state["tree"])
        memory = "\n".join(state["memory_lines"]) or "No observations yet."

        response = vf.UserMessage(
            content=TURN_PROMPT.format(
                result=result,
                tree_view=tree_view,
                memory=memory,
                budget_left=self.node_budget - state["node_counter"],
            )
        )
        return [response]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def load_environment(env_args: dict | None = None, **kwargs) -> MLGymTreeEnvV2:
    env_args = env_args or {}
    return MLGymTreeEnvV2(**env_args, **kwargs)
