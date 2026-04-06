"""
MLGym Tree Search environment v3 — actual tree-structured RL.

Extends v2 by:
1. Asking the scientist to select a PARENT for each proposal (deepen vs explore).
2. Tracking parent_id and depth on every tree node.
3. Rendering tree_view hierarchically so the scientist sees real structure.
4. Setting `state["final_env_response"]` when node_counter reaches node_budget,
   which fixes the off-by-one in verifiers' rollout loop (previously, max_turns=K
   yielded only K-1 actually-executed children because env_response runs between
   turns). With this fix, exactly `node_budget` proposals get executed per episode.

Keeps v2's reward classes (v6/v7/v8/v9) unchanged — reward is still computed
from state["best_score"] = max over all tree scores, regardless of topology.

Registered as `mlgym_tree_env_v3` (separate package) so running v2 jobs are
untouched by this change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import verifiers as vf

from air.ttt.mlgym_env import execute_in_container
from air.ttt.mlgym_env_v2 import (
    MLGymTreeEnvV2,
    parse_memory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree-aware scientist prompts — match the eval-time prompt format from
# llm_guided_tree_search.py (simplified but keeps the STRATEGIES + PARENT
# + MODE structure so train and eval are consistent).
# ---------------------------------------------------------------------------

TREE_SYSTEM_PROMPT = """You are a senior ML research scientist. You guide experiment design — you do NOT write code. A separate executor writes and runs the code based on your directions.

Your job: look at the experiment tree, decide what to try next, and give a high-level direction.

IMPORTANT RULES:
- Do NOT write code. Do NOT output python scripts. Only output the structured format below.
- Each direction you give spawns a new node in the experiment tree.
- You must choose which existing node to build on (PARENT field).
- PARENT: root means start a fresh approach. PARENT: node_3 means refine node_3's approach.

## Output Format (follow EXACTLY every turn)

REASONING:
[1-3 sentences: what worked, what failed, what to try next]

STRATEGIES:
1. [idea] → PARENT: root — [why]
2. [idea] → PARENT: node_1 — [why]
3. [idea] → PARENT: root — [why]
CHOSEN: [1/2/3] because [reason]

DIRECTION:
[What the executor should try. Be specific about the approach, hyperparameters, architecture choices. Do NOT write code — the executor handles implementation.]

MODE: explore

MEMORY:
[One-sentence insight from results so far, or NONE if first turn.]

## Example output (first turn, no prior results)

REASONING:
No experiments yet. Start with a simple baseline to establish a reference point.

STRATEGIES:
1. Random Forest on flattened pixels → PARENT: root — simple, fast, establishes baseline
2. Logistic Regression on raw pixels → PARENT: root — even simpler baseline
3. Small CNN with 2 conv layers → PARENT: root — standard image approach
CHOSEN: 1 because Random Forest is reliable and fast for a first experiment

DIRECTION:
Train a Random Forest classifier with 200 trees on the flattened 28x28 pixel features (784 dimensions). Use all 60000 training samples. Predict on the test set and save submission.csv.

MODE: explore

MEMORY: NONE

## Example output (later turn, with prior results)

REASONING:
node_1 (Random Forest) scored 0.87. node_2 (Logistic Regression) scored 0.84. Tree-based methods work better on this data. node_3 (CNN) failed due to timeout. Should try gradient boosting which is stronger than RF.

STRATEGIES:
1. XGBoost with tuned hyperparameters → PARENT: root — stronger tree method, new approach
2. Random Forest with more trees and feature engineering → PARENT: node_1 — refine RF result
3. LightGBM with histogram binning → PARENT: root — fast gradient boosting alternative
CHOSEN: 2 because node_1 already works well and more trees + PCA might push it higher

DIRECTION:
Build on node_1's Random Forest approach: increase to 500 trees, add PCA to reduce to 100 components before training, and use max_depth=20. Keep using all training samples.

MODE: exploit

MEMORY: RF=0.87 beats LR=0.84; tree methods work well on flattened pixels. CNN timed out — avoid for now."""


TREE_INITIAL_PROMPT = """## Task

{task_description}

## Task Details (what the executor sees)

{task_details}

Metric: {metric_name} ({direction} is better)
Baseline score: {baseline_score:.4f}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining out of {total_budget} total.

IMPORTANT: You are NOT the executor. Do NOT write code. Do NOT output python scripts.
Instead, output your analysis and direction in this EXACT format:

REASONING:
[1-3 sentences analyzing the tree]

STRATEGIES:
1. [idea] → PARENT: root — [why]
2. [idea] → PARENT: root — [why]
3. [idea] → PARENT: root — [why]
CHOSEN: [number] because [reason]

DIRECTION:
[What approach to try — describe the idea, not the code]

MODE: explore

MEMORY: NONE"""


TREE_TURN_PROMPT = """## Result of Last Expansion

{result}

## Current Experiment Tree

{tree_view}

## Memory

{memory}

You have {budget_left} nodes remaining.

Do NOT write code. Respond with REASONING, STRATEGIES (with PARENT: for each), CHOSEN, DIRECTION, MODE, MEMORY."""


# ---------------------------------------------------------------------------
# Parsers (tree-aware)
# ---------------------------------------------------------------------------

def parse_direction_v3(text: str) -> str:
    """Extract DIRECTION section, stopping at MODE / MEMORY / EXECUTOR_GUIDANCE.

    Accepts both `DIRECTION:\\n<content>` and `DIRECTION: <content>` (inline).
    Captures everything up to the next section header or end of text.
    Falls back to using the whole text as a direction if the model doesn't
    follow the structured format (e.g. outputs raw code).
    """
    m = re.search(
        r"DIRECTION:\s*(.*?)(?=\n(?:MODE|MEMORY|EXECUTOR_GUIDANCE|REASONING|STRATEGIES):|\Z)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    # Fallback: from DIRECTION: to end of text
    m = re.search(r"DIRECTION:\s*(.*)\Z", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:800]
    # Last resort: if model produced no DIRECTION section at all (e.g. raw code),
    # treat the entire output as the direction so the episode doesn't waste a turn.
    if len(text.strip()) > 10:
        logger.warning("[tree v3] No DIRECTION section found, using full text as direction")
        return text.strip()[:800]
    return ""


def parse_parent(text: str, valid_ids: set[str]) -> str:
    """Extract parent_id from CHOSEN strategy's PARENT annotation.

    Pattern matches e.g. "1. Strategy description → PARENT: node_2 — reason"
    or "PARENT: root —". Falls back to 'root' if the chosen strategy's parent
    isn't a known node id.
    """
    chosen_match = re.search(r"CHOSEN:\s*(\d+)", text)
    strat_lines = re.findall(
        r"(\d+)\.\s.*?(?:→|->)\s*PARENT:\s*[\"']?([A-Za-z0-9_]+)[\"']?\s*(?:—|-|$)",
        text,
    )
    if chosen_match and strat_lines:
        chosen_num = int(chosen_match.group(1))
        for num_str, parent in strat_lines:
            if int(num_str) == chosen_num:
                pid = parent.strip().rstrip("—-").strip()
                if pid == "root" or pid in valid_ids:
                    return pid
                return "root"  # invalid → default
    # Fallback 1: any PARENT: line at all
    any_parent = re.search(r"PARENT:\s*[\"']?([A-Za-z0-9_]+)[\"']?", text)
    if any_parent:
        pid = any_parent.group(1)
        if pid == "root" or pid in valid_ids:
            return pid
    return "root"


# ---------------------------------------------------------------------------
# The tree-structured env
# ---------------------------------------------------------------------------

class MLGymTreeEnvV3(MLGymTreeEnvV2):
    """Branching tree version.

    Overrides:
      - _format_initial_prompt: uses TREE_INITIAL_PROMPT
      - _format_tree: hierarchical rendering with parent/child indentation
      - setup_state: adds parent_id/depth on root
      - env_response: parses PARENT, attaches children to it, sets
        final_env_response when budget is reached
      - _log_rollout (via new _log_rollout_tree): includes parent_id
    """

    def __init__(self, *args, **kwargs):
        # Force tree-aware system prompt even if caller didn't specify.
        kwargs["system_prompt"] = TREE_SYSTEM_PROMPT
        super().__init__(*args, **kwargs)

        # CRITICAL OFF-BY-ONE FIX: verifiers' MultiTurnEnv runs a loop like:
        #   while not is_completed(state):
        #       prompt = get_prompt_messages(state)  # env_response runs HERE
        #       response = model.generate(prompt)
        #       trajectory.append(response)
        # env_response executes the container and adds a tree node, but it
        # runs INSIDE get_prompt_messages (between turns). The loop's
        # max_turns_reached check uses len(trajectory) >= max_turns. With
        # max_turns = node_budget, only (node_budget - 1) env_responses
        # fire before the loop terminates, so only (node_budget - 1) nodes
        # get added. We need max_turns = node_budget + 1 so the trajectory
        # can reach one more step, giving env_response a chance to run
        # node_budget times and set final_env_response for clean termination.
        self.max_turns = self.node_budget + 1
        logger.info(f"[tree v3] max_turns overridden to {self.max_turns} "
                    f"(node_budget={self.node_budget} + 1 for K executions)")

        # Rebuild dataset with tree-aware initial prompt. The super().__init__
        # already built one using v2's INITIAL_PROMPT; we rebuild here.
        from datasets import Dataset
        n_train = len(self.dataset) if self.dataset is not None else 0
        if n_train > 0:
            self.dataset = Dataset.from_list(
                [{"question": self._format_initial_prompt(), "answer": ""}
                 for _ in range(n_train)]
            )
        if self.eval_dataset is not None and len(self.eval_dataset) > 0:
            self.eval_dataset = Dataset.from_list(
                [{"question": self._format_initial_prompt(), "answer": ""}
                 for _ in range(len(self.eval_dataset))]
            )

    def _format_initial_prompt(self) -> str:
        tp = self.task_profile
        baseline = self._baseline_score
        root_tree = [{"id": "root", "parent_id": None, "depth": 0,
                      "score": baseline, "strategy": "baseline (no experiment)"}]
        return TREE_INITIAL_PROMPT.format(
            task_description=tp.name,
            task_details=self.task_details,
            metric_name=tp.primary_metric,
            direction="higher" if tp.higher_is_better else "lower",
            baseline_score=baseline,
            tree_view=self._format_tree(root_tree),
            memory="No experiments run yet.",
            budget_left=self.node_budget,
            total_budget=self.node_budget,
        )

    def _format_tree(self, tree: list[dict]) -> str:
        """Hierarchical rendering using parent_id links."""
        if not tree:
            return "  (empty tree)"
        node_by_id = {n["id"]: n for n in tree}
        children_map: dict[str, list[str]] = {}
        for n in tree:
            pid = n.get("parent_id")
            if pid is not None:
                children_map.setdefault(pid, []).append(n["id"])

        lines: list[str] = []

        def _render(node_id: str, prefix: str, is_last: bool, is_root: bool):
            n = node_by_id.get(node_id)
            if n is None:
                return
            sc = n.get("score")
            sc_str = f"{sc:.4f}" if sc is not None else "FAILED"
            strat = (n.get("strategy") or "").replace("\n", " ")[:100]
            if is_root:
                lines.append(f"{node_id}: baseline={sc_str}")
                new_prefix = ""
            else:
                branch = "└─ " if is_last else "├─ "
                lines.append(f"{prefix}{branch}{node_id} [{sc_str}] {strat}")
                new_prefix = prefix + ("   " if is_last else "│  ")
            kids = children_map.get(node_id, [])
            for i, cid in enumerate(kids):
                _render(cid, new_prefix, i == len(kids) - 1, is_root=False)

        _render("root", "", True, is_root=True)
        best = max((n["score"] for n in tree
                    if n.get("score") is not None and n.get("id") != "root"),
                   default=None)
        if best is None:
            lines.append("\n(no scored children yet)")
        else:
            lines.append(f"\nBest child score: {best:.4f}")
        return "\n".join(lines)

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        # Ensure root carries parent_id/depth fields
        if state.get("tree"):
            state["tree"][0]["parent_id"] = None
            state["tree"][0]["depth"] = 0
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        last_msg = messages[-1] if messages else None
        raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        direction = parse_direction_v3(raw_text)
        memory_update = parse_memory(raw_text)
        valid_ids = {n["id"] for n in state.get("tree", [])}
        parent_id = parse_parent(raw_text, valid_ids)

        # Validate direction
        if not direction or len(direction.strip()) < 5:
            state["last_score"] = None
            state["executor_fault_count"] += 1
            tree_view = self._format_tree(state["tree"])
            memory = "\n".join(state.get("memory_lines", [])) or "No observations yet."
            response = vf.UserMessage(
                content=TREE_TURN_PROMPT.format(
                    result="Invalid direction. Please include DIRECTION: section.",
                    tree_view=tree_view,
                    memory=memory,
                    budget_left=self.node_budget - state["node_counter"],
                )
            )
            return [response]

        if memory_update:
            state.setdefault("memory_lines", []).append(memory_update)

        logger.info(
            f"[tree v3] Executing (parent={parent_id}, depth="
            f"{next((n.get('depth',0) for n in state['tree'] if n['id']==parent_id), 0) + 1}): "
            f"{direction[:80]}..."
        )

        # Execute in container (retry once on fault)
        max_retries = 2
        score = None
        feedback = ""
        executor_fault = False
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
            logger.info(f"[tree v3] Executor fault attempt {attempt+1}, retrying")

        # Persist rollout log (tree-aware)
        self._log_rollout_tree(state, direction, score, feedback, executor_fault,
                               raw_text, parent_id)

        # Attach new node to the chosen parent
        state["node_counter"] += 1
        node_id = f"node_{state['node_counter']}"
        parent_node = next(
            (n for n in state["tree"] if n["id"] == parent_id),
            state["tree"][0],  # fallback: root
        )
        new_depth = parent_node.get("depth", 0) + 1
        state["tree"].append({
            "id": node_id,
            "parent_id": parent_id,
            "depth": new_depth,
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
                logger.info(f"[tree v3] NEW BEST: {score:.4f}")

        # Build result message for the next scientist turn
        if score is not None:
            result = f"Score: {score:.4f}. {feedback[:300]}"
        elif executor_fault:
            result = f"FAILED (executor error). {feedback[:300]}"
        else:
            result = f"FAILED. {feedback[:300]}"

        tree_view = self._format_tree(state["tree"])
        memory = "\n".join(state.get("memory_lines", [])) or "No observations yet."
        response = vf.UserMessage(
            content=TREE_TURN_PROMPT.format(
                result=result,
                tree_view=tree_view,
                memory=memory,
                budget_left=self.node_budget - state["node_counter"],
            )
        )

        # THE CRITICAL TREE-FIX: signal terminal env response AFTER the final
        # execution. This causes verifiers' rollout loop to skip the next
        # model generation and terminate, so the last proposal *actually runs*
        # (otherwise max_turns would stop one turn before the last execution).
        if state["node_counter"] >= self.node_budget:
            state["final_env_response"] = [response]
            logger.info(f"[tree v3] Budget reached ({self.node_budget}), terminating episode")

        return [response]

    def _log_rollout_tree(self, state, direction, score, feedback, executor_fault,
                          scientist_output, parent_id):
        log_dir = Path("/scratch/jarnav/rollout_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.task_name}_{self.reward_scheme}_tree_rollouts.jsonl"
        parent_depth = next(
            (n.get("depth", 0) for n in state["tree"] if n["id"] == parent_id),
            0,
        )
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": self.task_name,
            "scheme": self.reward_scheme,
            "tree_mode": True,
            "node_counter": state.get("node_counter", 0),
            "parent_id": parent_id,
            "new_node_depth": parent_depth + 1,
            "baseline_score": state.get("baseline_score", 0),
            "best_so_far": state.get("best_score", 0),
            "score": score,
            "executor_fault": executor_fault,
            "feedback": (feedback or "")[:300],
            "direction": (direction or "")[:300],
            "tree_size": len(state.get("tree", [])),
        }
        # Extract reasoning + strategies from the raw scientist output for debugging
        reasoning_match = re.search(
            r"REASONING:\s*\n(.*?)(?:\nSTRATEGIES:|\Z)",
            scientist_output, re.DOTALL,
        )
        if reasoning_match:
            entry["scientist_reasoning"] = reasoning_match.group(1).strip()[:300]
        strategies_match = re.search(
            r"STRATEGIES:\s*\n(.*?)(?:\nCHOSEN:|\nDIRECTION:|\Z)",
            scientist_output, re.DOTALL,
        )
        if strategies_match:
            entry["scientist_strategies"] = strategies_match.group(1).strip()[:500]
        mode_match = re.search(r"MODE:\s*(explore|exploit)", scientist_output, re.IGNORECASE)
        if mode_match:
            entry["mode"] = mode_match.group(1).lower()
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[tree v3] rollout log write failed: {e}")


def load_environment(env_args: dict | None = None, **kwargs) -> MLGymTreeEnvV3:
    env_args = env_args or {}
    return MLGymTreeEnvV3(**env_args, **kwargs)
