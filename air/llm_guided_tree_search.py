"""
Experiment 4: LLM-Guided Tree Search.

Replaces formula-based node selection (UCB, Softmax, Open-Ended) with an
LLM "scientist" that sees the full tree state, reasons about why things
worked or failed, and makes informed decisions about what to expand next.

Two-model setup:
    - Scientist (selector): larger API model (GPT-4o / Claude) that analyzes
      the tree and decides what to expand
    - Executor (worker): Qwen3-4B via vLLM that implements strategies in
      MLGym containers (unchanged from Exp 2/3)

Usage:
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/air-agent \
        python /home/ubuntu/MLScientist/air-agent/air/llm_guided_tree_search.py \
        --task-config tasks/titanic.yaml \
        --scientist-model gpt-4o \
        --executor-model Qwen/Qwen3-4B-Instruct-2507 \
        --executor-url http://localhost:8000/v1 \
        --node-budget 15
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env for API keys
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from air.tree_search import (
    TaskProfile,
    TASK_PROFILES,
    get_task_profile,
    ContainerManager,
    LLMClient,
    TreeNode,
    extract_command,
    MLGYM_PATH,
)


# ---------------------------------------------------------------------------
# ScientistDecision — parsed output from the scientist LLM
# ---------------------------------------------------------------------------

@dataclass
class ScientistDecision:
    action: str          # "expand" or "draft_from_root"
    node_id: str         # which node to expand (empty for draft_from_root)
    direction: str       # specific instruction for executor
    mode: str            # "explore" or "exploit"
    memory_update: str   # observation to remember (or empty)
    reasoning: str       # scientist's analysis (logged)
    executor_guidance: str = ""  # warnings/tips passed directly to executor


# ---------------------------------------------------------------------------
# Scientist Prompt
# ---------------------------------------------------------------------------

SCIENTIST_PROMPT_TURN1 = """You are a senior research scientist mentoring a junior coder.
Your job is to guide them to solve this task:

{task_description}

## Task Details (this is what the executor sees)

{task_details}

The metric is: {metric_name} ({direction} is better)
Baseline score (no model, just default): {baseline_score}

## How This Works

Your junior coder (the "executor") is a small 4B-parameter language model. Each time
you give a direction, the executor writes code from scratch in a container, runs it,
and validates. It has {max_actions} actions (shell commands) per attempt. Each attempt
creates one "node" in your search tree.

IMPORTANT: The executor already has ALL source files from the workspace pre-loaded in
its context. It can see the full code. Do NOT waste a node asking it to "read" or
"examine" files — it already knows the code. Every direction you give should be an
ACTIONABLE change (modify config, write code, tune hyperparameters), never exploration.

You have {budget_left} nodes remaining out of {total_budget} total.

## Understanding Your Executor

Your executor is a 4B model. Think of it as a junior developer who is good at
following cookbook recipes but bad at debugging novel code. Be specific and realistic:

WHAT IT CAN DO WELL (give these kinds of tasks):
- Short, self-contained Python scripts (<100 lines)
- For ML tasks: sklearn/XGBoost/LightGBM/CatBoost pipelines, simple pandas preprocessing
- For game theory tasks: simple strategy functions with clear logic
- For RL tasks: modifying config files, hyperparameter tuning
- Hyperparameter changes when you spell out exact values

WHAT IT CANNOT DO (never ask for these):
- PyTorch/TensorFlow custom models (will crash and burn all {max_actions} actions debugging)
- Complex multi-step logic or algorithms requiring >150 lines of code
- Multi-file code, imports from custom modules
- Debugging subtle errors (it rewrites the same broken code 5+ times)

## Your Search Tree

{tree_view}

## Your Accumulated Knowledge

{memory_section}

## Your Task Now

Before making a decision, you may inspect the actual code and commands that the
executor ran for any node. This lets you understand EXACTLY what was tried and why
it succeeded or failed.

Look at the tree above and decide which nodes you want to inspect. You can request
0 to 3 nodes. Request 0 if the tree is empty or you already understand what happened.

Respond in EXACTLY this format:

INSPECT: node_id_1, node_id_2
[OR]
INSPECT: NONE

Brief explanation of what you want to understand from inspecting these nodes."""


SCIENTIST_PROMPT_TURN2 = """Good. Now make your decision.

{code_inspection}

## Your Role as Mentor

You are COACHING the executor. Give it a direction — you decide the right level
of specificity. The executor can read all source files and figure out implementation
details on its own. Focus on the IDEA, not the code.

## Decision Process

Look at the tree and decide what to do next.

Step 1. DIAGNOSE: What worked and what failed? Look at scores and errors.

Step 2. DECIDE: Based on the tree, choose one of two modes:
   A. DEEPEN an existing direction — if you see a promising branch that hasn't
      been fully explored and has potential for improvement, propose the next
      idea to try along that direction. Expand from the relevant node.
   B. EXPLORE something brand new — if based on your learnings so far you want
      to try a fundamentally different approach, start a new branch from root.

Step 3. BRAINSTORM 3 DIVERSE STRATEGIES: Imagine a probability distribution over
   ALL possible strategies. Sample 3 such that each has probability < 0.2 — this
   forces you beyond the obvious first ideas into less common approaches. Each
   strategy must be fundamentally different from the others.

Step 4. CHOOSE: Pick ONE strategy by sampling from your 3 candidates with roughly
   equal probability (do NOT always pick the "safest" one). Consider:
   - Has something similar already been tried? Don't repeat what failed.
   - Can the executor realistically implement this?
   - Budget awareness: {budget_left} nodes left. With >=5, prefer exploring.
     With <=2, prefer refining the best working approach.

## Your Output

Respond in EXACTLY this format:

REASONING:
[Your analysis: what worked, what failed and why. Identify which DIMENSIONS
of the solution space have been explored vs unexplored.]

STRATEGIES:
1. [Strategy] → PARENT: [node_id or "root"] — [why this parent / risk assessment]
2. [Strategy] → PARENT: [node_id or "root"] — [why this parent / risk assessment]
3. [Strategy] → PARENT: [node_id or "root"] — [why this parent / risk assessment]
CHOSEN: [number] because [reason — consider executor capability and diversity]

DIRECTION:
[Instructions for the executor for the CHOSEN strategy.
The executor can read all source files — focus on the idea and target values,
not code-level implementation details.]

EXECUTOR_GUIDANCE:
[Warnings and tips for the executor based on what you learned from the tree.
E.g., "Do NOT use get_dummies — it causes memory errors on this dataset.
Use LabelEncoder instead." Write NONE if no specific warnings.]

MODE: explore
[OR]
MODE: exploit

MEMORY:
[One sentence about what you LEARNED. Must include evidence (what was tried,
what score) and an insight. Do NOT repeat anything already in your memory.
GOOD: "CatBoost (0.91) and LightGBM (0.90) both plateau — try feature engineering next."
BAD: "CatBoost works well." (repeats known info, no new insight)
Write NONE if no genuinely new insight.]"""


# ---------------------------------------------------------------------------
# LLM-Guided Tree Search
# ---------------------------------------------------------------------------

class LLMGuidedTreeSearch:
    """Tree search where an LLM scientist replaces formula-based selection."""

    def __init__(
        self,
        scientist: LLMClient,
        executor: LLMClient,
        container: ContainerManager,
        task_profile: TaskProfile,
        node_budget: int = 12,
        initial_breadth: int = 3,
        max_actions: int = 15,
        output_dir: str = "outputs/llm_guided_search",
        verbose: bool = False,
    ):
        self.scientist = scientist
        self.executor = executor
        self.container = container
        self.task = task_profile
        self.node_budget = node_budget
        self.initial_breadth = initial_breadth
        self.max_actions = max_actions
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.nodes: dict[str, TreeNode] = {}
        self.memory: list[str] = []
        self._child_counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "nodes").mkdir(exist_ok=True)

        # ---- Phase 1: Create baseline root ----
        print("\n" + "=" * 60)
        print("LLM-GUIDED TREE SEARCH - Phase 1: Root (baseline)")
        print("=" * 60)
        root = self._execute_root()
        self.nodes[root.node_id] = root
        self._save_node(root)

        # ---- Phase 2: Scientist-guided loop (all budget) ----
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Scientist-guided search ({self.node_budget} expansions)")
        print("=" * 60)

        for step in range(self.node_budget):
            print(f"\n--- Scientist step {step + 1}/{self.node_budget} ---")
            budget_left = self.node_budget - step

            # Ask scientist what to do
            decision = self._scientist_decide(budget_left)

            # Update memory (cap at 5 entries to prevent context blowup)
            if decision.memory_update and decision.memory_update.upper() != "NONE":
                self.memory.append(decision.memory_update)
                if len(self.memory) > 5:
                    self.memory = self.memory[-5:]
                print(f"  Memory updated: {decision.memory_update[:80]}")

            # Execute the decision
            if decision.action == "draft_from_root":
                parent_id = "root"
            else:
                # Validate node_id exists
                parent_id = decision.node_id
                if parent_id not in self.nodes:
                    print(f"  WARNING: Scientist chose non-existent node '{parent_id}', falling back to root")
                    parent_id = "root"

            print(f"  -> Expanding {parent_id} (mode={decision.mode})")
            print(f"  -> Direction: {decision.direction[:100]}")

            child = self._expand_one(
                parent_id,
                mode=decision.mode,
                direction=decision.direction,
                executor_guidance=decision.executor_guidance,
            )

        # ---- Results ----
        return self._compile_results(start)

    # ------------------------------------------------------------------
    # Root node
    # ------------------------------------------------------------------

    def _read_workspace_files(self) -> str:
        """Read all source files from the workspace and return as a string."""
        file_list = self.container.communicate(
            "find /home/agent/workspace -type f "
            "\\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' "
            "-o -name '*.json' -o -name '*.cfg' -o -name '*.txt' "
            "-o -name '*.sh' \\) "
            "! -path '*/checkpoints/*' ! -path '*/__pycache__/*' "
            "| sort"
        ).strip()
        if not file_list:
            return ""

        parts = []
        for fpath in file_list.split("\n"):
            fpath = fpath.strip()
            if not fpath:
                continue
            content = self.container.communicate(f"cat {fpath}")
            # Skip very large files (>5000 chars)
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            rel = fpath.replace("/home/agent/workspace/", "")
            parts.append(f"=== {rel} ===\n{content}")

        return "\n\n".join(parts)

    def _execute_root(self) -> TreeNode:
        data_head = ""
        if self.task.data_head_cmd:
            data_head = self.container.communicate(self.task.data_head_cmd)

        task_desc = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head=data_head,
        )

        # Read all workspace source files and inject into context
        workspace_files = self._read_workspace_files()
        if workspace_files:
            file_context = (
                "Here are the current source files in your workspace. "
                "You do NOT need to read them again — they are already provided:\n\n"
                f"{workspace_files}"
            )
            task_desc = f"{task_desc}\n\n{file_context}"

        messages = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        snap = self.container.save_snapshot("root")
        baseline = self.container.baseline_score
        print(f"  [root] Baseline (score={baseline:.4f})")

        return TreeNode(
            node_id="root", parent_id=None, depth=0,
            strategy="Baseline (no model execution)",
            score=baseline, actions=[],
            conversation_history=messages,
            snapshot_path=snap,
        )

    # ------------------------------------------------------------------
    # Tree view for scientist
    # ------------------------------------------------------------------

    def _build_tree_view(self) -> str:
        """Build a compact structured view of all nodes for the scientist."""
        lines = []

        def _node_line(nid: str, indent: int = 0):
            n = self.nodes[nid]
            prefix = "  " * indent

            # Score
            if n.score is not None:
                score_str = f"{n.score:.4f}"
            else:
                score_str = "FAILED"

            # Strategy summary
            strategy = n.strategy[:120] if n.strategy else "N/A"

            # Error diagnosis — give the scientist enough to coach the executor
            error_str = ""
            if n.score is None and n.actions:
                # Count how many actions were errors vs productive
                error_actions = [
                    a for a in n.actions
                    if "Traceback" in a.get("observation", "")
                    or "Error" in a.get("observation", "")
                ]
                # Show the LAST error (most diagnostic)
                if error_actions:
                    last_err = error_actions[-1]["observation"]
                    # Extract just the error type and message
                    err_lines = last_err.strip().split("\n")
                    err_msg = err_lines[-1][:120] if err_lines else last_err[:120]
                    error_str = (
                        f"\n{prefix}  FAILURE: {len(error_actions)}/{len(n.actions)} actions hit errors. "
                        f"Last error: {err_msg}"
                    )
                elif len(n.actions) >= self.max_actions:
                    error_str = f"\n{prefix}  FAILURE: Ran out of actions ({len(n.actions)}) without validating"
                else:
                    error_str = f"\n{prefix}  FAILURE: {n.error[:120]}" if n.error else ""
            elif n.error:
                error_str = f"\n{prefix}  Error: {n.error[:120]}"

            # Parent comparison
            parent_note = ""
            if n.parent_id and n.parent_id in self.nodes and n.score is not None:
                parent = self.nodes[n.parent_id]
                if parent.score is not None:
                    diff = n.score - parent.score
                    if self.task.higher_is_better:
                        parent_note = f" ({'better' if diff > 0 else 'worse'} than parent by {abs(diff):.4f})"
                    else:
                        parent_note = f" ({'better' if diff < 0 else 'worse'} than parent by {abs(diff):.4f})"

            lines.append(
                f"{prefix}Node {nid} [{strategy}]\n"
                f"{prefix}  Score: {score_str} | Actions: {len(n.actions)} | "
                f"Children: {len(n.children)}{parent_note}{error_str}"
            )

            for cid in n.children:
                if cid in self.nodes:
                    _node_line(cid, indent + 1)

        if "root" in self.nodes:
            _node_line("root")
        else:
            lines.append("(empty tree)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scientist decision
    # ------------------------------------------------------------------

    def _format_node_code(self, node_id: str) -> str:
        """Format a node's executor actions for the scientist to inspect."""
        if node_id not in self.nodes:
            return f"Node {node_id} not found."

        node = self.nodes[node_id]
        if not node.actions:
            return f"Node {node_id}: No actions (baseline node)."

        lines = [f"=== Node {node_id} (score: {node.score}) ==="]
        for i, action in enumerate(node.actions):
            cmd = action.get("action", "")
            obs = action.get("observation", "")
            # Truncate very long observations (e.g. training logs)
            if len(obs) > 500:
                obs = obs[:500] + "\n... (truncated)"
            lines.append(f"--- Action {i} ---")
            lines.append(f"$ {cmd}")
            lines.append(obs)

        return "\n".join(lines)

    def _parse_inspect_response(self, text: str) -> list[str]:
        """Parse the INSPECT: line from turn 1 to get node IDs."""
        match = re.search(r"INSPECT:\s*(.+)", text)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.upper() == "NONE":
            return []
        # Split by comma and clean up
        node_ids = [nid.strip() for nid in raw.split(",") if nid.strip()]
        # Validate they exist and cap at 3
        valid = [nid for nid in node_ids if nid in self.nodes]
        return valid[:3]

    def _scientist_decide(self, budget_left: int) -> ScientistDecision:
        """Call the scientist LLM in two turns: inspect, then decide."""
        tree_view = self._build_tree_view()

        # Memory section
        if self.memory:
            memory_section = "\n".join(f"- {m}" for m in self.memory)
        else:
            memory_section = "(No accumulated knowledge yet — this is the first scientist step.)"

        # Task description
        task_desc = f"{self.task.name}"
        if self.task.task_type:
            task_desc += f" ({self.task.task_type})"

        # Build task details from what the executor sees
        task_details = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head="(data preview omitted)",
        )
        task_details = (
            f"EXECUTOR SYSTEM PROMPT:\n{self.task.system_prompt}\n\n"
            f"TASK DESCRIPTION:\n{task_details}"
        )

        # --- Turn 1: Show tree, ask what to inspect ---
        turn1_prompt = SCIENTIST_PROMPT_TURN1.format(
            task_description=task_desc,
            task_details=task_details,
            metric_name=self.task.primary_metric,
            direction="higher" if self.task.higher_is_better else "lower",
            baseline_score=f"{self.container.baseline_score:.4f}",
            max_actions=self.max_actions,
            budget_left=budget_left,
            total_budget=self.node_budget,
            tree_view=tree_view,
            memory_section=memory_section,
        )

        messages = [{"role": "user", "content": turn1_prompt}]

        try:
            turn1_resp = self.scientist.chat(messages, temperature=0.3)
        except Exception as e:
            print(f"  WARNING: Scientist turn 1 failed: {e}")
            turn1_resp = "INSPECT: NONE"

        # Parse which nodes to inspect
        inspect_ids = self._parse_inspect_response(turn1_resp)

        # --- Build code inspection content ---
        if inspect_ids:
            print(f"  Scientist inspecting: {inspect_ids}")
            code_parts = [self._format_node_code(nid) for nid in inspect_ids]
            code_inspection = (
                "Here is the code and output from the nodes you requested:\n\n"
                + "\n\n".join(code_parts)
            )
        else:
            code_inspection = "(No nodes inspected.)"

        # --- Turn 2: Make decision with code context ---
        turn2_prompt = SCIENTIST_PROMPT_TURN2.format(
            code_inspection=code_inspection,
            budget_left=budget_left,
            max_actions=self.max_actions,
        )

        messages.append({"role": "assistant", "content": turn1_resp})
        messages.append({"role": "user", "content": turn2_prompt})

        try:
            turn2_resp = self.scientist.chat(messages, temperature=0.3)
            decision = self._parse_scientist_response(turn2_resp)
        except Exception as e:
            print(f"  WARNING: Scientist turn 2 failed: {e}")
            decision = ScientistDecision(
                action="draft_from_root", node_id="root",
                direction="Try a robust sklearn pipeline with cross-validation",
                mode="explore", memory_update="",
                reasoning=f"Fallback due to scientist error: {e}",
            )

        # Log scientist reasoning
        print(f"\n  SCIENTIST REASONING:\n  {decision.reasoning[:300]}")
        if inspect_ids:
            print(f"  Inspected nodes: {inspect_ids}")
        self._save_scientist_log(
            decision, budget_left, tree_view,
            inspected_nodes=inspect_ids, turn1_response=turn1_resp,
        )

        return decision

    def _parse_scientist_response(self, text: str) -> ScientistDecision:
        """Parse the scientist's structured output."""
        # Extract REASONING
        reasoning_match = re.search(
            r"REASONING:\s*\n(.*?)(?=\nSTRATEGIES:|$)", text, re.DOTALL
        )
        if not reasoning_match:
            # Fallback: try old format
            reasoning_match = re.search(
                r"REASONING:\s*\n(.*?)(?=\nACTION:|$)", text, re.DOTALL
            )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]

        # Extract node from STRATEGIES + CHOSEN (new format)
        # Look for "N. ... → PARENT: node_id" lines, then use CHOSEN to pick one
        action = "draft_from_root"
        node_id = "root"

        strat_lines = re.findall(
            r"(\d+)\.\s.*?(?:→|->)\s*PARENT:\s*[\"']?(\S+?)[\"']?\s*(?:—|-|$)",
            text
        )
        chosen_match = re.search(r"CHOSEN:\s*(\d+)", text)

        if strat_lines and chosen_match:
            # New format: extract parent from the chosen strategy
            chosen_num = int(chosen_match.group(1))
            for num_str, parent in strat_lines:
                if int(num_str) == chosen_num:
                    node_id = parent.strip().rstrip("—-").strip()
                    break
            # Determine action from node_id
            if node_id == "root":
                action = "draft_from_root"
            else:
                action = "expand"
        else:
            # Fallback: old ACTION format
            action_match = re.search(r"ACTION:\s*(expand\s+(\S+)|draft_from_root)", text)
            if action_match:
                full = action_match.group(1).strip()
                if full.startswith("expand"):
                    action = "expand"
                    node_id = action_match.group(2) or "root"
                else:
                    action = "draft_from_root"
                    node_id = "root"

        # Extract DIRECTION
        direction_match = re.search(
            r"DIRECTION:\s*\n(.*?)(?=\nEXECUTOR_GUIDANCE:|MODE:)", text, re.DOTALL
        )
        direction = direction_match.group(1).strip() if direction_match else ""

        # Extract EXECUTOR_GUIDANCE
        guidance_match = re.search(
            r"EXECUTOR_GUIDANCE:\s*\n(.*?)(?=\nMODE:)", text, re.DOTALL
        )
        executor_guidance = ""
        if guidance_match:
            g = guidance_match.group(1).strip()
            if g.upper() != "NONE":
                executor_guidance = g

        # Extract MODE
        mode = "explore"
        mode_match = re.search(r"MODE:\s*(explore|exploit)", text, re.IGNORECASE)
        if mode_match:
            mode = mode_match.group(1).lower()

        # Extract MEMORY
        memory_update = ""
        memory_match = re.search(r"MEMORY:\s*\n?(.*?)$", text, re.DOTALL)
        if memory_match:
            memory_update = memory_match.group(1).strip()

        return ScientistDecision(
            action=action, node_id=node_id,
            direction=direction, mode=mode,
            memory_update=memory_update,
            reasoning=reasoning,
            executor_guidance=executor_guidance,
        )

    def _save_scientist_log(self, decision: ScientistDecision, budget_left: int,
                            tree_view: str, inspected_nodes: list[str] | None = None,
                            turn1_response: str = ""):
        """Save scientist decision to a log file for post-hoc analysis."""
        log_dir = self.output_dir / "scientist_logs"
        log_dir.mkdir(exist_ok=True)
        step_num = self.node_budget - budget_left
        log_file = log_dir / f"step_{step_num:03d}.json"
        data = {
            "step": step_num,
            "budget_left": budget_left,
            "action": decision.action,
            "node_id": decision.node_id,
            "direction": decision.direction,
            "executor_guidance": decision.executor_guidance,
            "mode": decision.mode,
            "memory_update": decision.memory_update,
            "reasoning": decision.reasoning,
            "tree_view": tree_view,
            "memory_state": list(self.memory),
            "inspected_nodes": inspected_nodes or [],
            "turn1_response": turn1_response,
        }
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Expand a single node
    # ------------------------------------------------------------------

    def _expand_one(self, parent_id: str, mode: str,
                    direction: str = "",
                    executor_guidance: str = "") -> TreeNode | None:
        """Create and execute a single child from the given parent."""
        parent = self.nodes[parent_id]

        # Assign child index
        if parent_id not in self._child_counter:
            self._child_counter[parent_id] = 0
        child_idx = self._child_counter[parent_id]
        self._child_counter[parent_id] += 1
        child_id = f"{parent_id}_{child_idx}"

        # Strategy comes directly from scientist — no separate VS
        strategy_text = direction or f"Attempt {child_idx}"

        print(f"  [{child_id}] mode={mode}, strategy: {strategy_text[:80]}")

        # Restore parent workspace
        self.container.restore_snapshot(parent.snapshot_path)
        if self.task.submission_file:
            self.container.communicate(
                f"rm -f /home/agent/workspace/{self.task.submission_file}"
            )

        # Build child conversation
        child_msgs = self._build_child_messages(parent, strategy_text, mode, executor_guidance)

        # Execute until validate
        try:
            score, actions, final_msgs = self._execute_until_validate(
                child_msgs, child_id
            )
            snap = self.container.save_snapshot(child_id)
            error = None
        except Exception as e:
            print(f"  ERROR: {e}")
            score, actions, final_msgs = None, [], child_msgs
            snap = ""
            error = str(e)

        child = TreeNode(
            node_id=child_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            strategy=strategy_text,
            score=score,
            actions=actions,
            conversation_history=final_msgs,
            snapshot_path=snap,
            error=error,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        self._save_node(child)

        if score is not None:
            print(f"  [{child_id}] score={score:.4f}")
        else:
            print(f"  [{child_id}] FAILED")

        return child

    # ------------------------------------------------------------------
    # Build child conversation messages
    # ------------------------------------------------------------------

    def _build_child_messages(
        self, parent: TreeNode, strategy_text: str, mode: str,
        executor_guidance: str = "",
    ) -> list[dict]:
        """Build the conversation messages for a child node."""
        child_msgs = copy.deepcopy(parent.conversation_history)
        write_instr = self.task.branch_write_instruction
        is_from_baseline = len(parent.actions) == 0

        parts = []

        # Inject executor guidance from the scientist (warnings, tips)
        if executor_guidance:
            parts.append(
                f"IMPORTANT WARNINGS FROM YOUR SUPERVISOR:\n{executor_guidance}"
            )

        score_str = f"{parent.score:.4f}" if parent.score is not None else "N/A (previous attempt failed)"

        if is_from_baseline:
            parts.append(f"Strategy to try: {strategy_text}")
            parts.append(write_instr)
        elif mode == "exploit":
            parts.append(
                f"Your current score is {score_str}. "
                f"Refine your current approach to improve it."
            )
            parts.append(f"Variation to try: {strategy_text}")
            parts.append(
                "IMPORTANT: Stay within the same approach as before. "
                "Do NOT switch to a completely different algorithm. "
                "Just tune or tweak the existing approach."
            )
            parts.append(write_instr)
        else:  # explore
            parts.append(
                f"Your current score is {score_str}. "
                f"Try a FUNDAMENTALLY DIFFERENT approach to improve it."
            )
            parts.append(f"Strategy: {strategy_text}")
            parts.append(write_instr)

        child_msgs.append({"role": "user", "content": "\n\n".join(parts)})
        return child_msgs

    # ------------------------------------------------------------------
    # Execute until validate (adapted from adaptive_tree_search)
    # ------------------------------------------------------------------

    def _execute_until_validate(
        self, messages: list[dict], node_id: str
    ) -> tuple[float | None, list[dict], list[dict]]:
        """Execute actions until validate is called.

        Returns (score, action_log, final_messages).
        """
        action_log = []
        score = None

        for step in range(self.max_actions):
            try:
                raw = self.executor.chat(messages)
            except Exception as e:
                time.sleep(2)
                try:
                    raw = self.executor.chat(messages)
                except Exception:
                    raise RuntimeError(f"LLM failed: {e}")

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "No command detected. Output a valid command.",
                })
                action_log.append({
                    "action": raw[:100], "observation": "No command", "step": step,
                })
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            if self.verbose:
                print(f"    [{node_id}] step {step}: {action[:80]}")

            obs, info = self.container.step(action)

            if self.verbose:
                if "validate" in action.strip().lower():
                    print(f"    [{node_id}] validate score={info.get('score')}")
                elif action.strip().startswith("python"):
                    has_error = "Traceback" in (obs or "") or "Error" in (obs or "")
                    if has_error:
                        print(f"    [{node_id}] python ERROR: {(obs or '')[-150:]}")

            action_log.append({
                "action": action[:2000],
                "observation": obs[:2000] if obs else "",
                "step": step,
            })
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            # Check for score
            score_found = self._extract_score(info, obs)
            if score_found is not None:
                score = score_found
                break
        else:
            # Force validate at max actions
            print(f"  [{node_id}] Max actions reached, forcing validate")
            obs, info = self.container.step("validate")
            messages.append({"role": "assistant", "content": "validate"})
            messages.append({"role": "user", "content": obs})
            action_log.append({
                "action": "validate (forced)",
                "observation": obs[:500],
                "step": self.max_actions,
            })
            score = self._extract_score(info, obs)

        return score, action_log, messages

    def _extract_score(self, info: dict, obs: str) -> float | None:
        """Extract the primary metric score from validate output."""
        if info.get("score"):
            score_data = info["score"][-1]
            if isinstance(score_data, dict):
                return score_data.get(
                    self.task.primary_metric, list(score_data.values())[0]
                )
            return score_data

        if obs and "Evaluation Score" in obs:
            import ast
            m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
            if m:
                try:
                    score_dict = ast.literal_eval(m.group(1))
                    return list(score_dict.values())[0]
                except Exception:
                    pass
        return None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _global_best_score(self) -> float:
        scored = [n.score for n in self.nodes.values() if n.score is not None]
        if not scored:
            return 0.0
        return max(scored) if self.task.higher_is_better else min(scored)

    def _compile_results(self, start_time: float) -> dict:
        scored_nodes = [
            (nid, n.score) for nid, n in self.nodes.items()
            if n.score is not None
        ]
        if not scored_nodes:
            best_id, best_score = "root", 0.0
        elif self.task.higher_is_better:
            best_id, best_score = max(scored_nodes, key=lambda x: x[1])
        else:
            best_id, best_score = min(scored_nodes, key=lambda x: x[1])

        elapsed = time.time() - start_time

        result = {
            "task": self.task.name,
            "primary_metric": self.task.primary_metric,
            "higher_is_better": self.task.higher_is_better,
            "selection_strategy": "llm_guided",
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "elapsed_seconds": round(elapsed, 1),
            "memory": list(self.memory),
            "tree_shape": {
                nid: {
                    "depth": n.depth,
                    "num_children": len(n.children),
                    "score": n.score,
                    "strategy": n.strategy[:100],
                }
                for nid, n in self.nodes.items()
            },
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "strategy": n.strategy[:100],
                    "score": n.score,
                    "actions_count": len(n.actions),
                    "children": n.children,
                    "error": n.error,
                }
                for nid, n in self.nodes.items()
            },
        }

        with open(self.output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        self._print_tree(best_id)
        return result

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
        print("LLM-GUIDED TREE SEARCH RESULTS")
        print(f"{'=' * 70}")

        best = self.nodes.get(best_id)
        if best and best.score is not None:
            print(
                f"Baseline: {self.container.baseline_score:.4f} | "
                f"Best: {best.score:.4f} (node: {best_id}) | "
                f"Improvement: {best.score - self.container.baseline_score:+.4f}"
            )
        print(f"Nodes explored: {len(self.nodes)}")
        if self.memory:
            print(f"Accumulated memory ({len(self.memory)} entries):")
            for m in self.memory:
                print(f"  - {m[:100]}")
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
                if cid in self.nodes:
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
            score_str = f"{n.score:.4f}" if n.score is not None else "N/A"
            print(f"  {p}: [{score_str}] {n.strategy[:80]}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: LLM-guided tree search with scientist + executor",
    )
    parser.add_argument("--task-config", default="tasks/titanic.yaml")
    parser.add_argument("--output-dir", default="outputs/llm_guided_search")
    parser.add_argument("--node-budget", type=int, default=12)
    parser.add_argument("--initial-breadth", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--env-gpu", default="7")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    parser.add_argument("--verbose", action="store_true")

    # Scientist model
    parser.add_argument("--scientist-model", default="gpt-4o",
                        help="Model for scientist (e.g., gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--scientist-url", default="",
                        help="API base URL for scientist (empty = OpenAI default)")
    parser.add_argument("--scientist-temperature", type=float, default=0.3)
    parser.add_argument("--scientist-thinking-budget", type=int, default=0,
                        help="Thinking budget tokens for scientist (0 = disabled)")

    # Executor model
    parser.add_argument("--executor-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model for executor (e.g., local vLLM model)")
    parser.add_argument("--executor-url", default="http://localhost:8000/v1",
                        help="vLLM URL for executor")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Executor temperature")
    parser.add_argument("--executor-thinking-budget", type=int, default=0,
                        help="Thinking budget tokens for executor (0 = disabled)")

    # Convenience aliases
    parser.add_argument("--model", default="",
                        help="Override executor model (backward compat)")
    parser.add_argument("--vllm-url", default="",
                        help="Override executor URL (backward compat)")

    args = parser.parse_args()

    # Handle backward-compat overrides
    executor_model = args.model or args.executor_model
    executor_url = args.vllm_url or args.executor_url

    task_profile = get_task_profile(args.task_config)
    print("=" * 60)
    print(f"Task: {task_profile.name}")
    print(f"LLM-Guided Tree Search (Experiment 4)")
    print("=" * 60)
    print(f"Scientist: {args.scientist_model}")
    print(f"Executor:  {executor_model}")
    print(f"Node budget: {args.node_budget}, Initial breadth: {args.initial_breadth}")
    print(f"Max actions/node: {args.max_actions}")
    print(f"Primary metric: {task_profile.primary_metric} "
          f"({'higher' if task_profile.higher_is_better else 'lower'} is better)")
    print()

    # Create LLM clients
    scientist = LLMClient(
        base_url=args.scientist_url or "",
        model=args.scientist_model,
        temperature=args.scientist_temperature,
        thinking_budget=args.scientist_thinking_budget,
    )
    executor = LLMClient(
        base_url=executor_url,
        model=executor_model,
        temperature=args.temperature,
        thinking_budget=args.executor_thinking_budget,
    )

    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name,
        task_profile=task_profile,
    )
    print("Creating MLGym container...")
    container.create()

    search = LLMGuidedTreeSearch(
        scientist=scientist,
        executor=executor,
        container=container,
        task_profile=task_profile,
        node_budget=args.node_budget,
        initial_breadth=args.initial_breadth,
        max_actions=args.max_actions,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    try:
        result = search.run()
        print(f"Results saved to {args.output_dir}/result.json")
    finally:
        container.close()


if __name__ == "__main__":
    main()
