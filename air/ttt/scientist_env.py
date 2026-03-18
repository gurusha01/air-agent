"""RL environment wrapping the MLGym search for scientist GRPO training.

State  = (tree_view, memory, task_desc, budget_left)
Action = scientist's structured text output
Step   = parse output → execute direction via executor → return reward info
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from air.tree_search import (
    ContainerManager,
    LLMClient,
    TaskProfile,
    TreeNode,
    classify_execution,
    extract_command,
    get_task_profile,
)
from air.ttt.prompts import (
    SCIENTIST_PROMPT,
    ScientistOutput,
    auto_select_inspect_nodes,
    format_node_code,
    parse_scientist_output,
)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScientistState:
    tree_view: str = ""
    memory: list[str] = field(default_factory=list)
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    budget_left: int = 0
    step_idx: int = 0
    prompt_text: str = ""


# ---------------------------------------------------------------------------
# Single environment
# ---------------------------------------------------------------------------

class ScientistEnv:
    """One MLGym search environment for the scientist."""

    def __init__(
        self,
        task_config: str,
        env_gpu: str,
        image_name: str,
        executor: LLMClient,
        max_actions: int = 15,
        node_budget: int = 5,
        verbose: bool = False,
    ):
        self.task_config = task_config
        self.env_gpu = env_gpu
        self.image_name = image_name
        self.executor = executor
        self.max_actions = max_actions
        self.node_budget = node_budget
        self.verbose = verbose

        self.task: TaskProfile = get_task_profile(task_config)
        self.container: ContainerManager | None = None
        self.nodes: dict[str, TreeNode] = {}
        self.memory: list[str] = []
        self._child_counter: dict[str, int] = {}

    # --- Lifecycle ---

    def create(self):
        """Create the MLGym container and root node."""
        self.container = ContainerManager(
            self.task_config, self.env_gpu, self.image_name,
            task_profile=self.task,
        )
        self.container.create()

    def reset(self) -> ScientistState:
        """Reset to a fresh search: create container + root baseline node."""
        if self.container is None:
            self.create()

        self.nodes = {}
        self.memory = []
        self._child_counter = {}

        root = self._execute_root()
        self.nodes[root.node_id] = root

        return self.get_state(self.node_budget)

    def close(self):
        if self.container:
            self.container.close()
            self.container = None

    # --- State ---

    def get_state(self, budget_left: int) -> ScientistState:
        """Build the current state (tree view + prompt)."""
        tree_view = self._build_tree_view()

        # Auto-select nodes for inspection
        inspect_ids = auto_select_inspect_nodes(self.nodes)
        if inspect_ids:
            code_parts = [format_node_code(nid, self.nodes) for nid in inspect_ids]
            code_inspections = "\n\n".join(code_parts)
        else:
            code_inspections = "(No nodes to inspect yet.)"

        memory_section = (
            "\n".join(f"- {m}" for m in self.memory)
            if self.memory
            else "(No accumulated knowledge yet.)"
        )

        # Truncate tree view and code inspections to prevent prompt overflow
        if len(tree_view) > 3000:
            tree_view = tree_view[:3000] + "\n... (truncated)"
        if len(code_inspections) > 2000:
            code_inspections = code_inspections[:2000] + "\n... (truncated)"

        task_desc = f"{self.task.name}"
        if self.task.task_type:
            task_desc += f" ({self.task.task_type})"

        task_details = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head="(data preview omitted for brevity)",
        )

        prompt = SCIENTIST_PROMPT.format(
            task_description=task_desc,
            task_details=task_details,
            metric_name=self.task.primary_metric,
            direction="higher" if self.task.higher_is_better else "lower",
            baseline_score=f"{self.container.baseline_score:.4f}",
            max_actions=self.max_actions,
            budget_left=budget_left,
            total_budget=self.node_budget,
            tree_view=tree_view,
            code_inspections=code_inspections,
            memory_section=memory_section,
        )

        return ScientistState(
            tree_view=tree_view,
            memory=list(self.memory),
            nodes=dict(self.nodes),
            budget_left=budget_left,
            prompt_text=prompt,
        )

    # --- Step ---

    def step(self, scientist_output: str) -> dict:
        """Execute one scientist decision. Returns info dict."""
        decision = parse_scientist_output(scientist_output)

        # Update memory
        if decision.memory_update:
            self.memory.append(decision.memory_update)
            if len(self.memory) > 5:
                self.memory = self.memory[-5:]

        # Validate parent
        parent_id = decision.parent_id
        if parent_id not in self.nodes:
            parent_id = "root"

        # Expand
        child = self._expand_one(
            parent_id,
            mode=decision.mode,
            direction=decision.direction,
            executor_guidance=decision.executor_guidance,
        )

        return {
            "child_id": child.node_id,
            "parent_id": parent_id,
            "score": child.score,
            "execution_status": child.execution_status,
            "error_type": child.error_type,
            "parsed_decision": decision,
        }

    # --- Sync state from another env (for parallel execution) ---

    def sync_from(self, other: ScientistEnv):
        """Copy tree state and memory from another env."""
        self.nodes = copy.deepcopy(other.nodes)
        self.memory = list(other.memory)
        self._child_counter = dict(other._child_counter)

    def merge_child(self, child_id: str, source_env: ScientistEnv):
        """Merge a child node from another env into this one."""
        if child_id in source_env.nodes:
            self.nodes[child_id] = copy.deepcopy(source_env.nodes[child_id])
            # Update parent's children list
            parent_id = source_env.nodes[child_id].parent_id
            if parent_id and parent_id in self.nodes:
                if child_id not in self.nodes[parent_id].children:
                    self.nodes[parent_id].children.append(child_id)

    # --- Root node ---

    def _execute_root(self) -> TreeNode:
        data_head = ""
        if self.task.data_head_cmd:
            data_head = self.container.communicate(self.task.data_head_cmd)

        task_desc = self.task.root_task_desc.format(
            baseline_score=self.container.baseline_score,
            data_head=data_head,
        )
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
            execution_status="success",
        )

    # --- Expand one node ---

    def _expand_one(
        self, parent_id: str, mode: str, direction: str,
        executor_guidance: str = "",
    ) -> TreeNode:
        parent = self.nodes[parent_id]

        if parent_id not in self._child_counter:
            self._child_counter[parent_id] = 0
        child_idx = self._child_counter[parent_id]
        self._child_counter[parent_id] += 1
        child_id = f"{parent_id}_{child_idx}"

        strategy_text = direction or f"Attempt {child_idx}"
        if self.verbose:
            print(f"  [{child_id}] mode={mode}, strategy: {strategy_text[:80]}")

        # Restore parent workspace
        self.container.restore_snapshot(parent.snapshot_path)
        if self.task.submission_file:
            self.container.communicate(
                f"rm -f /home/agent/workspace/{self.task.submission_file}"
            )

        # Build child conversation
        child_msgs = self._build_child_messages(parent, strategy_text, mode, executor_guidance)

        # Execute
        try:
            score, actions, final_msgs = self._execute_until_validate(child_msgs, child_id)
            snap = self.container.save_snapshot(child_id)
            error = None
        except Exception as e:
            if self.verbose:
                print(f"  ERROR: {e}")
            score, actions, final_msgs = None, [], child_msgs
            snap = ""
            error = str(e)

        exec_status, err_type = classify_execution(actions, score)

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
            execution_status=exec_status,
            error_type=err_type,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)

        if self.verbose:
            s = f"score={score:.4f}" if score is not None else "FAILED"
            print(f"  [{child_id}] {s} (status={exec_status})")

        return child

    def _build_child_messages(
        self, parent: TreeNode, strategy_text: str, mode: str,
        executor_guidance: str = "",
    ) -> list[dict]:
        child_msgs = copy.deepcopy(parent.conversation_history)
        write_instr = self.task.branch_write_instruction
        is_from_baseline = len(parent.actions) == 0

        parts = []
        if executor_guidance:
            parts.append(f"IMPORTANT WARNINGS FROM YOUR SUPERVISOR:\n{executor_guidance}")

        score_str = f"{parent.score:.4f}" if parent.score is not None else "N/A"

        if is_from_baseline:
            parts.append(f"Strategy to try: {strategy_text}")
            parts.append(write_instr)
        elif mode == "exploit":
            parts.append(f"Your current score is {score_str}. Refine your approach.")
            parts.append(f"Variation: {strategy_text}")
            parts.append("Stay within the same approach. Just tune or tweak.")
            parts.append(write_instr)
        else:
            parts.append(f"Your current score is {score_str}. Try a DIFFERENT approach.")
            parts.append(f"Strategy: {strategy_text}")
            parts.append(write_instr)

        child_msgs.append({"role": "user", "content": "\n\n".join(parts)})
        return child_msgs

    def _execute_until_validate(
        self, messages: list[dict], node_id: str
    ) -> tuple[float | None, list[dict], list[dict]]:
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
                messages.append({"role": "user", "content": "No command detected. Output a valid command."})
                action_log.append({"action": raw[:100], "observation": "No command", "step": step})
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            obs, info = self.container.step(action)

            action_log.append({
                "action": action[:2000],
                "observation": obs[:2000] if obs else "",
                "step": step,
            })
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            score_found = self._extract_score(info, obs)
            if score_found is not None:
                score = score_found
                break
        else:
            obs, info = self.container.step("validate")
            messages.append({"role": "assistant", "content": "validate"})
            messages.append({"role": "user", "content": obs})
            action_log.append({"action": "validate (forced)", "observation": obs[:500], "step": self.max_actions})
            score = self._extract_score(info, obs)

        return score, action_log, messages

    def _extract_score(self, info: dict, obs: str) -> float | None:
        import ast
        if info.get("score"):
            score_data = info["score"][-1]
            if isinstance(score_data, dict):
                return score_data.get(self.task.primary_metric, list(score_data.values())[0])
            return score_data
        if obs and "Evaluation Score" in obs:
            m = re.search(r"Evaluation Score:\s*(\{[^}]+\})", obs)
            if m:
                try:
                    score_dict = ast.literal_eval(m.group(1))
                    return list(score_dict.values())[0]
                except Exception:
                    pass
        return None

    # --- Tree view ---

    def _build_tree_view(self) -> str:
        lines = []

        def _node_line(nid: str, indent: int = 0):
            n = self.nodes[nid]
            prefix = "  " * indent
            score_str = f"{n.score:.4f}" if n.score is not None else "FAILED"
            strategy = n.strategy[:120] if n.strategy else "N/A"

            # Status label
            status_str = ""
            if nid != "root":
                status = n.execution_status or ""
                err_t = n.error_type or ""
                if status == "success":
                    status_str = f" [success]"
                elif status == "training_failed":
                    fallback = ""
                    if self.container and self.container.baseline_score is not None and n.score is not None:
                        if abs(n.score - self.container.baseline_score) < 0.02:
                            fallback = " BASELINE_FALLBACK"
                    status_str = f" [training_failed:{err_t}{fallback}]"
                elif status:
                    status_str = f" [{status}:{err_t}]" if err_t else f" [{status}]"

            lines.append(
                f"{prefix}Node {nid} [{strategy}]\n"
                f"{prefix}  Score: {score_str}{status_str} | "
                f"Actions: {len(n.actions)} | Children: {len(n.children)}"
            )

            for cid in n.children:
                if cid in self.nodes:
                    _node_line(cid, indent + 1)

        if "root" in self.nodes:
            _node_line("root")
        else:
            lines.append("(empty tree)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parallel K-way execution using K containers
# ---------------------------------------------------------------------------

class ParallelScientistEnv:
    """K-way scientist execution using K parallel containers.

    Each of K scientist outputs runs in its own container simultaneously
    via ThreadPoolExecutor. All containers share the same vLLM endpoint.
    """

    def __init__(
        self,
        K: int,
        task_config: str,
        env_gpu: str,
        image_name: str,
        executor: LLMClient,
        max_actions: int = 15,
        node_budget: int = 5,
        verbose: bool = False,
    ):
        self.K = K
        # Primary env holds the canonical tree state
        self.primary = ScientistEnv(
            task_config, env_gpu, image_name, executor,
            max_actions, node_budget, verbose,
        )
        # K worker envs, each with its own container
        self.workers: list[ScientistEnv] = []
        for _ in range(K):
            self.workers.append(ScientistEnv(
                task_config, env_gpu, image_name, executor,
                max_actions, node_budget, verbose,
            ))

    def create_all(self):
        """Create all K+1 containers."""
        print(f"Creating {self.K + 1} containers...")
        self.primary.create()
        for i, w in enumerate(self.workers):
            print(f"  Worker {i + 1}/{self.K}")
            w.create()
        print(f"All {self.K + 1} containers ready.")

    def reset(self) -> ScientistState:
        """Reset primary to fresh search with root node."""
        state = self.primary.reset()
        # Sync all workers to the root snapshot
        root_snap = self.primary.nodes["root"].snapshot_path
        for w in self.workers:
            w.container.restore_snapshot(root_snap)
        return state

    def get_state(self, budget_left: int) -> ScientistState:
        return self.primary.get_state(budget_left)

    def step_parallel(
        self, scientist_outputs: list[str]
    ) -> list[dict]:
        """Execute K scientist outputs in PARALLEL across K containers.

        Each worker k:
          1. Syncs tree state from primary
          2. Gets a unique child_counter offset (so child IDs don't collide)
          3. Executes in its own container
        All K run simultaneously via ThreadPoolExecutor.
        Returns K info dicts.
        """
        from concurrent.futures import ThreadPoolExecutor

        assert len(scientist_outputs) == self.K

        # Save primary state
        saved_nodes = copy.deepcopy(self.primary.nodes)
        saved_memory = list(self.primary.memory)
        saved_counters = dict(self.primary._child_counter)

        def _run_worker(k: int) -> tuple[dict, TreeNode, str]:
            w = self.workers[k]
            # Sync tree state from primary
            w.nodes = copy.deepcopy(saved_nodes)
            w.memory = list(saved_memory)
            # Give each worker a unique counter offset
            w._child_counter = {
                key: val + k for key, val in saved_counters.items()
            }
            # For new parents not in saved_counters, _expand_one starts at 0.
            # Add the offset via a wrapper — but simpler: just use k as suffix.
            # Actually _expand_one uses _child_counter[parent_id] which starts
            # at 0 for new parents. With K workers, we need unique IDs.
            # Override _next_child_id behavior by pre-setting counter for all
            # possible parents.
            for nid in list(w.nodes.keys()):
                if nid not in w._child_counter:
                    w._child_counter[nid] = k

            # Execute
            info = w.step(scientist_outputs[k])

            # Capture child node and snapshot
            child_id = info["child_id"]
            child_node = copy.deepcopy(w.nodes[child_id])
            snap = w.nodes[child_id].snapshot_path

            return info, child_node, snap

        # Run all K in parallel
        with ThreadPoolExecutor(max_workers=self.K) as pool:
            futures = [pool.submit(_run_worker, k) for k in range(self.K)]
            worker_results = [f.result() for f in futures]

        results = []
        all_child_nodes = []
        all_child_snaps = []
        for info, child_node, snap in worker_results:
            results.append(info)
            all_child_nodes.append(child_node)
            all_child_snaps.append(snap)

        # Merge all K children into primary tree
        self.primary.nodes = copy.deepcopy(saved_nodes)
        self.primary.memory = list(saved_memory)
        self.primary._child_counter = dict(saved_counters)

        for child_node in all_child_nodes:
            self.primary.nodes[child_node.node_id] = child_node
            parent_id = child_node.parent_id
            if parent_id and parent_id in self.primary.nodes:
                if child_node.node_id not in self.primary.nodes[parent_id].children:
                    self.primary.nodes[parent_id].children.append(
                        child_node.node_id
                    )

        # Update child counter past all K children
        if all_child_nodes:
            parent_id = all_child_nodes[0].parent_id
            if parent_id:
                self.primary._child_counter[parent_id] = (
                    saved_counters.get(parent_id, 0) + self.K
                )

        self._all_child_snaps = all_child_snaps
        return results

    def commit_child(self, chosen_k: int, results: list[dict]):
        """Commit the chosen child: restore primary workspace to its snapshot."""
        snap = self._all_child_snaps[chosen_k]
        if snap:
            self.primary.container.restore_snapshot(snap)
            # Also restore chosen worker's snapshot to all other workers
            for w in self.workers:
                w.container.restore_snapshot(snap)

        # Update memory from the chosen decision
        decision = results[chosen_k]["parsed_decision"]
        if decision.memory_update:
            self.primary.memory.append(decision.memory_update)
            if len(self.primary.memory) > 5:
                self.primary.memory = self.primary.memory[-5:]

    def close_all(self):
        self.primary.close()
        for w in self.workers:
            w.close()
