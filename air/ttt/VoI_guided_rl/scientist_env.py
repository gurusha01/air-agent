"""
VoI-guided scientist environment for GRPO training.

Wraps the tree search with:
- thought.md instead of flat memory
- Node type enforcement (explore/validate/challenge)
- VoI computation for explore nodes
- Three-component rewards (R1 + R2 + R3)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .thought import ThoughtDoc, HypothesisStatus
from .prompts import (
    SCIENTIST_SYSTEM,
    format_scientist_prompt,
    format_proposal_prompt,
    parse_scientist_output,
    validate_node_type,
)
from .rewards import (
    NodeReward,
    TreeReward,
    compute_r1,
    compute_r2,
    compute_r3,
    compute_tree_rewards,
    parse_prediction,
)
from .voi import compute_voi, estimate_p_true, sample_proposals

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class VoIState:
    """State for GRPO: everything needed to generate a scientist proposal."""
    prompt: str = ""
    thought_md: str = ""
    tree_view: str = ""
    task_description: str = ""
    budget_left: int = 0
    step_idx: int = 0


@dataclass
class StepResult:
    """Result of one tree search step."""
    node_id: str = ""
    node_type: str = ""
    hypothesis_id: str = ""
    score: float | None = None
    reward: NodeReward = field(default_factory=NodeReward)
    valid: bool = True
    error: str = ""


class VoIScientistEnv:
    """Tree search environment with VoI-guided hypothesis-driven search.

    Designed for GRPO training: at each step, sample K proposals from the
    scientist, compute rewards, and provide advantages for GRPO update.
    """

    def __init__(
        self,
        task_config: str,
        task_description: str,
        task_name: str,
        metric_name: str,
        higher_is_better: bool,
        baseline_score: float,
        node_budget: int = 12,
        validate_budget_per_hypothesis: int = 3,
        voi_K: int = 32,
        alpha: float = 0.4,   # R1 weight
        beta: float = 0.3,    # R2 weight
        gamma: float = 0.3,   # R3 weight
    ):
        self.task_config = task_config
        self.task_description = task_description
        self.task_name = task_name
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.baseline_score = baseline_score
        self.node_budget = node_budget
        self.validate_budget = validate_budget_per_hypothesis
        self.voi_K = voi_K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # State
        self.thought = ThoughtDoc()
        self.tree_nodes: list[dict] = []  # ordered list of executed nodes
        self.node_rewards: dict[str, NodeReward] = {}
        self.best_score: float = baseline_score
        self.step_count: int = 0

    def reset(self) -> VoIState:
        """Reset to empty tree with just the baseline node."""
        self.thought = ThoughtDoc()
        self.tree_nodes = [{
            "node_id": "root",
            "depth": 0,
            "score": self.baseline_score,
            "strategy": "Baseline (no model execution)",
            "type": "baseline",
            "hypothesis": None,
        }]
        self.node_rewards = {}
        self.best_score = self.baseline_score
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> VoIState:
        """Build current state for the scientist."""
        tree_view = self._render_tree()
        thought_md = self.thought.render()
        budget_left = self.node_budget - len(self.tree_nodes) + 1  # +1 for root

        direction = "higher" if self.higher_is_better else "lower"
        prompt = format_scientist_prompt(
            task_description=self.task_description,
            task_details=self.task_description,  # simplified
            metric_name=self.metric_name,
            direction=direction,
            baseline_score=f"{self.baseline_score:.4f}",
            thought_md=thought_md,
            tree_view=tree_view,
            budget_left=budget_left,
            total_budget=self.node_budget,
            validate_budget=self.validate_budget,
        )

        return VoIState(
            prompt=prompt,
            thought_md=thought_md,
            tree_view=tree_view,
            task_description=self.task_description,
            budget_left=budget_left,
            step_idx=self.step_count,
        )

    def _render_tree(self) -> str:
        """Render the experiment tree as text."""
        lines = []
        for node in self.tree_nodes:
            depth = node.get("depth", 0)
            indent = "  " * depth
            score = node.get("score")
            score_str = f"{score:.4f}" if score is not None else "FAIL"
            strategy = node.get("strategy", "")[:100]
            ntype = node.get("type", "")
            lines.append(f"{indent}{node['node_id']} [{score_str}] ({ntype}) {strategy}")
        return "\n".join(lines)

    def step(
        self,
        scientist_output: str,
        actual_score: float | None,
        model: "PreTrainedModel | None" = None,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
    ) -> StepResult:
        """Process one scientist proposal and compute rewards.

        Args:
            scientist_output: raw text from the scientist model
            actual_score: score from running the experiment (or None if failed)
            model: scientist model (needed for VoI computation)
            tokenizer: tokenizer (needed for VoI computation)

        Returns:
            StepResult with node info and rewards
        """
        self.step_count += 1
        parsed = parse_scientist_output(scientist_output)
        node_type = parsed["type"]
        hypothesis_text = parsed["hypothesis"]
        experiment = parsed["experiment"]
        prediction_text = parsed.get("prediction", "")
        thought_update = parsed.get("thought_update", "")

        # --- Determine hypothesis ID ---
        hypothesis_id = ""
        if node_type == "explore":
            # New hypothesis
            hypothesis_id = self.thought.add_hypothesis(
                hypothesis_text, validate_budget=self.validate_budget
            )
        else:
            # Existing hypothesis reference
            hypothesis_id = hypothesis_text.strip()
            if not hypothesis_id.startswith("H"):
                # Try to find by partial match
                for h in self.thought.hypotheses.values():
                    if hypothesis_text.lower() in h.claim.lower():
                        hypothesis_id = h.id
                        break

        # --- Validate node type ---
        valid, error = validate_node_type(node_type, hypothesis_id, self.thought)
        if not valid:
            return StepResult(
                node_id=f"node_{self.step_count}",
                node_type=node_type,
                hypothesis_id=hypothesis_id,
                valid=False,
                error=error,
                reward=NodeReward(),
            )

        # --- Create node ---
        node_id = f"node_{self.step_count}"
        parent_id = self.tree_nodes[-1]["node_id"] if self.tree_nodes else "root"
        depth = self.tree_nodes[-1].get("depth", 0) + 1 if self.tree_nodes else 1

        node = {
            "node_id": node_id,
            "depth": min(depth, 3),  # cap depth for readability
            "score": actual_score,
            "strategy": experiment[:100],
            "type": node_type,
            "hypothesis": hypothesis_id,
        }
        self.tree_nodes.append(node)

        if actual_score is not None and (
            (self.higher_is_better and actual_score > self.best_score) or
            (not self.higher_is_better and actual_score < self.best_score)
        ):
            self.best_score = actual_score

        # --- Compute R1: Resolution ---
        r1 = 0.0
        prediction = parse_prediction(prediction_text)
        if prediction and actual_score is not None:
            r1 = compute_r1(prediction[0], prediction[1], actual_score)

        # --- Compute R2: Information (VoI for explore nodes only) ---
        r2 = 0.0
        if node_type == "explore" and model is not None and tokenizer is not None:
            state = self._get_state()
            proposal_prompt = format_proposal_prompt(
                self.task_description, state.thought_md, state.tree_view
            )

            # Prepend hypothesis confirmed/rejected
            h_claim = hypothesis_text
            prompt_h_true = format_proposal_prompt(
                self.task_description,
                state.thought_md + f"\n{hypothesis_id}: CONFIRMED TRUE - {h_claim}",
                state.tree_view,
            )
            prompt_h_false = format_proposal_prompt(
                self.task_description,
                state.thought_md + f"\n{hypothesis_id}: REJECTED FALSE - {h_claim}",
                state.tree_view,
            )

            proposals_u = sample_proposals(model, tokenizer, proposal_prompt, K=self.voi_K)
            proposals_t = sample_proposals(model, tokenizer, prompt_h_true, K=self.voi_K)
            proposals_f = sample_proposals(model, tokenizer, prompt_h_false, K=self.voi_K)

            p_true = estimate_p_true(model, tokenizer, state.thought_md, h_claim, self.task_description)

            r2 = compute_voi(proposals_u, proposals_t, proposals_f, p_true, model, tokenizer)

            # Update hypothesis confidence from logits
            if hypothesis_id in self.thought.hypotheses:
                self.thought.hypotheses[hypothesis_id].confidence = p_true

        # --- Update thought.md ---
        if node_type == "validate" and hypothesis_id in self.thought.hypotheses:
            success = actual_score is not None and r1 > 0
            result_text = f"score={actual_score}" if actual_score else "FAILED"
            self.thought.record_validate(hypothesis_id, result_text, success)

        elif node_type == "challenge" and hypothesis_id in self.thought.hypotheses:
            survives = actual_score is not None and r1 == 0  # prediction wrong = hypothesis challenged
            result_text = f"score={actual_score}" if actual_score else "FAILED"
            self.thought.record_challenge(hypothesis_id, result_text, survives)

        if thought_update:
            # Parse structural claims and open questions
            if "structural claim:" in thought_update.lower():
                claim = re.search(r'structural claim:\s*(.+?)(?:\n|$)', thought_update, re.IGNORECASE)
                if claim:
                    self.thought.add_structural_claim(claim.group(1).strip())
            if "open question:" in thought_update.lower():
                q = re.search(r'open question:\s*(.+?)(?:\n|$)', thought_update, re.IGNORECASE)
                if q:
                    self.thought.add_open_question(q.group(1).strip())

        # --- Build reward ---
        reward = NodeReward(
            r1_resolution=r1,
            r2_information=r2,
            r_node=self.alpha * r1 + self.beta * r2,
        )
        self.node_rewards[node_id] = reward

        return StepResult(
            node_id=node_id,
            node_type=node_type,
            hypothesis_id=hypothesis_id,
            score=actual_score,
            reward=reward,
            valid=True,
        )

    def get_tree_rewards(self) -> dict[str, float]:
        """Compute final per-node cumulative rewards including R3."""
        node_order = [n["node_id"] for n in self.tree_nodes if n["node_id"] != "root"]
        return compute_tree_rewards(
            self.node_rewards,
            self.best_score,
            self.baseline_score,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            node_order=node_order,
        )

    def get_summary(self) -> dict:
        """Get a summary of the tree search for logging."""
        return {
            "task": self.task_name,
            "baseline": self.baseline_score,
            "best_score": self.best_score,
            "improvement": self.best_score - self.baseline_score,
            "nodes": len(self.tree_nodes),
            "hypotheses": len(self.thought.hypotheses),
            "validated": len(self.thought.get_validated()),
            "thought_md": self.thought.to_dict(),
            "tree": self.tree_nodes,
            "rewards": {nid: {"r1": nr.r1_resolution, "r2": nr.r2_information, "r_node": nr.r_node}
                       for nid, nr in self.node_rewards.items()},
        }
