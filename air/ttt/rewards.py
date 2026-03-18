"""Reward computation for scientist GRPO training.

Three components:
  R_explore  — information value of the chosen direction
  R_exploit  — genuine score improvement (gated by execution_status)
  R_memory   — surprise absorbed after memory update
"""

from __future__ import annotations

import math
import re
from typing import Optional

from air.tree_search import TreeNode
from air.ttt.prompts import ScientistOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard_tokens(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _extract_score_from_range(text: str) -> Optional[tuple[float, float]]:
    """Try to extract a (low, high) score range from free text."""
    # Match patterns like "0.85-0.90", "0.85 to 0.90", "[0.85, 0.90]"
    m = re.search(r"([\d.]+)\s*(?:to|-|–)\s*([\d.]+)", text)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# R_explore: information value of the direction
# ---------------------------------------------------------------------------

def r_explore(
    decision: ScientistOutput,
    child: TreeNode,
    all_nodes: dict[str, TreeNode],
) -> float:
    """Reward for exploration value of the chosen direction.

    Two components:
    1. Novelty: how different is this direction from all prior node strategies?
    2. Discrimination: does the experiment design distinguish between hypotheses?
    """
    # --- Novelty ---
    direction = decision.direction
    if not direction:
        return 0.0

    prior_strategies = [
        n.strategy for nid, n in all_nodes.items()
        if nid != "root" and nid != child.node_id and n.strategy
    ]
    if prior_strategies:
        max_sim = max(_jaccard_tokens(direction, s) for s in prior_strategies)
        novelty = 0.5 * (1.0 - max_sim)
    else:
        novelty = 0.5  # first node always novel

    # --- Discrimination value ---
    # Did the scientist make falsifiable predictions?
    discrimination = 0.0

    pred_range = decision.predicted_score_range
    has_prediction = pred_range[0] != 0.0 or pred_range[1] != 0.0

    if has_prediction and child.score is not None:
        lo, hi = pred_range
        if lo > hi:
            lo, hi = hi, lo

        in_range = lo <= child.score <= hi
        if not in_range:
            # Prediction was falsified — high information gain
            discrimination = 0.35
        else:
            # Prediction confirmed — some info gain
            discrimination = 0.1

        # Bonus: did EXPECTED_IF_TRUE and EXPECTED_IF_FALSE have non-overlapping ranges?
        range_true = _extract_score_from_range(decision.expected_if_true)
        range_false = _extract_score_from_range(decision.expected_if_false)
        if range_true and range_false:
            # Non-overlapping = good experimental design
            overlap = max(0, min(range_true[1], range_false[1]) - max(range_true[0], range_false[0]))
            total = max(range_true[1], range_false[1]) - min(range_true[0], range_false[0])
            if total > 0:
                separation = 1.0 - (overlap / total)
                discrimination += 0.15 * separation

    elif child.execution_status and child.execution_status != "success":
        # Discovered a new error type — some info value
        seen_errors = {
            n.error_type for n in all_nodes.values()
            if n.error_type and n.node_id != child.node_id
        }
        if child.error_type and child.error_type not in seen_errors:
            discrimination = 0.2

    return min(novelty + discrimination, 1.0)


# ---------------------------------------------------------------------------
# R_exploit: genuine improvement, gated by execution quality
# ---------------------------------------------------------------------------

def r_exploit(
    child: TreeNode,
    parent: TreeNode,
    baseline_score: float,
    higher_is_better: bool,
) -> float:
    """Reward for exploitation: genuine score improvement.

    Returns 0.0 for training_failed (baseline fallback scores get no credit).
    """
    # Gate: no reward for crashed executions
    if child.execution_status != "success":
        if child.execution_status in ("no_submission_produced", "no_validate_called"):
            return -0.1  # penalty for wasting budget
        return 0.0  # training_failed: no signal

    if child.score is None or parent.score is None:
        return 0.0

    # Signed improvement
    if higher_is_better:
        delta = child.score - parent.score
    else:
        delta = parent.score - child.score  # lower is better

    # Normalize by distance from baseline
    scale = max(abs(parent.score - baseline_score), 0.01)
    normalized = delta / scale

    # Tanh squashing to [-1, 1], then shift to [0, 1]
    squashed = math.tanh(normalized)
    result = (squashed + 1.0) / 2.0

    # Depth bonus: small reward for deepening rather than restarting from root
    depth_bonus = 0.05 * min(child.depth, 5)

    return min(result + depth_bonus, 1.0)


# ---------------------------------------------------------------------------
# R_memory: surprise absorbed after memory update
# ---------------------------------------------------------------------------

def r_memory(
    decision: ScientistOutput,
    child: TreeNode,
    all_nodes: dict[str, TreeNode],
) -> float:
    """Reward for memory update quality.

    Measures:
    1. Did the scientist notice surprising observations? (penalize missing them)
    2. Is the memory update specific and evidence-based?
    3. Is it non-redundant (not restating known facts)?
    """
    mem = decision.memory_update
    score = child.score
    reward = 0.0

    # --- Check if something surprising happened that warrants a memory update ---
    surprising = False

    # Predicted vs actual (surprise detection)
    pred_lo, pred_hi = decision.predicted_score_range
    has_pred = pred_lo != 0.0 or pred_hi != 0.0
    if has_pred and score is not None:
        if pred_lo > pred_hi:
            pred_lo, pred_hi = pred_hi, pred_lo
        if score < pred_lo or score > pred_hi:
            surprising = True

    # New error type
    if child.execution_status == "training_failed" and child.error_type:
        surprising = True

    # No memory update when something surprising happened → penalty
    if surprising and not mem:
        return -0.3

    if not mem:
        return 0.0  # nothing to update, nothing surprising → neutral

    # --- Evaluate memory update quality ---

    # Specificity: contains numbers, node references, or comparison language
    has_numbers = bool(re.search(r"\d+\.\d+", mem))
    has_comparison = bool(re.search(r"better|worse|improved|failed|higher|lower|plateau", mem, re.I))
    has_node_ref = bool(re.search(r"root_\d|node", mem, re.I))

    specificity = 0.0
    if has_numbers:
        specificity += 0.15
    if has_comparison:
        specificity += 0.1
    if has_node_ref:
        specificity += 0.05
    reward += specificity

    # Non-redundancy: check against existing memory (approximated from prior node strategies)
    existing_memories = [
        n.strategy for nid, n in all_nodes.items()
        if nid != "root" and nid != child.node_id
    ]
    if existing_memories:
        max_sim = max(_jaccard_tokens(mem, s) for s in existing_memories)
        if max_sim > 0.7:
            reward -= 0.2  # too similar to existing knowledge

    # Actionability: does it suggest what to try next?
    has_action = bool(re.search(r"try|should|next|instead|focus|avoid", mem, re.I))
    if has_action:
        reward += 0.1

    return max(min(reward, 0.5), -0.5)


# ---------------------------------------------------------------------------
# Combined reward + group advantages
# ---------------------------------------------------------------------------

def compute_reward(
    child: TreeNode,
    parent: TreeNode,
    all_nodes: dict[str, TreeNode],
    decision: ScientistOutput,
    baseline_score: float,
    higher_is_better: bool,
    w_explore: float = 0.3,
    w_exploit: float = 0.5,
    w_memory: float = 0.2,
    reward_mode: str = "granular",
    epsilon: float = 0.0,
) -> tuple[float, dict]:
    """Compute reward for a scientist decision.

    reward_mode:
      "granular" — weighted R_explore + R_exploit + R_memory (default)
      "binary"   — 1.0 if score > baseline + epsilon, else 0.0

    Returns (total_reward, component_dict).
    """
    if reward_mode == "binary":
        if child.score is None or child.execution_status != "success":
            total = 0.0
        elif higher_is_better:
            total = 1.0 if child.score > baseline_score + epsilon else 0.0
        else:
            total = 1.0 if child.score < baseline_score - epsilon else 0.0

        components = {
            "r_explore": 0.0,
            "r_exploit": total,
            "r_memory": 0.0,
            "total": total,
        }
        return total, components

    # Granular mode
    re_ = r_explore(decision, child, all_nodes)
    rx = r_exploit(child, parent, baseline_score, higher_is_better)
    rm = r_memory(decision, child, all_nodes)

    total = w_explore * re_ + w_exploit * rx + w_memory * rm

    components = {
        "r_explore": round(re_, 4),
        "r_exploit": round(rx, 4),
        "r_memory": round(rm, 4),
        "total": round(total, 4),
    }
    return total, components


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute GRPO advantages: normalize rewards within the group."""
    n = len(rewards)
    if n <= 1:
        return [0.0] * n

    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = max(var_r ** 0.5, 1e-8)

    return [(r - mean_r) / std_r for r in rewards]
