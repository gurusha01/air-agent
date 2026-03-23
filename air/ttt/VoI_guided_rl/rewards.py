"""
Three-component reward structure for VoI-guided RL.

R1 — Resolution: Did the experiment resolve its hypothesis?
     Sharpness-weighted prediction scoring.

R2 — Information: Did the explore node introduce a valuable hypothesis?
     Measured by VoI (centroid cosine distance).

R3 — Performance: Did the tree find a good solution?
     (best_score - baseline) / baseline. Applied once at end of tree.

Combined:
    R_node = alpha * R1 + beta * R2     (per node, dense signal)
    R_final = gamma * R3                 (end of tree, sparse signal)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NodeReward:
    r1_resolution: float = 0.0
    r2_information: float = 0.0
    r_node: float = 0.0


@dataclass
class TreeReward:
    r3_performance: float = 0.0
    node_rewards: dict[str, NodeReward] = None  # node_id -> NodeReward

    def __post_init__(self):
        if self.node_rewards is None:
            self.node_rewards = {}


# --- R1: Resolution Reward ---

def compute_r1(
    prediction_lower: float,
    prediction_upper: float,
    actual_score: float,
    max_r1: float = 1.0,
) -> float:
    """Compute resolution reward from sharpness-weighted prediction.

    R1 = prediction_correct / max(prediction_interval, 0.01)

    Narrow correct prediction → high reward.
    Wide vague prediction → low reward even if correct.
    Wrong prediction → zero.
    """
    interval = max(prediction_upper - prediction_lower, 0.01)
    correct = prediction_lower <= actual_score <= prediction_upper

    if correct:
        # Reward inversely proportional to interval width
        # Cap at max_r1 to prevent extreme values from tiny intervals
        return min(max_r1, 1.0 / interval)
    else:
        return 0.0


def parse_prediction(text: str) -> tuple[float, float] | None:
    """Extract prediction interval [lower, upper] from scientist output.

    Looks for patterns like:
    - "prediction: score will be 0.85-0.90"
    - "prediction: improvement of 0.02-0.04"
    - "prediction: [0.85, 0.90]"
    """
    # Try [lower, upper] format
    m = re.search(r'prediction.*?(\d+\.?\d*)\s*[-–,to]\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        lower, upper = float(m.group(1)), float(m.group(2))
        if lower > upper:
            lower, upper = upper, lower
        return (lower, upper)

    # Try single value "will score > X"
    m = re.search(r'(?:will|should|expect).*?(?:score|improve).*?(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        # Create a narrow interval around the prediction
        return (val * 0.95, val * 1.05)

    return None


# --- R2: Information Reward ---

def compute_r2(voi_score: float) -> float:
    """R2 = VoI for explore nodes, 0 for validate/challenge nodes.

    The VoI score is already computed in voi.py.
    """
    return voi_score


# --- R3: Performance Reward ---

def compute_r3(best_score: float, baseline_score: float) -> float:
    """R3 = (best_score - baseline) / max(|baseline|, 0.01)

    Applied once at end of tree. Grounds the search in actual performance.
    """
    if baseline_score == 0:
        return best_score  # avoid division by zero
    return (best_score - baseline_score) / max(abs(baseline_score), 0.01)


# --- Combined Reward ---

def compute_node_reward(
    r1: float,
    r2: float,
    alpha: float = 0.4,
    beta: float = 0.3,
) -> float:
    """Per-node reward: R_node = alpha * R1 + beta * R2"""
    return alpha * r1 + beta * r2


def compute_tree_rewards(
    node_rewards: dict[str, NodeReward],
    best_score: float,
    baseline_score: float,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    gamma_discount: float = 0.95,
    node_order: list[str] | None = None,
) -> dict[str, float]:
    """Compute final per-node cumulative rewards.

    Per-node: R_node = alpha * R1 + beta * R2
    End of tree: R3 = performance, discounted backward through tree
    Total per node: R_node + discounted R3 contribution

    Returns: {node_id: total_reward}
    """
    r3 = compute_r3(best_score, baseline_score)

    if node_order is None:
        node_order = sorted(node_rewards.keys())

    n = len(node_order)
    total_rewards = {}

    for i, nid in enumerate(node_order):
        nr = node_rewards[nid]
        # Per-node reward
        r_node = compute_node_reward(nr.r1_resolution, nr.r2_information, alpha, beta)

        # Discounted R3: later nodes get more credit (they benefited from earlier exploration)
        # But we also give credit to early nodes that set direction
        discount = gamma_discount ** (n - 1 - i)  # earlier nodes get less R3
        r_final = gamma * r3 * discount

        total_rewards[nid] = r_node + r_final

    return total_rewards
