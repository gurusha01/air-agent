"""
Experiment 3: Adaptive Explore-Exploit Tree Search (MCTS-style).

This replaces the fixed BFS expansion of tree_search.py (Experiment 2) with an
MCTS-style loop where two decisions are made at each iteration:

    1. WHICH node to expand (selection policy)
    2. HOW to expand it: explore (Tail VS) or exploit (Local VS)

Three selection strategies are available (--selection-strategy):

    "signals" (original) — weighted combination of five toggleable signals
        (variance, regret, llm-guidance, coverage, depth). Each is independently
        normalized to [0,1] and summed. Explore/exploit decided separately by
        percentile threshold.

    "ucb" — classic UCB1 (Upper Confidence Bound) from MCTS literature.
        score(node) = Q(node) + C * sqrt(ln(N) / (n_children + 1))
        where Q = best_child / global_best, N = total expansions.
        Explore/exploit is coupled: if exploration term > exploitation term,
        use explore mode; otherwise exploit.

    "open-ended" — UCB with trend bonus + path commitment for open-ended
        exploration. Adds two mechanisms beyond vanilla UCB:
        (1) Trend bonus: branches showing improvement (even from low base)
            get a bonus proportional to their improvement rate. This prevents
            abandoning a 0.6→0.66→0.7 chain that might reach 0.98.
        (2) Path commitment: after selecting a node, if the previous expansion
            of that same branch showed improvement, commit to K more expansions
            before re-evaluating. This gives "valleys" time to recover.
        This captures the scientific insight that sometimes you need to go
        through a bad region to discover a breakthrough on the other side.

Additionally, the agent can operate in two INFORMATION CONTEXT modes:

    --context parent   : child only sees its parent's conversation history
                         (same as Exp 2 — siblings don't know about each other)
    --context global   : child also sees a summary of the entire tree —
                         what strategies were tried, what worked, what failed,
                         and what the best approach is so far.

Usage:
    # Terminal 1: Start vLLM (same as Exp 2)
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 32768

    # Terminal 2: Run with UCB selection
    cd /home/ubuntu/MLScientist/MLGym
    uv run --project /home/ubuntu/MLScientist/air-agent \\
        python /home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py \\
        --selection-strategy ucb --ucb-c 1.41 \\
        --context global \\
        --task-config tasks/titanic.yaml

    # Terminal 2: Run with open-ended exploration
    uv run --project /home/ubuntu/MLScientist/air-agent \\
        python /home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py \\
        --selection-strategy open-ended --ucb-c 1.41 \\
        --trend-weight 0.5 --commitment-threshold 2 \\
        --context global \\
        --task-config tasks/titanic.yaml

    # Terminal 2: Run with original signal-based selection
    uv run --project /home/ubuntu/MLScientist/air-agent \\
        python /home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py \\
        --selection-strategy signals \\
        --use-regret --use-depth \\
        --context global \\
        --task-config tasks/titanic.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import time

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from MLGym dir or project root (for OPENAI_API_KEY etc.)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# Reuse core components from Exp 2's tree_search.py. We import rather than
# copy so that tree_search.py remains the canonical Exp 2 implementation
# and any bug fixes there automatically apply here.
from air.tree_search import (
    TaskProfile,
    TASK_PROFILES,
    get_task_profile,
    ContainerManager,
    LLMClient,
    TreeNode,
    extract_command,
    STRATEGY_PROMPT_TAIL,
    STRATEGY_PROMPT_LOCAL,
    MLGYM_PATH,
)
from air.reflexion import build_reflection, inject_error_analysis_script, error_analysis_hint


# ---------------------------------------------------------------------------
# Guidance Client — separate (stronger) model for interestingness ratings
# ---------------------------------------------------------------------------

class GuidanceClient:
    """Thin wrapper for calling an external frontier model (e.g. GPT-5.2)
    for the interestingness/depth-potential LLM guidance signal.

    Uses the OpenAI API with OPENAI_API_KEY from the environment.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Needed for --guidance-model. "
                "Set it in .env or export it."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Strategy family taxonomy (for coverage / QD signal)
# ---------------------------------------------------------------------------
# We define broad strategy families and detect them via keyword matching on
# the strategy description text. This is intentionally coarse — we just want
# to know "has the tree tried tree-based methods?" not "which exact model."

STRATEGY_FAMILIES = {
    "tree_based": [
        "random forest", "decision tree", "gradient boosting", "gbm",
        "xgboost", "lightgbm", "extra trees", "adaboost", "catboost",
    ],
    "linear": [
        "logistic regression", "linear regression", "ridge", "lasso",
        "elastic net", "logreg", "regularization", "l1", "l2",
    ],
    "svm": [
        "svm", "support vector", "svc", "svr", "rbf kernel",
    ],
    "neural": [
        "neural network", "mlp", "deep learning", "dense layer",
        "dropout", "nn", "lstm", "cnn", "transformer",
    ],
    "ensemble": [
        "stacking", "voting", "bagging", "blending", "ensemble",
    ],
    "knn": [
        "knn", "k-nearest", "neighbors",
    ],
    "bayesian": [
        "bayesian", "naive bayes", "gaussian process",
    ],
    "feature_eng": [
        "feature engineering", "polynomial", "interaction terms",
        "target encoding", "pca", "dimensionality reduction",
    ],
}


def classify_strategy(strategy_text: str) -> set[str]:
    """Classify a strategy description into one or more families.

    Returns the set of family names whose keywords appear in the text.
    A strategy can belong to multiple families (e.g., "XGBoost with PCA"
    matches both tree_based and feature_eng).
    """
    text_lower = strategy_text.lower()
    families = set()
    for family, keywords in STRATEGY_FAMILIES.items():
        if any(kw in text_lower for kw in keywords):
            families.add(family)
    return families if families else {"other"}


# ---------------------------------------------------------------------------
# Selection Policy — computes a score for each expandable node
# ---------------------------------------------------------------------------

class SelectionPolicy:
    """Computes selection scores for expandable nodes using a weighted
    combination of independently toggleable signals.

    Each signal is normalized to roughly [0, 1] so that they contribute
    comparably when combined. The final selection score for a node is:

        score = sum(weight_i * signal_i(node))  for each enabled signal

    where weight_i is 1.0 if the signal is enabled, 0.0 if disabled.
    """

    def __init__(
        self,
        use_variance: bool = False,
        use_regret: bool = False,
        use_llm_guidance: bool = False,
        use_coverage: bool = False,
        use_depth: bool = False,
        llm_client: LLMClient | None = None,
        guidance_client: "GuidanceClient | None" = None,
        task_profile: TaskProfile | None = None,
    ):
        self.use_variance = use_variance
        self.use_regret = use_regret
        self.use_llm_guidance = use_llm_guidance
        self.use_coverage = use_coverage
        self.use_depth = use_depth
        self.llm = llm_client
        # guidance_client: a separate (stronger) model for interestingness/
        # depth-potential ratings. Falls back to self.llm if not provided.
        self.guidance = guidance_client
        self.task = task_profile

        # At least one signal must be enabled; fall back to uniform random
        self.any_enabled = any([
            use_variance, use_regret, use_llm_guidance, use_coverage, use_depth
        ])

    def select(
        self,
        nodes: dict[str, TreeNode],
        candidates: list[str],
        global_best: float,
    ) -> tuple[str, dict[str, float]]:
        """Select the best node to expand from the candidate list.

        Args:
            nodes: all nodes in the tree (for computing signals)
            candidates: list of node_ids eligible for expansion
            global_best: best score seen so far in the tree

        Returns:
            (selected_node_id, debug_scores_dict)
            where debug_scores_dict maps node_id -> final selection score
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1 or not self.any_enabled:
            # No selection needed (or no signals enabled — pick randomly)
            return candidates[0], {c: 1.0 for c in candidates}

        # Collect all strategy families used in the tree (for coverage signal)
        all_families_used = set()
        for n in nodes.values():
            if n.strategy and n.score is not None:
                all_families_used |= classify_strategy(n.strategy)

        # Compute raw signal values for each candidate
        raw_scores: dict[str, dict[str, float]] = {}
        for cid in candidates:
            node = nodes[cid]
            raw_scores[cid] = {}

            if self.use_variance:
                raw_scores[cid]["variance"] = self._child_variance(node, nodes)

            if self.use_regret:
                raw_scores[cid]["regret"] = self._regret(node, nodes, global_best)

            if self.use_coverage:
                raw_scores[cid]["coverage"] = self._coverage_gap(
                    node, nodes, all_families_used
                )

            if self.use_depth:
                raw_scores[cid]["depth"] = self._depth_visit_bonus(node)

        # LLM guidance is called after the cheap signals (it's expensive).
        # Use the dedicated guidance client (frontier model) if available,
        # otherwise fall back to the local LLM.
        guidance_model = self.guidance or self.llm
        if self.use_llm_guidance and guidance_model:
            llm_scores = self._llm_guidance_batch(
                nodes, candidates, global_best, guidance_model
            )
            for cid in candidates:
                raw_scores[cid]["llm"] = llm_scores.get(cid, 0.5)

        # Normalize each signal across candidates to [0, 1] so they're
        # comparable when summed. We use min-max normalization; if all
        # values are equal, every candidate gets 0.5.
        signal_names = set()
        for scores in raw_scores.values():
            signal_names |= set(scores.keys())

        normalized: dict[str, dict[str, float]] = {c: {} for c in candidates}
        for sig in signal_names:
            vals = [raw_scores[c].get(sig, 0.0) for c in candidates]
            lo, hi = min(vals), max(vals)
            for c in candidates:
                v = raw_scores[c].get(sig, 0.0)
                normalized[c][sig] = (v - lo) / (hi - lo) if hi > lo else 0.5

        # Final score = sum of normalized signals (each weighted 1.0)
        final: dict[str, float] = {}
        for c in candidates:
            final[c] = sum(normalized[c].values())

        # Select the candidate with the highest combined score
        best_cid = max(candidates, key=lambda c: final[c])

        # Debug: print signal breakdown
        print(f"\n  Selection scores ({len(candidates)} candidates):")
        for c in sorted(candidates, key=lambda c: final[c], reverse=True)[:5]:
            parts = " + ".join(
                f"{sig}={normalized[c][sig]:.2f}" for sig in sorted(normalized[c])
            )
            marker = " <-- SELECTED" if c == best_cid else ""
            print(f"    {c}: {final[c]:.3f} = {parts}{marker}")

        return best_cid, final

    # --- Individual signals ---

    def _child_variance(self, node: TreeNode, nodes: dict[str, TreeNode]) -> float:
        """Signal (a): Variance among child scores.

        High variance => the strategy space around this node is rich and
        sensitive to choices. Low variance => the region is saturated.

        For nodes with 0-1 children, we return a high default value (1.0)
        to encourage exploring them — we simply don't know yet whether the
        region is interesting.
        """
        child_scores = [
            nodes[cid].score for cid in node.children
            if cid in nodes and nodes[cid].score is not None
        ]
        if len(child_scores) < 2:
            # Not enough children to compute variance — assume high potential
            return 1.0
        return statistics.stdev(child_scores)

    def _regret(
        self, node: TreeNode, nodes: dict[str, TreeNode], global_best: float
    ) -> float:
        """Signal (b): Regret = global_best - best_child(node).

        Nodes whose best child is far below the global best have the most
        "room to improve." Expanding them has higher expected marginal value.

        For leaf nodes (no children), regret = global_best - node.score.
        """
        child_scores = [
            nodes[cid].score for cid in node.children
            if cid in nodes and nodes[cid].score is not None
        ]
        if child_scores:
            best_child = max(child_scores) if (self.task and self.task.higher_is_better) else min(child_scores)
        else:
            # Leaf node — use the node's own score
            best_child = node.score if node.score is not None else 0.0

        return abs(global_best - best_child)

    def _coverage_gap(
        self,
        node: TreeNode,
        nodes: dict[str, TreeNode],
        all_families_used: set[str],
    ) -> float:
        """Signal (d): Strategy coverage gap (QD-inspired).

        Count how many strategy families have NOT been tried by this node's
        children. A high count means many unexplored strategy types remain.

        All possible families = keys of STRATEGY_FAMILIES + "other".
        """
        all_possible = set(STRATEGY_FAMILIES.keys()) | {"other"}

        # Families tried by this node's children
        child_families = set()
        for cid in node.children:
            if cid in nodes and nodes[cid].strategy:
                child_families |= classify_strategy(nodes[cid].strategy)

        # Gap = families that haven't been tried from THIS node
        gap = all_possible - child_families
        # Normalize by total number of families
        return len(gap) / len(all_possible)

    def _depth_visit_bonus(self, node: TreeNode) -> float:
        """Signal (e): Prefer shallow, under-expanded nodes.

        Shallow nodes have more potential subtree beneath them. Nodes with
        fewer children have been explored less. We combine both:

            bonus = 1 / (depth * (num_children + 1) + 1)

        This gives root (depth=0) the highest bonus, and penalizes deep
        nodes that already have many children.
        """
        return 1.0 / (node.depth * (len(node.children) + 1) + 1)

    def _llm_guidance_batch(
        self,
        nodes: dict[str, TreeNode],
        candidates: list[str],
        global_best: float,
        model: "LLMClient | GuidanceClient | None" = None,
    ) -> dict[str, float]:
        """Signal (c): Ask the LLM to rate nodes on two axes.

        This is inspired by AlphaZero's PUCT prior — use domain knowledge to
        estimate which nodes are worth expanding and HOW.

        The LLM rates each candidate on two axes:
        - INTERESTINGNESS (0-1): Is this strategy direction novel/creative?
          Are there fundamentally different approaches branching from here that
          haven't been explored? High = worth EXPLORING from this node.
        - DEPTH POTENTIAL (0-1): Is this approach promising enough that going
          deeper (refining hyperparameters, adding features, ensembling) would
          yield meaningful gains? High = worth EXPLOITING this node.

        The selection score is max(interestingness, depth_potential) — we want
        to expand nodes that are promising on EITHER axis.

        The explore/exploit recommendation is stored in self._llm_mode_hint
        so _decide_mode() can use it when --use-llm-guidance is enabled:
        if interestingness > depth_potential → explore, else → exploit.

        Returns dict of node_id -> combined selection score in [0.0, 1.0].
        """
        # Build description of each candidate
        node_descriptions = []
        for i, cid in enumerate(candidates):
            n = nodes[cid]
            child_scores = [
                nodes[c].score for c in n.children
                if c in nodes and nodes[c].score is not None
            ]
            child_info = ", ".join(f"{s:.4f}" for s in child_scores) if child_scores else "none yet"
            # Also show child strategies for richer context
            child_strats = []
            for c in n.children:
                if c in nodes and nodes[c].strategy:
                    cs = nodes[c]
                    s_str = f"{cs.score:.3f}" if cs.score is not None else "FAIL"
                    child_strats.append(f"    - {cs.strategy[:80]} → {s_str}")
            child_strat_text = "\n".join(child_strats) if child_strats else "    (no children yet)"

            node_descriptions.append(
                f"Node {i+1} (id={cid}):\n"
                f"  Strategy: {n.strategy[:200]}\n"
                f"  Score: {n.score:.4f}\n"
                f"  Depth: {n.depth}, Num children: {len(n.children)}\n"
                f"  Children tried:\n{child_strat_text}"
            )

        prompt = (
            f"You are evaluating nodes in an ML strategy search tree.\n"
            f"Task: {self.task.strategy_topic if self.task else 'ML task'}\n"
            f"Global best score so far: {global_best:.4f}\n\n"
            f"Rate each node on TWO axes:\n\n"
            f"1. INTERESTINGNESS (0.0–1.0): How novel/creative is this strategy\n"
            f"   direction? Are there fundamentally different approaches branching\n"
            f"   from here that haven't been explored? A node using Random Forest\n"
            f"   that hasn't tried neural nets or SVMs yet is INTERESTING. A node\n"
            f"   whose children have already tried many model families is LESS\n"
            f"   interesting (the region is well-explored).\n\n"
            f"2. DEPTH POTENTIAL (0.0–1.0): Is this approach promising enough\n"
            f"   that going deeper (tuning hyperparameters, adding feature\n"
            f"   engineering, ensembling) would yield meaningful score gains?\n"
            f"   A node scoring 0.88 with a basic GBM has HIGH depth potential\n"
            f"   (lots of tuning room). A node scoring 0.93 with a highly-tuned\n"
            f"   ensemble has LOW depth potential (near ceiling).\n\n"
            + "\n\n".join(node_descriptions) +
            f"\n\nFor each node, output exactly this format:\n"
            f"Node <number>: interest=<score> depth=<score>\n"
            f"Example: Node 1: interest=0.7 depth=0.3"
        )

        try:
            client = model or self.llm
            resp = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            # Parse "Node 1: interest=0.7 depth=0.3"
            scores = {}
            self._llm_mode_hint = {}  # node_id -> "explore" or "exploit"

            for m in re.finditer(
                r"Node\s+(\d+):\s*interest(?:ingness)?=([0-9.]+)\s*"
                r"depth(?:\s*potential)?=([0-9.]+)",
                resp
            ):
                idx = int(m.group(1)) - 1
                interest = min(max(float(m.group(2)), 0.0), 1.0)
                depth_pot = min(max(float(m.group(3)), 0.0), 1.0)
                if 0 <= idx < len(candidates):
                    cid = candidates[idx]
                    # Apply child-count decay: nodes that have already been
                    # expanded many times should become less attractive.
                    # Without this, the LLM gives sticky scores (e.g., always
                    # rates neural-net nodes as interest=0.9) and the same
                    # node gets selected repeatedly.
                    n_children = len(nodes[cid].children)
                    decay = 1.0 / (1.0 + 0.3 * n_children)
                    raw_score = max(interest, depth_pot)
                    scores[cid] = raw_score * decay
                    # Mode hint: explore if more interesting, exploit if more depth potential
                    self._llm_mode_hint[cid] = "explore" if interest > depth_pot else "exploit"
                    print(f"    LLM: {cid} interest={interest:.2f} depth={depth_pot:.2f} → {self._llm_mode_hint[cid]}")

            # Fill defaults
            for cid in candidates:
                if cid not in scores:
                    scores[cid] = 0.5
                if cid not in self._llm_mode_hint:
                    self._llm_mode_hint[cid] = "explore"
            return scores

        except Exception as e:
            print(f"  WARNING: LLM guidance failed: {e}")
            self._llm_mode_hint = {cid: "explore" for cid in candidates}
            return {cid: 0.5 for cid in candidates}


# ---------------------------------------------------------------------------
# UCB Selection Policy — classic MCTS Upper Confidence Bound
# ---------------------------------------------------------------------------

class UCBSelectionPolicy:
    """Classic UCB1 selection for MCTS-style tree search.

    score(node) = Q(node) + C * sqrt(ln(N) / (n_children + 1))

    where:
        Q(node)    = exploitation value = best_child_score / global_best
                     (1.0 means this node's best child equals the global best)
        C          = exploration constant (sqrt(2) ≈ 1.41 is standard)
        N          = total expansions so far across the tree
        n_children = how many times this node has been expanded

    The explore/exploit mode is coupled to the selection:
        - If exploration_bonus > exploitation_value → explore (Tail VS)
        - Otherwise → exploit (Local VS)
    """

    def __init__(self, c: float = 1.41, task_profile: TaskProfile | None = None):
        self.c = c
        self.task = task_profile

    def select(
        self,
        nodes: dict[str, TreeNode],
        candidates: list[str],
        global_best: float,
        total_expansions: int = 1,
    ) -> tuple[str, dict[str, float], dict[str, str]]:
        """Select the best node to expand.

        Returns:
            (selected_node_id, debug_scores, mode_hints)
            mode_hints maps node_id -> "explore" or "exploit"
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            mode = self._mode_for_node(nodes[candidates[0]], global_best, total_expansions)
            return candidates[0], {candidates[0]: 1.0}, {candidates[0]: mode}

        N = max(total_expansions, 1)
        ln_N = math.log(N)

        scores = {}
        mode_hints = {}
        debug_parts = {}

        for cid in candidates:
            node = nodes[cid]
            n_children = len(node.children)

            # Q(node) = exploitation value
            best_child_score = self._best_descendant_score(node, nodes)
            if global_best > 0:
                q_value = best_child_score / global_best
            else:
                q_value = best_child_score

            # Exploration bonus
            explore_bonus = self.c * math.sqrt(ln_N / (n_children + 1))

            scores[cid] = q_value + explore_bonus
            debug_parts[cid] = (q_value, explore_bonus)

            # Mode: explore if the exploration term dominates
            mode_hints[cid] = "explore" if explore_bonus > q_value else "exploit"

        best_cid = max(candidates, key=lambda c: scores[c])

        # Debug output
        print(f"\n  UCB scores ({len(candidates)} candidates, C={self.c:.2f}, N={N}):")
        for c in sorted(candidates, key=lambda c: scores[c], reverse=True)[:5]:
            q, e = debug_parts[c]
            n_ch = len(nodes[c].children)
            marker = " <-- SELECTED" if c == best_cid else ""
            print(f"    {c}: {scores[c]:.3f} = Q={q:.3f} + explore={e:.3f}"
                  f"  (children={n_ch}, mode={mode_hints[c]}){marker}")

        return best_cid, scores, mode_hints

    def _best_descendant_score(self, node: TreeNode, nodes: dict[str, TreeNode]) -> float:
        """Best score among node's children, or node's own score if leaf."""
        child_scores = [
            nodes[cid].score for cid in node.children
            if cid in nodes and nodes[cid].score is not None
        ]
        if child_scores:
            return max(child_scores) if (self.task and self.task.higher_is_better) else min(child_scores)
        return node.score if node.score is not None else 0.0

    def _mode_for_node(self, node: TreeNode, global_best: float, total_expansions: int) -> str:
        """Compute mode for a single node."""
        n_children = len(node.children)
        best = self._best_descendant_score(node, {})
        q = best / global_best if global_best > 0 else best
        e = self.c * math.sqrt(math.log(max(total_expansions, 1)) / (n_children + 1))
        return "explore" if e > q else "exploit"


# ---------------------------------------------------------------------------
# Open-Ended Selection Policy — UCB + trend bonus + path commitment
# ---------------------------------------------------------------------------

class OpenEndedSelectionPolicy:
    """UCB with mechanisms for open-ended scientific exploration.

    The core insight: in scientific discovery, you sometimes need to go through
    a "valley" to reach a higher peak. A path like:
        0.8 → 0.6 → 0.66 → 0.7 → 0.5 → 0.68 → 0.98
    would be abandoned by vanilla UCB at the 0.6 step. But the direction was
    promising — it just needed more commitment.

    This policy adds two mechanisms on top of UCB:

    1. TREND BONUS: Instead of only looking at absolute score, we measure the
       improvement trajectory along the path from root to each node. If a branch
       shows positive momentum (scores improving), it gets a bonus even if its
       absolute score is below the global best.

       trend(node) = weighted average of (child_score - parent_score) along the
                     path from root to this node. Positive trend = improving.

       This means a path [0.6, 0.62, 0.66, 0.70] gets higher trend bonus than
       [0.8, 0.81, 0.80, 0.79] even though the latter has higher absolute scores.

    2. PATH COMMITMENT: When we expand a node and its child shows improvement
       over the parent (positive delta), we "commit" to this branch — the next
       expansion is forced to continue down this same branch (expand the child)
       rather than going back to global selection. This gives promising directions
       time to develop through the valley.

       Commitment continues as long as the latest child shows improvement over
       its parent, up to a maximum of K committed steps. If a child is worse
       than its parent, commitment breaks and we go back to global selection.

    The combined formula:
        score(node) = Q(node) + C * sqrt(ln(N) / (n+1)) + W_trend * trend(node)

    where W_trend controls how much weight the trend gets.
    """

    def __init__(
        self,
        c: float = 1.41,
        trend_weight: float = 0.5,
        commitment_threshold: int = 2,
        task_profile: TaskProfile | None = None,
    ):
        self.c = c
        self.trend_weight = trend_weight
        self.commitment_threshold = commitment_threshold
        self.task = task_profile

        # Path commitment state
        self._committed_branch: str | None = None  # node_id we're committed to expanding
        self._commitment_steps: int = 0  # how many committed steps taken so far

    def select(
        self,
        nodes: dict[str, TreeNode],
        candidates: list[str],
        global_best: float,
        total_expansions: int = 1,
    ) -> tuple[str, dict[str, float], dict[str, str]]:
        """Select node to expand, respecting path commitment."""
        if not candidates:
            raise ValueError("No candidates to select from")

        # --- Path commitment check ---
        # If we're committed to a branch and it's still a valid candidate,
        # continue expanding it (skip global selection).
        if self._committed_branch is not None:
            # Find the latest leaf of the committed branch
            leaf = self._find_latest_leaf(self._committed_branch, nodes)
            if leaf in candidates:
                self._commitment_steps += 1
                # Check if commitment should continue:
                # the latest child must have improved over its parent
                should_continue = self._check_improvement(leaf, nodes)
                if should_continue and self._commitment_steps < self.commitment_threshold:
                    print(f"\n  PATH COMMITMENT: step {self._commitment_steps}/{self.commitment_threshold}"
                          f" — continuing branch at {leaf}")
                    mode = "explore"  # committed branches explore to find breakthroughs
                    return leaf, {leaf: 999.0}, {leaf: mode}
                else:
                    reason = "no improvement" if not should_continue else f"reached {self.commitment_threshold} steps"
                    print(f"\n  PATH COMMITMENT ENDED: {reason}")
                    self._committed_branch = None
                    self._commitment_steps = 0

        # --- Standard UCB + trend selection ---
        if len(candidates) == 1:
            self._maybe_start_commitment(candidates[0], nodes)
            mode = self._compute_mode(nodes[candidates[0]], nodes, global_best, total_expansions)
            return candidates[0], {candidates[0]: 1.0}, {candidates[0]: mode}

        N = max(total_expansions, 1)
        ln_N = math.log(N)

        scores = {}
        mode_hints = {}
        debug_parts = {}

        for cid in candidates:
            node = nodes[cid]
            n_children = len(node.children)

            # Q(node) = exploitation value
            best_child_score = self._best_descendant_score(node, nodes)
            q_value = best_child_score / global_best if global_best > 0 else best_child_score

            # UCB exploration bonus
            explore_bonus = self.c * math.sqrt(ln_N / (n_children + 1))

            # Trend bonus: path improvement momentum
            trend = self._compute_trend(cid, nodes)
            trend_bonus = self.trend_weight * max(trend, 0.0)  # only reward positive trends

            scores[cid] = q_value + explore_bonus + trend_bonus
            debug_parts[cid] = (q_value, explore_bonus, trend_bonus, trend)

            # Mode: explore if exploration + trend dominates, exploit otherwise
            mode_hints[cid] = "explore" if (explore_bonus + trend_bonus) > q_value else "exploit"

        best_cid = max(candidates, key=lambda c: scores[c])

        # Start path commitment if the selected node's branch is trending up
        self._maybe_start_commitment(best_cid, nodes)

        # Debug output
        print(f"\n  Open-ended scores ({len(candidates)} candidates, C={self.c:.2f}, "
              f"W_trend={self.trend_weight:.2f}, N={N}):")
        for c in sorted(candidates, key=lambda c: scores[c], reverse=True)[:5]:
            q, e, t, raw_t = debug_parts[c]
            n_ch = len(nodes[c].children)
            marker = " <-- SELECTED" if c == best_cid else ""
            commit_marker = " [COMMITTED]" if c == self._committed_branch else ""
            print(f"    {c}: {scores[c]:.3f} = Q={q:.3f} + explore={e:.3f} + trend={t:.3f}"
                  f"  (raw_trend={raw_t:+.3f}, children={n_ch}, mode={mode_hints[c]}){marker}{commit_marker}")

        return best_cid, scores, mode_hints

    def _compute_trend(self, node_id: str, nodes: dict[str, TreeNode]) -> float:
        """Compute the improvement trend along the path from root to this node.

        We walk up from node to root, collecting (parent_score, child_score)
        pairs, then compute a weighted average of the deltas. More recent
        deltas (closer to the leaf) get higher weight.

        Returns a float: positive = improving, negative = degrading, 0 = flat.
        The value is normalized by global_best to keep it scale-independent.
        """
        # Collect path from root to node
        path_scores = []
        current = nodes.get(node_id)
        while current is not None:
            if current.score is not None:
                path_scores.append(current.score)
            current = nodes.get(current.parent_id) if current.parent_id else None

        path_scores.reverse()  # root → ... → node

        if len(path_scores) < 2:
            return 0.0

        # Compute weighted deltas (recency-weighted: later deltas count more)
        deltas = []
        weights = []
        for i in range(1, len(path_scores)):
            delta = path_scores[i] - path_scores[i - 1]
            weight = i  # linear recency weighting: last delta has highest weight
            deltas.append(delta)
            weights.append(weight)

        if not weights:
            return 0.0

        total_weight = sum(weights)
        weighted_trend = sum(d * w for d, w in zip(deltas, weights)) / total_weight

        return weighted_trend

    def _check_improvement(self, node_id: str, nodes: dict[str, TreeNode]) -> bool:
        """Check if this node improved over its parent."""
        node = nodes.get(node_id)
        if node is None or node.parent_id is None:
            return False
        parent = nodes.get(node.parent_id)
        if parent is None:
            return False
        if node.score is None or parent.score is None:
            return False
        if self.task and self.task.higher_is_better:
            return node.score >= parent.score
        else:
            return node.score <= parent.score

    def _find_latest_leaf(self, branch_root: str, nodes: dict[str, TreeNode]) -> str:
        """Find the deepest descendant of branch_root that has been most recently added."""
        current = branch_root
        while True:
            node = nodes.get(current)
            if node is None or not node.children:
                return current
            # Follow the most recent child (highest index = latest created)
            valid_children = [c for c in node.children if c in nodes]
            if not valid_children:
                return current
            current = valid_children[-1]  # last child = most recently created

    def _maybe_start_commitment(self, node_id: str, nodes: dict[str, TreeNode]) -> None:
        """Start path commitment if the selected node's branch shows positive trend."""
        trend = self._compute_trend(node_id, nodes)
        if trend > 0.0 and self._committed_branch is None:
            self._committed_branch = node_id
            self._commitment_steps = 0
            print(f"  Starting PATH COMMITMENT to branch {node_id} (trend={trend:+.3f})")

    def _best_descendant_score(self, node: TreeNode, nodes: dict[str, TreeNode]) -> float:
        """Best score among node's children, or node's own score if leaf."""
        child_scores = [
            nodes[cid].score for cid in node.children
            if cid in nodes and nodes[cid].score is not None
        ]
        if child_scores:
            return max(child_scores) if (self.task and self.task.higher_is_better) else min(child_scores)
        return node.score if node.score is not None else 0.0

    def _compute_mode(self, node, nodes, global_best, total_expansions):
        n_children = len(node.children)
        best = self._best_descendant_score(node, nodes)
        q = best / global_best if global_best > 0 else best
        e = self.c * math.sqrt(math.log(max(total_expansions, 1)) / (n_children + 1))
        t = self.trend_weight * max(self._compute_trend(node.node_id, nodes), 0.0)
        return "explore" if (e + t) > q else "exploit"


class SoftmaxSelectionPolicy:
    """Softmax sampling over composite score (ALMA-inspired).

    Instead of argmax selection, nodes are sampled with probability
    proportional to softmax(score / tau). This naturally prevents
    chain degeneration because even non-optimal nodes get expanded.

    Composite score:
        score(node) = Q + C*sqrt(ln(N)/(n+1)) + W_trend*trend - alpha*log(1+visits)

    The visit penalty (-alpha * log(1 + visits)) ensures that repeatedly
    expanded nodes become less attractive, forcing the tree to branch.
    """

    def __init__(
        self,
        c: float = 1.41,
        trend_weight: float = 0.5,
        tau: float = 0.5,
        visit_alpha: float = 0.5,
        task_profile: TaskProfile | None = None,
    ):
        self.c = c
        self.trend_weight = trend_weight
        self.tau = tau
        self.visit_alpha = visit_alpha
        self.task = task_profile

    def select(
        self,
        nodes: dict[str, TreeNode],
        candidates: list[str],
        global_best: float,
        total_expansions: int = 1,
    ) -> tuple[str, dict[str, float], dict[str, str]]:
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            mode = self._compute_mode(nodes[candidates[0]], nodes, global_best, total_expansions)
            return candidates[0], {candidates[0]: 1.0}, {candidates[0]: mode}

        N = max(total_expansions, 1)
        ln_N = math.log(N)

        raw_scores = {}
        mode_hints = {}
        debug_parts = {}

        for cid in candidates:
            node = nodes[cid]
            n_children = len(node.children)

            # Q(node) = exploitation value
            best_child_score = self._best_descendant_score(node, nodes)
            q_value = best_child_score / global_best if global_best > 0 else best_child_score

            # UCB exploration bonus
            explore_bonus = self.c * math.sqrt(ln_N / (n_children + 1))

            # Trend bonus
            trend = self._compute_trend(cid, nodes)
            trend_bonus = self.trend_weight * max(trend, 0.0)

            # Visit penalty (ALMA-inspired)
            visit_penalty = self.visit_alpha * math.log1p(n_children)

            raw_scores[cid] = q_value + explore_bonus + trend_bonus - visit_penalty
            debug_parts[cid] = (q_value, explore_bonus, trend_bonus, visit_penalty, trend)

            mode_hints[cid] = "explore" if (explore_bonus + trend_bonus) > q_value else "exploit"

        # Softmax sampling
        score_vals = np.array([raw_scores[c] for c in candidates])
        logits = score_vals / max(self.tau, 1e-8)
        logits = logits - np.max(logits)  # numerical stability
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)

        selected_idx = np.random.choice(len(candidates), p=probs)
        selected_cid = candidates[selected_idx]

        # Debug output
        print(f"\n  Softmax scores ({len(candidates)} candidates, C={self.c:.2f}, "
              f"tau={self.tau:.2f}, alpha={self.visit_alpha:.2f}, N={N}):")
        sorted_cands = sorted(candidates, key=lambda c: raw_scores[c], reverse=True)
        for c in sorted_cands[:7]:
            q, e, t, vp, raw_t = debug_parts[c]
            n_ch = len(nodes[c].children)
            p = probs[candidates.index(c)]
            marker = " <-- SELECTED" if c == selected_cid else ""
            print(f"    {c}: {raw_scores[c]:.3f} = Q={q:.3f} + ucb={e:.3f} + trend={t:.3f}"
                  f" - visit={vp:.3f}  (p={p:.3f}, children={n_ch}, mode={mode_hints[c]}){marker}")

        return selected_cid, raw_scores, mode_hints

    def _compute_trend(self, node_id: str, nodes: dict[str, TreeNode]) -> float:
        """Compute improvement trend along path from root to node."""
        path_scores = []
        current = nodes.get(node_id)
        while current is not None:
            if current.score is not None:
                path_scores.append(current.score)
            current = nodes.get(current.parent_id) if current.parent_id else None
        path_scores.reverse()
        if len(path_scores) < 2:
            return 0.0
        deltas = []
        weights = []
        for i in range(1, len(path_scores)):
            deltas.append(path_scores[i] - path_scores[i - 1])
            weights.append(i)
        total_weight = sum(weights)
        return sum(d * w for d, w in zip(deltas, weights)) / total_weight if total_weight else 0.0

    def _best_descendant_score(self, node: TreeNode, nodes: dict[str, TreeNode]) -> float:
        child_scores = [
            nodes[cid].score for cid in node.children
            if cid in nodes and nodes[cid].score is not None
        ]
        if child_scores:
            return max(child_scores) if (self.task and self.task.higher_is_better) else min(child_scores)
        return node.score if node.score is not None else 0.0

    def _compute_mode(self, node, nodes, global_best, total_expansions):
        n_children = len(node.children)
        best = self._best_descendant_score(node, nodes)
        q = best / global_best if global_best > 0 else best
        e = self.c * math.sqrt(math.log(max(total_expansions, 1)) / (n_children + 1))
        t = self.trend_weight * max(self._compute_trend(node.node_id, nodes), 0.0)
        return "explore" if (e + t) > q else "exploit"


# ---------------------------------------------------------------------------
# Tree Context Builder — generates context for the child agent
# ---------------------------------------------------------------------------

class TreeContextBuilder:
    """Builds the information context that a child node receives when expanded.

    Two modes:
        "parent"  — child only sees parent's conversation history (same as Exp 2)
        "global"  — child also sees a structured summary of the entire tree:
                    what strategies were tried, what worked, what failed, and
                    what the best approach is. This gives the agent "meta-learning"
                    signal from the search.
    """

    def __init__(self, context_mode: str, task_profile: TaskProfile):
        self.mode = context_mode  # "parent" or "global"
        self.task = task_profile

    def build_tree_summary(self, nodes: dict[str, TreeNode], global_best_id: str) -> str:
        """Generate a natural-language summary of the entire search tree.

        This summary is injected into the child's prompt when context=global,
        giving it visibility into what has been tried and what works.
        """
        if self.mode != "global":
            return ""

        scored = [
            (nid, n) for nid, n in nodes.items()
            if n.score is not None and n.node_id != "root"
        ]
        if not scored:
            return ""

        # Sort by score (best first)
        scored.sort(key=lambda x: x[1].score, reverse=self.task.higher_is_better)

        # Build strategy summary grouped by family
        family_results: dict[str, list[tuple[str, float]]] = {}
        for nid, n in scored:
            families = classify_strategy(n.strategy)
            for fam in families:
                if fam not in family_results:
                    family_results[fam] = []
                family_results[fam].append((n.strategy[:80], n.score))

        # Format the summary
        lines = ["SEARCH TREE SUMMARY (what has been tried so far):"]
        lines.append(f"  Total nodes explored: {len(scored)}")
        lines.append(f"  Best score: {scored[0][1].score:.4f} ({scored[0][1].strategy[:60]})")
        if len(scored) > 1:
            lines.append(f"  Worst score: {scored[-1][1].score:.4f} ({scored[-1][1].strategy[:60]})")

        lines.append("\n  Results by strategy family:")
        for fam, results in sorted(family_results.items()):
            scores = [s for _, s in results]
            avg = statistics.mean(scores)
            best = max(scores) if self.task.higher_is_better else min(scores)
            example = results[0][0][:50]
            lines.append(
                f"    {fam}: {len(results)} tried, "
                f"avg={avg:.4f}, best={best:.4f} (e.g., \"{example}\")"
            )

        # Identify what works and what doesn't
        if len(scored) >= 3:
            top_strategies = [n.strategy[:60] for _, n in scored[:3]]
            bottom_strategies = [n.strategy[:60] for _, n in scored[-2:]]
            lines.append(f"\n  Top approaches: {'; '.join(top_strategies)}")
            lines.append(f"  Weak approaches: {'; '.join(bottom_strategies)}")

        # Untried families
        tried = set(family_results.keys())
        all_fams = set(STRATEGY_FAMILIES.keys())
        untried = all_fams - tried
        if untried:
            lines.append(f"\n  Strategy families NOT yet tried: {', '.join(sorted(untried))}")

        return "\n".join(lines)

    def build_sibling_summary(self, parent: TreeNode, nodes: dict[str, TreeNode]) -> str:
        """Summarize what this node's siblings tried (for past-attempt awareness).

        When exploring from a node, the child should know what siblings already
        tried so it doesn't repeat the same strategies.
        """
        sibling_info = []
        for cid in parent.children:
            if cid in nodes:
                sib = nodes[cid]
                score_str = f"{sib.score:.4f}" if sib.score is not None else "FAILED"
                sibling_info.append(f"- {sib.strategy[:80]} -> {score_str}")

        if not sibling_info:
            return ""

        return (
            "Strategies already tried from this parent node:\n"
            + "\n".join(sibling_info)
            + "\n\nGenerate a strategy that is DIFFERENT from all of the above."
        )


# ---------------------------------------------------------------------------
# Adaptive Tree Search — main MCTS-style loop
# ---------------------------------------------------------------------------

class AdaptiveTreeSearch:
    """MCTS-style tree search with adaptive node selection and explore/exploit.

    Unlike Exp 2's fixed BFS (expand all nodes at each depth), this uses a
    budget-based loop:

        for each of N budget iterations:
            1. select a node to expand (using SelectionPolicy)
            2. decide explore or exploit (using score threshold)
            3. generate a strategy (Tail VS for explore, Local VS for exploit)
            4. execute the strategy (create child, run in container, validate)
            5. update tree statistics

    This produces variable-shape trees: promising branches go deeper while
    dead branches are abandoned.

    The initial expansion (depth-1) always creates `initial_breadth` children
    from root using Tail VS to establish a diverse frontier before the
    adaptive loop kicks in.
    """

    def __init__(
        self,
        llm: LLMClient,
        container: ContainerManager,
        task_profile: TaskProfile,
        selection_policy: "SelectionPolicy | UCBSelectionPolicy | OpenEndedSelectionPolicy",
        context_builder: TreeContextBuilder,
        node_budget: int = 12,
        initial_breadth: int = 3,
        max_actions: int = 15,
        exploit_threshold_pct: float = 0.75,
        selection_strategy: str = "signals",
        output_dir: str = "outputs/adaptive_search",
        verbose: bool = False,
        reflexion: bool = True,
    ):
        self.llm = llm
        self.container = container
        self.task = task_profile
        self.policy = selection_policy
        self.context = context_builder
        self.node_budget = node_budget
        self.initial_breadth = initial_breadth
        self.max_actions = max_actions
        # exploit_threshold_pct: a node is "good enough to exploit" if its
        # score is in the top (1 - pct) of all scores. E.g., 0.75 means
        # nodes above the 75th percentile get exploited.
        self.exploit_threshold_pct = exploit_threshold_pct
        self.selection_strategy = selection_strategy  # "signals", "ucb", or "open-ended"
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.reflexion = reflexion
        self.nodes: dict[str, TreeNode] = {}
        self._child_counter: dict[str, int] = {}  # tracks next child index per parent

    def run(self) -> dict:
        """Main MCTS-style search loop."""
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "nodes").mkdir(exist_ok=True)

        # ---- Phase 1: Create baseline root ----
        print("\n" + "=" * 60)
        print("ADAPTIVE TREE SEARCH - Phase 1: Root (baseline)")
        print("=" * 60)
        root = self._execute_root()
        self.nodes[root.node_id] = root
        self._save_node(root)

        # ---- Phase 2: Initial breadth — create diverse depth-1 children ----
        # Always use Tail VS here to establish a diverse frontier.
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Initial breadth ({self.initial_breadth} children from root, Tail VS)")
        print("=" * 60)

        budget_used = 0
        for i in range(min(self.initial_breadth, self.node_budget)):
            child = self._expand_one(root.node_id, mode="explore")
            if child:
                budget_used += 1

        # ---- Phase 3: Adaptive MCTS loop ----
        remaining = self.node_budget - budget_used
        print(f"\n{'=' * 60}")
        print(f"Phase 3: Adaptive search ({remaining} expansions remaining)")
        print(f"  Selection strategy: {self.selection_strategy}")
        print("=" * 60)

        for step in range(remaining):
            print(f"\n--- Adaptive step {step + 1}/{remaining} ---")

            # Compute global best
            global_best = self._global_best_score()
            total_expansions = budget_used + step  # total nodes created so far

            # Find expandable nodes: any node with a valid score.
            # Exclude root once it has >= initial_breadth children AND there
            # are other valid candidates — root is a blank slate (score=0),
            # so regret+depth always selects it, causing all budget to be
            # spent on flat breadth-first GBM monoculture. But if all initial
            # children failed, we must fall back to root to keep searching.
            non_root = [
                nid for nid, n in self.nodes.items()
                if n.score is not None and nid != "root"
            ]
            root_node = self.nodes.get("root")
            root_saturated = (
                root_node is not None
                and len(root_node.children) >= self.initial_breadth
            )
            if non_root and root_saturated:
                candidates = non_root
            else:
                candidates = [
                    nid for nid, n in self.nodes.items()
                    if n.score is not None
                ]
            if not candidates:
                print("  No expandable nodes — stopping")
                break

            # 1. SELECT + 2. DECIDE: strategy-dependent
            if self.selection_strategy in ("ucb", "open-ended", "softmax"):
                # UCB and open-ended policies return mode hints directly
                selected_id, debug_scores, mode_hints = self.policy.select(
                    self.nodes, candidates, global_best,
                    total_expansions=total_expansions,
                )
                mode = mode_hints.get(selected_id, "explore")
            else:
                # Original signal-based selection
                selected_id, debug_scores = self.policy.select(
                    self.nodes, candidates, global_best
                )
                # Transfer LLM mode hints from SelectionPolicy to this object
                # so _decide_mode() can use them for explore/exploit decisions.
                if hasattr(self.policy, '_llm_mode_hint'):
                    self._llm_mode_hint = self.policy._llm_mode_hint
                mode = self._decide_mode(selected_id, global_best)

            # 3. EXPAND: create and execute one child
            print(f"  -> Expanding {selected_id} (mode={mode})")
            child = self._expand_one(selected_id, mode=mode)

        # ---- Results ----
        return self._compile_results(start)

    def _execute_root(self) -> TreeNode:
        """Create the baseline root node (no model execution, just snapshot)."""
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

        # Inject error analysis script before snapshot so it's baked in
        if self.reflexion:
            inject_error_analysis_script(self.container, self.task)

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

    def _decide_mode(self, node_id: str, global_best: float) -> str:
        """Decide whether to explore or exploit from this node.

        When --use-llm-guidance is enabled, the LLM rates each node on
        "interestingness" (explore) vs "depth potential" (exploit). If
        the LLM thinks the node is more interesting than deep, we explore;
        otherwise we exploit. This replaces the simple percentile threshold.

        Fallback (no LLM guidance): exploit if the node's score is in the
        top percentile of all scores, otherwise explore.
        """
        # Check if LLM guidance provided a mode hint for this node
        if hasattr(self, '_llm_mode_hint') and node_id in self._llm_mode_hint:
            hint = self._llm_mode_hint[node_id]
            print(f"  Mode decision: LLM hint → {hint}")
            return hint

        # Fallback: percentile-based threshold
        node = self.nodes[node_id]
        all_scores = [
            n.score for n in self.nodes.values()
            if n.score is not None and n.node_id != "root"
        ]

        if not all_scores or node.score is None:
            return "explore"

        # Compute the threshold as a percentile of all scores
        sorted_scores = sorted(all_scores)
        threshold_idx = int(len(sorted_scores) * self.exploit_threshold_pct)
        threshold_idx = min(threshold_idx, len(sorted_scores) - 1)

        if self.task.higher_is_better:
            threshold = sorted_scores[threshold_idx]
            return "exploit" if node.score >= threshold else "explore"
        else:
            threshold = sorted_scores[len(sorted_scores) - 1 - threshold_idx]
            return "exploit" if node.score <= threshold else "explore"

    def _expand_one(self, parent_id: str, mode: str) -> TreeNode | None:
        """Create and execute a single child from the given parent.

        Args:
            parent_id: node to expand from
            mode: "explore" (Tail VS) or "exploit" (Local VS)

        Returns:
            The new child TreeNode, or None on failure.
        """
        parent = self.nodes[parent_id]

        # Assign child index
        if parent_id not in self._child_counter:
            self._child_counter[parent_id] = 0
        child_idx = self._child_counter[parent_id]
        self._child_counter[parent_id] += 1
        child_id = f"{parent_id}_{child_idx}"

        # Tree-level reflection (if enabled and we have scored nodes)
        reflection = ""
        if self.reflexion and len(self.nodes) > 1:
            reflection = build_reflection(
                self.llm, self.nodes, parent_id, self.task,
                baseline_score=self.container.baseline_score,
            )

        # Generate strategy based on mode
        strategy_text = self._generate_strategy(parent, mode)
        if not strategy_text:
            strategy_text = f"{'Explore' if mode == 'explore' else 'Refine'} attempt {child_idx}"

        print(f"  [{child_id}] mode={mode}, strategy: {strategy_text[:80]}")

        # Restore parent workspace
        self.container.restore_snapshot(parent.snapshot_path)
        if self.task.submission_file:
            self.container.communicate(
                f"rm -f /home/agent/workspace/{self.task.submission_file}"
            )

        # Build child conversation
        child_msgs = self._build_child_messages(parent, strategy_text, mode, reflection)

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
            reflection=reflection,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        self._save_node(child)

        if score is not None:
            print(f"  [{child_id}] score={score:.4f}")
        else:
            print(f"  [{child_id}] FAILED")

        return child

    def _generate_strategy(self, parent: TreeNode, mode: str) -> str:
        """Generate a strategy for the child, using Tail VS or Local VS.

        For explore: uses STRATEGY_PROMPT_TAIL (diverse, unusual strategies).
        For exploit: uses STRATEGY_PROMPT_LOCAL (refine parent's approach).

        We generate just 1 strategy (not N) since we're expanding one child
        at a time in the MCTS loop.

        Sibling awareness: collects strategies already tried from this parent
        and passes them to the LLM so it avoids repeating the same approach.
        """
        is_baseline = len(parent.actions) == 0
        approach_summary = "No solution yet (baseline only)" if is_baseline else parent.strategy

        # Collect sibling strategies for diversity
        sibling_info = []
        for cid in parent.children:
            if cid in self.nodes:
                sib = self.nodes[cid]
                score_str = f"{sib.score:.4f}" if sib.score is not None else "FAILED"
                sibling_info.append(f"- {sib.strategy[:120]} -> {score_str}")

        sampling_mode = "tail" if mode == "explore" else "local"
        strategies = self.llm.generate_strategies(
            current_score=parent.score or 0.0,
            baseline_score=self.container.baseline_score,
            previous_approach=approach_summary,
            n=1,
            sampling_mode=sampling_mode,
            task_topic=self.task.strategy_topic,
            sibling_info=sibling_info if sibling_info else None,
        )

        return strategies[0][0] if strategies else ""

    def _build_child_messages(
        self, parent: TreeNode, strategy_text: str, mode: str,
        reflection: str = "",
    ) -> list[dict]:
        """Build the conversation messages for a child node.

        In parent-context mode: child gets parent's conversation + strategy.
        In global-context mode: child also gets a tree summary showing what
        strategies have been tried, what works, and what doesn't.
        """
        child_msgs = copy.deepcopy(parent.conversation_history)
        write_instr = self.task.branch_write_instruction
        is_from_baseline = len(parent.actions) == 0

        # --- Global context: inject tree summary ---
        tree_summary = self.context.build_tree_summary(self.nodes, "")
        sibling_summary = self.context.build_sibling_summary(parent, self.nodes)

        # Build the user message for the child
        parts = []

        # Prepend reflection if available
        if reflection:
            parts.append(
                f"REFLECTION ON PREVIOUS EXPERIMENTS:\n{reflection}"
            )

        # Add tree summary (global context only)
        if tree_summary:
            parts.append(tree_summary)

        # Add sibling summary (what was already tried from this parent)
        if sibling_summary:
            parts.append(sibling_summary)

        # Add strategy and instructions
        if is_from_baseline:
            parts.append(f"Strategy to try: {strategy_text}")
            parts.append(write_instr)
        elif mode == "exploit":
            parts.append(
                f"Your current score is {parent.score:.4f}. "
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
                f"Your current score is {parent.score:.4f}. "
                f"Try a FUNDAMENTALLY DIFFERENT approach to improve it."
            )
            parts.append(f"Strategy: {strategy_text}")
            parts.append(write_instr)

        # Add error analysis hint for classification/regression tasks
        if self.reflexion:
            hint = error_analysis_hint(self.task)
            if hint:
                parts.append(hint)

        child_msgs.append({"role": "user", "content": "\n\n".join(parts)})
        return child_msgs

    def _execute_until_validate(
        self, messages: list[dict], node_id: str
    ) -> tuple[float | None, list[dict], list[dict]]:
        """Execute actions until validate is called.

        This is the same execution logic as tree_search.py — the model
        generates commands one at a time, we execute them in the container,
        and stop when validate is called or max_actions is reached.

        Returns (score, action_log, final_messages).
        """
        action_log = []
        score = None

        for step in range(self.max_actions):
            try:
                raw = self.llm.chat(messages)
            except Exception as e:
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
            action_log.append({"action": "validate (forced)", "observation": obs[:500], "step": self.max_actions})
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

    def _global_best_score(self) -> float:
        """Get the best score across all nodes in the tree."""
        scored = [n.score for n in self.nodes.values() if n.score is not None]
        if not scored:
            return 0.0
        return max(scored) if self.task.higher_is_better else min(scored)

    def _compile_results(self, start_time: float) -> dict:
        """Compile and save final results."""
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
            "selection_strategy": self.selection_strategy,
            "best_node_id": best_id,
            "best_score": best_score,
            "baseline_score": self.container.baseline_score,
            "improvement": best_score - self.container.baseline_score,
            "total_nodes": len(self.nodes),
            "elapsed_seconds": round(elapsed, 1),
            # Capture tree structure for analysis
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

        # Print tree
        self._print_tree(best_id)
        return result

    def _save_node(self, node: TreeNode):
        """Save per-node data to JSON (for post-hoc analysis)."""
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
            "reflection": node.reflection or None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _print_tree(self, best_id: str):
        """Print ASCII tree with scores and best path."""
        print(f"\n{'=' * 70}")
        print("ADAPTIVE TREE SEARCH RESULTS")
        print(f"{'=' * 70}")

        best = self.nodes.get(best_id)
        if best and best.score is not None:
            print(
                f"Baseline: {self.container.baseline_score:.4f} | "
                f"Best: {best.score:.4f} (node: {best_id}) | "
                f"Improvement: {best.score - self.container.baseline_score:+.4f}"
            )
        print(f"Nodes explored: {len(self.nodes)}")

        # Compute max depth and tree shape stats
        depths = [n.depth for n in self.nodes.values()]
        child_counts = [len(n.children) for n in self.nodes.values()]
        print(f"Max depth: {max(depths)}, "
              f"Avg children: {statistics.mean(child_counts):.1f}")
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
# CLI — all experimental knobs exposed as command-line arguments
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Adaptive explore-exploit tree search with "
                    "configurable selection signals and information context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regret + depth signals, parent context only
  %(prog)s --use-regret --use-depth --context parent --node-budget 12

  # All signals except LLM, global context
  %(prog)s --use-variance --use-regret --use-coverage --use-depth --context global

  # LLM guidance only (expensive but informed)
  %(prog)s --use-llm-guidance --context global --node-budget 8

  # No signals = random selection (baseline)
  %(prog)s --context parent --node-budget 12
        """,
    )

    # --- Selection strategy ---
    strategy_group = parser.add_argument_group(
        "Selection strategy",
        "Choose between different node selection approaches.",
    )
    strategy_group.add_argument(
        "--selection-strategy", default="signals",
        choices=["signals", "ucb", "open-ended", "softmax"],
        help="'signals': original weighted-signal combination. "
             "'ucb': classic UCB1 (Q + C*sqrt(ln(N)/n)). "
             "'open-ended': UCB + trend bonus + path commitment. "
             "'softmax': softmax sampling with visit penalty (ALMA-inspired). "
             "(default: signals)",
    )
    strategy_group.add_argument(
        "--ucb-c", type=float, default=1.41,
        help="Exploration constant C for UCB/open-ended (default: 1.41 = sqrt(2))",
    )
    strategy_group.add_argument(
        "--trend-weight", type=float, default=0.5,
        help="Weight for trend bonus in open-ended strategy (default: 0.5)",
    )
    strategy_group.add_argument(
        "--commitment-threshold", type=int, default=2,
        help="Max consecutive committed expansions in open-ended strategy (default: 2)",
    )
    strategy_group.add_argument(
        "--softmax-tau", type=float, default=0.5,
        help="Temperature for softmax selection (lower = more greedy, default: 0.5)",
    )
    strategy_group.add_argument(
        "--visit-alpha", type=float, default=0.5,
        help="Weight for visit-count penalty in softmax strategy (default: 0.5)",
    )

    # --- Selection signal flags (for --selection-strategy signals) ---
    signals = parser.add_argument_group(
        "Selection signals (for --selection-strategy signals)",
        "Each flag enables one signal in the node selection policy. "
        "Signals are combined additively. If none are enabled, selection is random.",
    )
    signals.add_argument(
        "--use-variance", action="store_true",
        help="(a) Child variance: prefer nodes with high score variance among children",
    )
    signals.add_argument(
        "--use-regret", action="store_true",
        help="(b) Regret: prefer nodes whose best child is far below global best",
    )
    signals.add_argument(
        "--use-llm-guidance", action="store_true",
        help="(c) LLM guidance: ask model to rate interestingness vs depth potential",
    )
    signals.add_argument(
        "--guidance-model", type=str, default=None,
        help="External model for LLM guidance (e.g. 'gpt-4o-mini', 'gpt-4o'). "
             "Requires OPENAI_API_KEY in .env. If not set, uses the local vLLM model.",
    )
    signals.add_argument(
        "--use-coverage", action="store_true",
        help="(d) Strategy coverage: prefer nodes with untried strategy families (QD)",
    )
    signals.add_argument(
        "--use-depth", action="store_true",
        help="(e) Depth + visits: prefer shallow, under-expanded nodes",
    )

    # --- Information context ---
    context = parser.add_argument_group(
        "Information context",
        "Controls what information the child agent receives when expanded.",
    )
    context.add_argument(
        "--context", default="parent", choices=["parent", "global"],
        help="'parent': child sees only parent history. "
             "'global': child also sees tree summary (what worked/failed).",
    )

    # --- Search parameters ---
    search = parser.add_argument_group("Search parameters")
    search.add_argument("--node-budget", type=int, default=12,
                        help="Total number of model-executed nodes (default: 12)")
    search.add_argument("--initial-breadth", type=int, default=3,
                        help="Number of initial depth-1 children from root (default: 3)")
    search.add_argument("--max-actions", type=int, default=15,
                        help="Max actions per node before forced validate (default: 15)")
    search.add_argument("--exploit-threshold", type=float, default=0.75,
                        help="Score percentile above which to exploit (default: 0.75)")

    # --- Model and environment ---
    env = parser.add_argument_group("Model and environment")
    env.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    env.add_argument("--vllm-url", default="http://localhost:8000/v1")
    env.add_argument("--temperature", type=float, default=0.9)
    env.add_argument("--env-gpu", default="7")
    env.add_argument("--image-name", default="aigym/mlgym-agent:latest")
    env.add_argument("--task-config", default="tasks/titanic.yaml")
    env.add_argument("--output-dir", default="outputs/adaptive_search")
    env.add_argument("--verbose", action="store_true")

    # --- Reflexion ---
    env.add_argument("--reflexion", action="store_true", default=True,
                     help="Enable tree-level reflection before each expansion (default: on)")
    env.add_argument("--no-reflexion", dest="reflexion", action="store_false",
                     help="Disable tree-level reflection")

    args = parser.parse_args()

    # --- Setup ---
    task_profile = get_task_profile(args.task_config)

    print("=" * 60)
    print(f"Task: {task_profile.name}")
    print(f"Adaptive Tree Search (Experiment 3)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Node budget: {args.node_budget} (initial breadth: {args.initial_breadth})")
    print(f"Max actions/node: {args.max_actions}")
    print(f"Temperature: {args.temperature}")
    print(f"Selection strategy: {args.selection_strategy}")

    llm = LLMClient(args.vllm_url, args.model, args.temperature)
    container = ContainerManager(
        args.task_config, args.env_gpu, args.image_name, task_profile=task_profile
    )

    # Create the appropriate selection policy
    if args.selection_strategy == "ucb":
        print(f"  UCB C={args.ucb_c}")
        selection_policy = UCBSelectionPolicy(
            c=args.ucb_c,
            task_profile=task_profile,
        )
    elif args.selection_strategy == "open-ended":
        print(f"  UCB C={args.ucb_c}, trend_weight={args.trend_weight}, "
              f"commitment={args.commitment_threshold}")
        selection_policy = OpenEndedSelectionPolicy(
            c=args.ucb_c,
            trend_weight=args.trend_weight,
            commitment_threshold=args.commitment_threshold,
            task_profile=task_profile,
        )
    elif args.selection_strategy == "softmax":
        print(f"  UCB C={args.ucb_c}, trend_weight={args.trend_weight}, "
              f"tau={args.softmax_tau}, visit_alpha={args.visit_alpha}")
        selection_policy = SoftmaxSelectionPolicy(
            c=args.ucb_c,
            trend_weight=args.trend_weight,
            tau=args.softmax_tau,
            visit_alpha=args.visit_alpha,
            task_profile=task_profile,
        )
    else:
        # Original signal-based selection
        enabled = []
        if args.use_variance:  enabled.append("variance")
        if args.use_regret:    enabled.append("regret")
        if args.use_llm_guidance: enabled.append("llm_guidance")
        if args.use_coverage:  enabled.append("coverage")
        if args.use_depth:     enabled.append("depth")
        signal_desc = ", ".join(enabled) if enabled else "NONE (random selection)"
        print(f"  Signals: {signal_desc}")

        # Create guidance client if an external model is specified for LLM guidance
        guidance = None
        if args.use_llm_guidance and args.guidance_model:
            print(f"  Guidance model: {args.guidance_model} (via OpenAI API)")
            guidance = GuidanceClient(model=args.guidance_model)
        elif args.use_llm_guidance:
            print("  Guidance model: local vLLM (no --guidance-model specified)")

        selection_policy = SelectionPolicy(
            use_variance=args.use_variance,
            use_regret=args.use_regret,
            use_llm_guidance=args.use_llm_guidance,
            use_coverage=args.use_coverage,
            use_depth=args.use_depth,
            llm_client=llm if args.use_llm_guidance else None,
            guidance_client=guidance,
            task_profile=task_profile,
        )

    print(f"Context mode: {args.context}")
    print(f"Reflexion: {'enabled' if args.reflexion else 'disabled'}")
    print(f"Exploit threshold: {args.exploit_threshold:.0%}")
    print(f"Primary metric: {task_profile.primary_metric} "
          f"({'higher' if task_profile.higher_is_better else 'lower'} is better)")
    print()

    context_builder = TreeContextBuilder(args.context, task_profile)

    print("Creating MLGym container...")
    container.create()

    search = AdaptiveTreeSearch(
        llm=llm,
        container=container,
        task_profile=task_profile,
        selection_policy=selection_policy,
        context_builder=context_builder,
        node_budget=args.node_budget,
        initial_breadth=args.initial_breadth,
        max_actions=args.max_actions,
        exploit_threshold_pct=args.exploit_threshold,
        selection_strategy=args.selection_strategy,
        output_dir=args.output_dir,
        verbose=args.verbose,
        reflexion=args.reflexion,
    )

    try:
        result = search.run()
        print(f"Results saved to {args.output_dir}/result.json")
    finally:
        container.close()


if __name__ == "__main__":
    main()
