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

The executor is a language model that writes and runs code in a container.
It can read all source files. Focus on the IDEA, not the code.

WHAT THE EXECUTOR CAN DO WELL:
- Short, self-contained Python scripts (<100 lines)
- For ML tasks: sklearn/XGBoost/LightGBM pipelines, pandas preprocessing
- For RL tasks: modifying config files, hyperparameter tuning
- Hyperparameter changes when you spell out exact values

WHAT THE EXECUTOR CANNOT DO:
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


SYSTEM_PROMPT_PYTORCH = """You are a senior research scientist mentoring a capable coder.
You guide experiment design by proposing directions for the executor to implement.

The executor is a frontier language model that modifies and runs PyTorch code in a container.
It can read, modify, and rewrite source files. It has full PyTorch/distributed training capability.

WHAT THE EXECUTOR CAN DO WELL:
- Reading and modifying existing PyTorch training scripts
- Hyperparameter tuning (learning rates, batch sizes, schedules, optimizers)
- Architecture modifications (layers, heads, dimensions, activations, normalization)
- Training loop changes (gradient accumulation, mixed precision, compilation)
- Full file rewrites when needed

WHAT THE EXECUTOR CANNOT DO:
- Download new datasets or install new packages
- Run multi-step pipelines requiring complex orchestration
- Debug environment/CUDA issues (container is pre-configured)

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


# Task types that should use the PyTorch-aware scientist prompt
_PYTORCH_TASK_TYPES = {"language_modeling", "nlp", "deep_learning"}


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
    import os as _os
    log_dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
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


# ---------------------------------------------------------------------------
# Reward schemes — canonical spec
# ---------------------------------------------------------------------------
#
# For every scheme, s is the single valid executor score for an episode
# (None ⇔ executor fault). Only "higher is better" tasks shown here; the
# env flips sign for lower-is-better by negating s and b and the thresholds
# before calling these.
#
#   v6_binary        : {-0.5 fault, 0 s<=b, +1 s>b}
#   v7_fixed_tier    : {-0.5 fault, 0 s<b,  +0.2 b<=s<=tau, +1 s>tau}   tau=0.88
#   v8_global_best   : {-0.5, 0 s<=b, +0.2 b<s<best_ever, +1 s>=best_ever}
#                      best_ever starts at b, monotonically grows, end-of-step
#                      snapshot semantics (all rollouts in a step see the
#                      snapshot taken at step-start). Persisted to disk.
#   v9_percentile    : {-0.5, 0 s<=b, +0.2 b<s<=p_t, +1 s>p_t}
#                      p_t = percentile_q(window)  (window = last N valid scores)
#                      p_t = b if |window| < warmup.
#                      Defaults: N=64, q=70, warmup=8. End-of-step snapshot
#                      semantics. Persisted to disk.
# ---------------------------------------------------------------------------

import os as _os_top
REWARD_STATE_DIR = Path(_os_top.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))


def _episode_score(state) -> tuple[float | None, bool]:
    """Return (best_score_in_episode, higher_is_better).

    Returns the BEST score across all non-root tree nodes in the episode,
    not just the last node. This is critical for tree-structured episodes
    where the best score may appear at any depth, not necessarily the last
    expansion.

    Returns None if no valid score was achieved (executor fault on all nodes).
    """
    higher = state.get("higher_is_better", True)
    any_score = state.get("any_score_achieved", False)
    if not any_score:
        return None, higher
    # Use state["best_score"] which is correctly maintained by env_response
    # as the running max (or min for lower-is-better) across all tree nodes.
    best = state.get("best_score")
    baseline = state.get("baseline_score", 0)
    # best_score starts at baseline and is updated via > (or <) comparison
    # in env_response. If it's still at baseline and any_score is True,
    # it means all scored nodes were at or below baseline.
    if best is not None:
        return float(best), higher
    return None, higher


# -- v6 (stateless, binary) --------------------------------------------------

def reward_v6_binary(parser, completion, answer, state, **kwargs) -> float:
    s, higher = _episode_score(state)
    b = state.get("baseline_score", 0)
    if s is None:
        r = -0.5
    elif (higher and s > b) or ((not higher) and s < b):
        r = 1.0
    else:
        r = 0.0
    _log_reward("v6_binary", r, s, b, s is not None, state)
    return r


# Backwards-compat alias for old config files still referencing discrete_reward_v2.
discrete_reward_v2 = reward_v6_binary


# -- v7 (stateless, fixed-tier, tau=0.88) ------------------------------------

V7_TAU = 0.88

def reward_v7_fixed_tier(parser, completion, answer, state, **kwargs) -> float:
    return reward_v7_fixed_tier_param(parser, completion, answer, state, tau=V7_TAU, **kwargs)


def reward_v7_fixed_tier_param(parser, completion, answer, state, tau: float = V7_TAU, **kwargs) -> float:
    s, higher = _episode_score(state)
    b = state.get("baseline_score", 0)
    if s is None:
        r = -0.5
    elif higher:
        if s < b:           r = 0.0
        elif s <= tau:      r = 0.2
        else:               r = 1.0
    else:
        # For lower-is-better: tau is the target val_loss (lower = better).
        # Score must be below tau to get full reward.
        if s > b:           r = 0.0
        elif s >= tau:      r = 0.2
        else:               r = 1.0
    _log_reward("v7_fixed_tier", r, s, b, s is not None, state, extra={"tau": tau})
    return r


# Legacy alias (old configs reference this name)
discrete_reward_v7_fixed_tier = reward_v7_fixed_tier


# -- v8 (stateful global best_ever, end-of-step snapshot, persistent) --------

class RewardV8GlobalBest:
    """Reward with a monotonically growing best_ever threshold.

    End-of-step snapshot semantics: within a training step of batch_size
    rollouts, all rollouts are scored against the best_ever that was
    committed at the previous step boundary. After all batch_size rollouts
    in the step have been scored, best_ever is updated to
    max(best_ever, max of this step's valid scores).
    """

    # PRIME-RL/verifiers inspects reward_fn.__name__ for logging/tracking.
    # Plain functions have this, but class instances don't by default —
    # without this, every rollout is marked "Rollout failed: AttributeError".
    __name__ = "reward_v8_global_best"
    __qualname__ = "reward_v8_global_best"

    def __init__(self, b: float, batch_size: int, task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.batch_size = int(batch_size)
        self.task = task
        self.best_ever = float(b)
        self._pending: list[float] = []
        self._counter = 0
        self.persist_path = persist_path or (REWARD_STATE_DIR / f"reward_state_v8_{task}.json")
        self._load()

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                self.best_ever = float(d.get("best_ever", self.b))
                logger.info(f"[v8] loaded best_ever={self.best_ever:.4f} from {self.persist_path}")
        except Exception as e:
            logger.warning(f"[v8] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"best_ever": self.best_ever, "baseline": self.b}, f)
        except Exception as e:
            logger.warning(f"[v8] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s, higher = _episode_score(state)
        b = state.get("baseline_score", self.b)

        # Compute against the committed snapshot.
        if s is None:
            r = -0.5
        elif higher:
            if s <= b:                     r = 0.0
            elif s < self.best_ever:       r = 0.2
            else:                          r = 1.0   # s >= best_ever
        else:
            if s >= b:                     r = 0.0
            elif s > self.best_ever:       r = 0.2   # note: for lower-is-better, "best" is the min; < is "better than"
            else:                          r = 1.0

        # Buffer this score; commit at end of step.
        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            if self._pending:
                if higher:
                    self.best_ever = max(self.best_ever, max(self._pending))
                else:
                    self.best_ever = min(self.best_ever, min(self._pending))
            self._pending = []
            self._counter = 0
            self._save()

        _log_reward("v8_global_best", r, s, b, s is not None, state,
                    extra={"best_ever_snapshot": self.best_ever,
                           "pending_in_batch": len(self._pending)})
        return r


# -- v9 (stateful percentile, end-of-step snapshot, persistent) --------------

class RewardV9Percentile:
    """Reward with a p_q threshold over a rolling window of recent valid scores.

    End-of-step snapshot semantics: all rollouts within a step are scored
    against the window frozen at step-start. After the step is complete,
    the window is extended with this step's valid scores (FIFO, cap N).
    """

    # See note on RewardV8GlobalBest.__name__.
    __name__ = "reward_v9_percentile"
    __qualname__ = "reward_v9_percentile"

    def __init__(self, b: float, N: int, q: int, warmup: int, batch_size: int,
                 task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.N = int(N)
        self.q = int(q)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.task = task
        self.window: deque[float] = deque(maxlen=self.N)
        self._pending: list[float] = []
        self._counter = 0
        self._cached_p: float | None = None
        self.persist_path = persist_path or (REWARD_STATE_DIR / f"reward_state_v9_{task}.json")
        self._load()
        self._recompute_threshold()

    def _recompute_threshold(self):
        if len(self.window) < self.warmup:
            self._cached_p = self.b
        else:
            import numpy as np
            self._cached_p = float(np.percentile(list(self.window), self.q))

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                w = d.get("window", [])
                self.window = deque([float(x) for x in w[-self.N:]], maxlen=self.N)
                logger.info(f"[v9] loaded window of {len(self.window)} scores from {self.persist_path}")
        except Exception as e:
            logger.warning(f"[v9] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"window": list(self.window), "baseline": self.b,
                           "N": self.N, "q": self.q, "warmup": self.warmup}, f)
        except Exception as e:
            logger.warning(f"[v9] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s, higher = _episode_score(state)
        b = state.get("baseline_score", self.b)
        p = self._cached_p if self._cached_p is not None else self.b

        if s is None:
            r = -0.5
        elif higher:
            if s <= b:          r = 0.0
            elif s <= p:        r = 0.2
            else:               r = 1.0
        else:
            # lower-is-better: "above p" means "s < p_lowertail"; use (100-q) percentile
            if s >= b:          r = 0.0
            elif s >= p:        r = 0.2
            else:               r = 1.0

        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            for ps in self._pending:
                self.window.append(ps)
            self._pending = []
            self._counter = 0
            self._recompute_threshold()
            self._save()

        _log_reward("v9_percentile", r, s, b, s is not None, state,
                    extra={"p_threshold_snapshot": p,
                           "window_size": len(self.window),
                           "pending_in_batch": len(self._pending)})
        return r


# -- v10 hybrid (percentile middle tier + global best_ever top tier) --------

class RewardV10Hybrid:
    __name__ = "v10_hybrid"
    """Reward scheme combining v9 percentile bar with v8 global-best top tier.

    Tiers (higher_is_better):
      * fault             : -0.5
      * s <= baseline     :  0.0
      * baseline < s <= p75: 0.0   (below percentile threshold)
      * p75 < s <= best_ever: 0.4  (middle tier)
      * s > best_ever       : 1.0  (new global max)

    best_ever is monotonic; grows only when a rollout score exceeds it.
    """

    def __init__(self, b: float, N: int, q: int, warmup: int, batch_size: int,
                 task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.N = int(N)
        self.q = int(q)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.task = task
        self.window: deque[float] = deque(maxlen=self.N)
        self._pending: list[float] = []
        self._counter = 0
        self._cached_p: float | None = None
        self.best_ever = float(b)
        import os as _os
        _dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
        self.persist_path = persist_path or (_dir / f"reward_state_v10_{task}.json")
        self._load()
        self._recompute_threshold()

    def _recompute_threshold(self):
        if len(self.window) < self.warmup:
            self._cached_p = self.b
        else:
            import numpy as np
            self._cached_p = float(np.percentile(list(self.window), self.q))

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                w = d.get("window", [])
                self.window = deque([float(x) for x in w[-self.N:]], maxlen=self.N)
                self.best_ever = float(d.get("best_ever", self.b))
                logger.info(f"[v10] loaded window={len(self.window)} best_ever={self.best_ever:.4f}")
        except Exception as e:
            logger.warning(f"[v10] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"window": list(self.window), "best_ever": self.best_ever,
                           "baseline": self.b, "N": self.N, "q": self.q,
                           "warmup": self.warmup}, f)
        except Exception as e:
            logger.warning(f"[v10] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s, higher = _episode_score(state)
        b = state.get("baseline_score", self.b)
        p = self._cached_p if self._cached_p is not None else self.b
        be = self.best_ever

        if s is None:
            r = -0.5
        elif higher:
            if s <= b:          r = 0.0
            elif s <= p:        r = 0.2
            elif s <= be:       r = 0.4
            else:               r = 1.0
        else:
            if s >= b:          r = 0.0
            elif s >= p:        r = 0.2
            elif s >= be:       r = 0.4
            else:               r = 1.0

        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            for ps in self._pending:
                self.window.append(ps)
                if higher and ps > self.best_ever:
                    self.best_ever = ps
                elif not higher and ps < self.best_ever:
                    self.best_ever = ps
            self._pending = []
            self._counter = 0
            self._recompute_threshold()
            self._save()

        _log_reward("v10_hybrid", r, s, b, s is not None, state,
                    extra={"p_threshold_snapshot": p,
                           "best_ever_snapshot": be,
                           "window_size": len(self.window),
                           "pending_in_batch": len(self._pending)})
        return r


# -- v11 tiered (below-baseline penalty + stronger mid + best_ever top) ----

class RewardV11Tiered:
    __name__ = "v11_tiered"
    """Reward scheme with below-baseline penalty and stronger mid-tier.

    Tiers (higher_is_better):
      * fault               : -0.5
      * s < baseline        : -0.2   (regression penalty)
      * baseline <= s <= p70:  0.0
      * p70 < s < best_ever :  0.2
      * s >= best_ever      :  1.0
    """

    def __init__(self, b: float, N: int, q: int, warmup: int, batch_size: int,
                 task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.N = int(N)
        self.q = int(q)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.task = task
        self.window: deque[float] = deque(maxlen=self.N)
        self._pending: list[float] = []
        self._counter = 0
        self._cached_p: float | None = None
        self.best_ever = float(b)
        import os as _os
        _dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
        self.persist_path = persist_path or (_dir / f"reward_state_v11_{task}.json")
        self._load()
        self._recompute_threshold()

    def _recompute_threshold(self):
        if len(self.window) < self.warmup:
            self._cached_p = self.b
        else:
            import numpy as np
            self._cached_p = float(np.percentile(list(self.window), self.q))

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                w = d.get("window", [])
                self.window = deque([float(x) for x in w[-self.N:]], maxlen=self.N)
                self.best_ever = float(d.get("best_ever", self.b))
                logger.info(f"[v11] loaded window={len(self.window)} best_ever={self.best_ever:.4f}")
        except Exception as e:
            logger.warning(f"[v11] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"window": list(self.window), "best_ever": self.best_ever,
                           "baseline": self.b, "N": self.N, "q": self.q,
                           "warmup": self.warmup}, f)
        except Exception as e:
            logger.warning(f"[v11] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s, higher = _episode_score(state)
        b = state.get("baseline_score", self.b)
        p = self._cached_p if self._cached_p is not None else self.b
        be = self.best_ever

        if s is None:
            r = -0.5
        elif higher:
            if s < b:           r = -0.2
            elif s <= p:        r = 0.0
            elif s < be:        r = 0.2
            else:               r = 1.0
        else:
            if s > b:           r = -0.2
            elif s >= p:        r = 0.0
            elif s > be:        r = 0.2
            else:               r = 1.0

        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            for ps in self._pending:
                self.window.append(ps)
                if higher and ps > self.best_ever:
                    self.best_ever = ps
                elif not higher and ps < self.best_ever:
                    self.best_ever = ps
            self._pending = []
            self._counter = 0
            self._recompute_threshold()
            self._save()

        _log_reward("v11_tiered", r, s, b, s is not None, state,
                    extra={"p_threshold_snapshot": p,
                           "best_ever_snapshot": be,
                           "window_size": len(self.window),
                           "pending_in_batch": len(self._pending)})
        return r


# -- v12 individual (per-node reward, not tree-max) -------------------------

class RewardV12Individual:
    __name__ = "v12_individual"
    """Per-node reward using state['last_score'], 4-tier with moving p70.

    Tiers (higher_is_better):
      * fault (code fails)   : -0.5
      * s <= baseline        : -0.2
      * baseline < s <= p80  :  0.0
      * s > p80              :  1.0

    No best_ever gate — the p80 threshold rises naturally as scores improve,
    keeping the +1.0 reward meaningful throughout training.
    """

    def __init__(self, b: float, N: int, q: int, warmup: int, batch_size: int,
                 task: str, persist_path: Path | None = None):
        self.b = float(b)
        self.N = int(N)
        self.q = int(q)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.task = task
        self.window: deque[float] = deque(maxlen=self.N)
        self._pending: list[float] = []
        self._counter = 0
        self._cached_p: float | None = None
        import os as _os
        _dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
        self.persist_path = persist_path or (_dir / f"reward_state_v12_{task}.json")
        self._load()
        self._recompute_threshold()

    def _recompute_threshold(self):
        if len(self.window) < self.warmup:
            self._cached_p = self.b
        else:
            import numpy as np
            self._cached_p = float(np.percentile(list(self.window), self.q))

    def _load(self):
        try:
            if self.persist_path.exists():
                with open(self.persist_path) as f:
                    d = json.load(f)
                w = d.get("window", [])
                self.window = deque([float(x) for x in w[-self.N:]], maxlen=self.N)
                logger.info(f"[v12] loaded window={len(self.window)}")
        except Exception as e:
            logger.warning(f"[v12] could not load state: {e}")

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump({"window": list(self.window),
                           "baseline": self.b, "N": self.N, "q": self.q,
                           "warmup": self.warmup}, f)
        except Exception as e:
            logger.warning(f"[v12] could not save state: {e}")

    def __call__(self, parser, completion, answer, state, **kwargs) -> float:
        s = state.get("last_score")
        higher = state.get("higher_is_better", True)
        b = state.get("baseline_score", self.b)
        p = self._cached_p if self._cached_p is not None else self.b

        if s is None:
            r = -0.5
        elif higher:
            if s <= b:          r = -0.2
            elif s <= p:        r = 0.0
            else:               r = 1.0
        else:
            if s >= b:          r = -0.2
            elif s >= p:        r = 0.0
            else:               r = 1.0

        if s is not None:
            self._pending.append(s)
        self._counter += 1
        if self._counter >= self.batch_size:
            for ps in self._pending:
                self.window.append(ps)
            self._pending = []
            self._counter = 0
            self._recompute_threshold()
            self._save()

        _log_reward("v12_individual", r, s, b, s is not None, state,
                    extra={"p_threshold_snapshot": p,
                           "window_size": len(self.window),
                           "pending_in_batch": len(self._pending)})
        return r


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
        reward_scheme: str = "v6_binary",
        reward_batch_size: int = 8,
        v7_tau: float | None = None,
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

        # Select scientist prompt based on task type
        if system_prompt is None:
            if getattr(tp, "task_type", "") in _PYTORCH_TASK_TYPES:
                system_prompt = SYSTEM_PROMPT_PYTORCH
            else:
                system_prompt = SYSTEM_PROMPT

        # Parser extracts DIRECTION from freeform text
        parser = parser or vf.XMLParser(fields=["proposal"], answer_field="proposal")

        # Select reward function based on reward_scheme.
        # Stateless schemes use plain functions; stateful schemes instantiate a
        # class tied to this env instance so state persists across rollouts.
        self.reward_scheme = reward_scheme
        self.reward_batch_size = reward_batch_size
        logger.info(f"Using reward scheme: {reward_scheme} (batch_size={reward_batch_size})")

        if reward_scheme in ("v6_binary", "v6_moving_best"):
            reward_callable = reward_v6_binary
            self._reward_obj = None
        elif reward_scheme == "v7_fixed_tier":
            tau = v7_tau if v7_tau is not None else V7_TAU
            self._v7_tau = tau
            logger.info(f"v7 tau={tau}")
            def _v7_with_tau(parser, completion, answer, state, **kw):
                return reward_v7_fixed_tier_param(parser, completion, answer, state, tau=tau, **kw)
            reward_callable = _v7_with_tau
            self._reward_obj = None
        elif reward_scheme == "v8_global_best":
            self._reward_obj = RewardV8GlobalBest(
                b=self._baseline_score,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        elif reward_scheme == "v9_percentile":
            self._reward_obj = RewardV9Percentile(
                b=self._baseline_score,
                N=64, q=70, warmup=8,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        elif reward_scheme == "v11_tiered":
            self._reward_obj = RewardV11Tiered(
                b=self._baseline_score,
                N=64, q=70, warmup=8,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        elif reward_scheme == "v12_individual":
            self._reward_obj = RewardV12Individual(
                b=self._baseline_score,
                N=64, q=80, warmup=8,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        elif reward_scheme == "v10_hybrid":
            # -0.5 fault / 0 s<=b / 0 baseline<s<=p75 / 0.4 p75<s<=best_ever / 1.0 s>best_ever
            self._reward_obj = RewardV10Hybrid(
                b=self._baseline_score,
                N=64, q=75, warmup=8,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        elif reward_scheme == "v8_percentile":
            # Legacy alias for old v8 percentile-based scheme → map to v9.
            logger.warning("reward_scheme='v8_percentile' is deprecated; use 'v9_percentile'")
            self._reward_obj = RewardV9Percentile(
                b=self._baseline_score,
                N=64, q=70, warmup=8,
                batch_size=reward_batch_size,
                task=self.task_name,
            )
            reward_callable = self._reward_obj
        else:
            raise ValueError(
                f"Unknown reward_scheme={reward_scheme}. "
                "Options: v6_binary, v7_fixed_tier, v8_global_best, v9_percentile"
            )

        rubric = rubric or vf.Rubric(parser=parser)
        rubric.add_reward_func(reward_callable, weight=1.0)

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
        import time, os as _os
        log_dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
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
