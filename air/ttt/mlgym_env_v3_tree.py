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
import copy
import json
import logging
import os
import random
import re
import threading
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
# Tree archive for ε-persistence: keep each completed rollout's tree with
# probability ε; at the start of a new rollout, seed state["tree"] from the
# archive with probability ε. Rolling best_ever / p-threshold stay global.
# ---------------------------------------------------------------------------

class SharedTree:
    """Single growing shared tree persisted across rollouts.

    Flow:
      - Each rollout's setup_state clones the current shared tree as its starting state.
      - At rollout end, with probability ε, the rollout's newly-added nodes are merged
        into the shared tree (with renamed unique ids).
      - Dedup: if a new node's strategy is too similar to an existing node (Jaccard > 0.6),
        keep only the better-scoring one. This prevents the tree from filling with
        near-identical experiments.
      - Shared tree grows monotonically; capped at max_size (no eviction, just skip merges).
    """

    DEDUP_THRESHOLD = 0.6  # Jaccard similarity threshold for dedup

    def __init__(self, epsilon: float = 0.0, max_size: int = 500):
        self.epsilon = float(epsilon)
        self.max_size = int(max_size)
        self._tree: list[dict] | None = None
        self._merge_id = 0
        self._lock = threading.Lock()

    def _ensure_init(self):
        if self._tree is None:
            self._tree = [{
                "id": "root", "parent_id": None, "depth": 0,
                "score": None, "strategy": "", "code": None,
            }]

    @staticmethod
    def _strategy_tokens(strategy: str) -> set[str]:
        """Extract meaningful tokens from a strategy string for similarity comparison."""
        if not strategy:
            return set()
        # Remove common filler words, keep model names, techniques, numbers
        stop = {"the", "a", "an", "on", "in", "to", "of", "and", "with", "for",
                "using", "via", "from", "by", "is", "are", "be", "that", "this",
                "what", "how", "why", "fine-tune", "finetune", "mnli", "bert",
                "train", "training", "model", "base", "experiment"}
        tokens = set()
        for w in strategy.lower().replace("*", "").replace(",", " ").replace(".", " ").split():
            w = w.strip("()[]{}:;\"'")
            if len(w) > 1 and w not in stop:
                tokens.add(w)
        return tokens

    def _find_similar(self, strategy: str) -> dict | None:
        """Find an existing node with similar strategy (Jaccard > threshold).
        Returns the similar node dict, or None. Must be called with lock held."""
        new_tokens = self._strategy_tokens(strategy)
        if not new_tokens:
            return None
        for node in self._tree:
            if node["id"] == "root":
                continue
            existing_tokens = self._strategy_tokens(node.get("strategy", ""))
            if not existing_tokens:
                continue
            intersection = len(new_tokens & existing_tokens)
            union = len(new_tokens | existing_tokens)
            if union > 0 and intersection / union > self.DEDUP_THRESHOLD:
                return node
        return None

    def get_seed(self) -> list[dict]:
        with self._lock:
            self._ensure_init()
            return copy.deepcopy(self._tree)

    def size(self) -> int:
        with self._lock:
            self._ensure_init()
            return len(self._tree)

    def maybe_merge(self, completed_tree: list[dict], baseline_size: int):
        """With prob ε, merge new nodes (added after baseline_size) into shared tree.
        Deduplicates: if a similar strategy exists, keeps the better-scoring one."""
        if self.epsilon <= 0 or len(completed_tree) <= baseline_size:
            return
        if random.random() >= self.epsilon:
            return
        new_nodes = completed_tree[baseline_size:]
        with self._lock:
            self._ensure_init()
            if len(self._tree) >= self.max_size:
                return
            id_map: dict[str, str] = {}
            existing_ids = {n["id"] for n in self._tree}
            added = 0
            deduped = 0
            for node in new_nodes:
                pid = node.get("parent_id")
                if pid in existing_ids:
                    resolved = pid
                elif pid in id_map:
                    resolved = id_map[pid]
                else:
                    continue  # skip orphan

                # Dedup: check if a similar strategy already exists
                strategy = node.get("strategy", "")
                new_score = node.get("score")
                similar = self._find_similar(strategy)
                if similar is not None:
                    old_score = similar.get("score")
                    # Keep the better-scoring one
                    new_is_better = (
                        new_score is not None and
                        (old_score is None or new_score > old_score)
                    )
                    if new_is_better:
                        # Replace existing with new (better score)
                        similar["score"] = new_score
                        similar["strategy"] = strategy[:200]
                        similar["code"] = node.get("code")
                        deduped += 1
                    else:
                        deduped += 1
                    # Map the node id so children can still reference it
                    id_map[node["id"]] = similar["id"]
                    continue

                new_id = f"m{self._merge_id}"
                self._merge_id += 1
                id_map[node["id"]] = new_id
                merged = copy.deepcopy(node)
                merged["id"] = new_id
                merged["parent_id"] = resolved
                self._tree.append(merged)
                existing_ids.add(new_id)
                added += 1
            logger.info(f"[shared_tree] merged +{added}, deduped {deduped}, total={len(self._tree)}")


_SHARED_TREE = SharedTree(
    epsilon=float(os.environ.get("TREE_ARCHIVE_EPSILON", "0.0")),
    max_size=int(os.environ.get("TREE_ARCHIVE_MAX_SIZE", "500")),
)


# ---------------------------------------------------------------------------
# Tree-aware scientist prompts — match the eval-time prompt format from
# llm_guided_tree_search.py (simplified but keeps the STRATEGIES + PARENT
# + MODE structure so train and eval are consistent).
# ---------------------------------------------------------------------------

TREE_SYSTEM_PROMPT = """You are an ML research ADVISOR. You describe experiments in plain English. A separate coder executes them. You must never write code.

STRICTLY FORBIDDEN in your output:
- Any code block (no ```python, no ``` at all)
- Any import statements, class definitions, function definitions, or assignments
- Any executable syntax (no `=`, `def`, `import`, `for`, `if ... :`, `return`, etc.)
- Pseudocode, step-by-step procedures that look like code

If your output contains ``` or `import ` or `def ` anywhere, it is INVALID and will be discarded.

Given the experiment tree, propose 5 distinct strategies in natural-language sentences. One is sampled (by the given probabilities) and handed to the coder.

Output EXACTLY this format:

REASONING:
<1-3 sentences in prose>

<response>
<text><one English sentence describing a strategy. End with: "PARENT: <node_id>"></text>
<probability><number between 0 and 1></probability>
</response>
<response>
<text><another strategy>. PARENT: <node_id></text>
<probability><number></probability>
</response>
... (five total <response> blocks)

MODE: <explore or exploit>

MEMORY: <one-sentence insight, or NONE>

Rules for strategies:
- Each strategy is ONE English sentence, 15-40 words. No code, no bullet lists, no step numbering.
- Vary across axes: data (augmentation, preprocessing, sampling, relabeling), architecture (model choice, layer additions, pooling, heads), learning algorithm (loss function, regularization, optimizer, schedule, distillation), or their combinations. Do NOT propose 5 variants of the same knob.
- PARENT: root means fresh start. PARENT: <id> means build on node <id>.
- Probabilities sum to ~1.0, spread across strategies, none > 0.6.
- MODE: pick explore if the tree has few scored nodes or a promising branch is still shallow; pick exploit to refine the best known node.

Example of a valid <response> text (no code):
  "Replace the BERT-base encoder with a DistilBERT-base encoder and fine-tune with the same recipe, reducing latency and seeing if the smaller model regularizes. PARENT: root"

Example of an INVALID <response> text (contains code — do NOT do this):
  "```python\\nmodel = AutoModel.from_pretrained(...)\\n```"
"""

# ---- v12 prompts: single-experiment per episode ----------------------------

TREE_V12_SYSTEM_PROMPT = """You are an ML research ADVISOR. You describe experiments in plain English. A separate coder executes them. You must never write code.

STRICTLY FORBIDDEN: code blocks, import statements, def/class, assignments, pseudocode.
If your output contains ``` or `import ` or `def ` anywhere, it is INVALID.

Given the experiment tree, propose ONE experiment with ONE focused change. You must think deeply about what has been tried, what has NOT been tried, and where the most promising untapped potential lies.

Output format:

ANALYSIS:
<Analyze the tree thoroughly:
 - What axes have been explored? (HP tuning, architecture, data augmentation, loss functions, regularization, etc.)
 - What axes are MISSING or underexplored?
 - Which nodes show the most promise and why?
 - Which failed nodes reveal useful information (what went wrong, could it work with modifications)?
 - What is the current performance ceiling, and what type of change could break through it?
 - IMPORTANT: If all children of a node score worse than the parent, that branch may be at its ceiling. Consider trying something fundamentally different instead of adding more complexity to it.>

HYPOTHESIS:
<State your scientific hypothesis: why do you believe this specific experiment will improve over the current best? What mechanism or principle are you leveraging? Be specific — "it might help" is not a hypothesis.>

EXPERIMENT:
<A thorough description of the experiment in 2-4 sentences. Structure it as:
 1. WHAT: What ONE specific change are you making? Name exact models (e.g. "bert-large-uncased", "microsoft/deberta-v3-base"), exact techniques (e.g. "focal loss with gamma=2", "label smoothing with alpha=0.1"), exact augmentation methods (e.g. "synonym replacement using WordNet with 20% probability per token").
 2. HOW: How should the coder implement this? Describe the key modification clearly. Be specific enough that two different coders would write similar code.
 3. WHY: What do you expect to happen and what would success/failure tell us?
No code, but be as precise as a methods section in a research paper.>

PARENT: <node_id of the primary parent to build on>
COMBINES: <comma-separated node_ids whose ideas you incorporate, or NONE>

Rules:
- No code in your output. Describe everything in precise English.
- ONE CHANGE AT A TIME. Do NOT combine 3-4 techniques in one experiment. If you want to test contrastive loss, test ONLY contrastive loss. If you want to try BERT-large, try ONLY BERT-large with the same training recipe. This isolates what actually works and makes it easier for the coder to implement correctly.
- DIVERSITY: HP tuning is fine, but also explore: data (augmentation, preprocessing, sampling, relabeling), architecture (model swap like BERT-large/DeBERTa/ELECTRA, layer additions, pooling strategies, multi-head attention, classification head redesign), learning algorithm (focal loss, label smoothing, contrastive loss, knowledge distillation, curriculum learning). If the tree is dominated by one axis, deliberately explore another.
- EVOLVE: Build on high-scoring nodes by layering ONE new idea on top. Take a node that works well and change one thing about it.
- If all children of a node score worse than it, STOP building on that node. Try a completely different parent or start from root with a new approach.
- Failed nodes are valuable data — analyze WHY they failed and whether the core idea could work with a simpler implementation.
"""

TREE_V12_INITIAL_PROMPT = """[ROLE INSTRUCTIONS]
You are an ML research ADVISOR. You propose ONE experiment with ONE focused change per turn. A separate coder executes it. You must NEVER write code (no ``` blocks, no import statements, no def/class).

Think like a scientist: analyze what has been tried, identify gaps, form a hypothesis, then propose a specific experiment with ONE targeted change.

Output format:

ANALYSIS:
<Thorough analysis of the experiment tree:
 - What has been tried and what scores did they achieve?
 - What axes are MISSING or underexplored? (architecture changes? loss functions? data augmentation? regularization?)
 - Which high-scoring nodes could be improved further, and how?
 - What do the failures tell us?
 - If all children of a node score worse, that branch may be at its ceiling.>

HYPOTHESIS:
<Your scientific hypothesis: why will this ONE change improve performance? What mechanism are you leveraging?>

EXPERIMENT:
<2-4 sentences describing ONE specific change thoroughly:
 1. WHAT: Exact change — name specific models, techniques, parameter values.
 2. HOW: Key modification the coder should make.
 3. WHY: Expected outcome and what success/failure reveals.
No code, but be as precise as a methods section in a research paper.>

PARENT: <node_id>
COMBINES: <comma-separated node_ids, or NONE>

Rules:
- No code. Describe everything in precise English.
- ONE CHANGE AT A TIME. Do NOT combine 3-4 techniques. Test a model swap OR a loss change OR an augmentation OR an HP change — not all at once. This isolates what works and the coder implements it correctly.
- DIVERSITY: HP tuning, architecture changes (bert-large-uncased, microsoft/deberta-v3-base, google/electra-base-discriminator), data augmentation (synonym replacement via WordNet, back-translation), loss functions (focal loss, label smoothing, contrastive loss), regularization (R-drop, adversarial training, SWA). Explore what hasn't been tried.
- EVOLVE: Build on high-scoring nodes by changing ONE thing about them.
- If all children of a node score worse, abandon that branch. Try a new approach from root.
- Failed nodes are data — analyze why they failed and whether a simpler version could work.

[TASK]

{task_description}

## Task Context

{task_details}

Metric: {metric_name} ({direction} is better). Baseline: {baseline_score:.4f}.

## Experiment Tree

{tree_view}"""


TREE_INITIAL_PROMPT = """[ROLE INSTRUCTIONS]
You are an ML research ADVISOR. You describe experiments in plain English. A separate coder executes them. You must NEVER write code (no ``` blocks, no import statements, no def/class, no assignments).

Propose 5 distinct strategies in natural-language sentences. One is sampled (by the given probabilities) and handed to the coder.

Output EXACTLY this format:

REASONING:
<1-3 sentences in prose>

<response>
<text><one English sentence describing a strategy. End with: "PARENT: <node_id>"></text>
<probability><number between 0 and 1></probability>
</response>
<response>
<text><another strategy>. PARENT: <node_id></text>
<probability><number></probability>
</response>
... (five total <response> blocks)

MODE: <explore or exploit>

MEMORY: <one-sentence insight, or NONE>

Rules:
- Each strategy is ONE English sentence, 15-40 words. No code. No bullet lists. No step numbering.
- Vary across axes: data (augmentation, preprocessing, sampling, relabeling), architecture (model choice, layer additions, pooling, heads), learning algorithm (loss, regularization, optimizer, schedule, distillation), or combinations. Do NOT propose 5 variants of the same knob.
- PARENT: root means fresh start. PARENT: <id> means build on node <id>.
- Probabilities sum to ~1.0, spread across strategies, none > 0.6.
- MODE: pick "explore" if the tree is shallow or unexplored; pick "exploit" to refine the best node.

[TASK]

{task_description}

## Task Context

{task_details}

Metric: {metric_name} ({direction} is better). Baseline: {baseline_score:.4f}.

## Experiment Tree

{tree_view}

## Memory

{memory}"""


TREE_TURN_PROMPT = """[REMINDER]
You are the ADVISOR. Output plain-English strategies ONLY — no code, no ``` blocks, no imports.
Format: REASONING / five <response>..</response> blocks with <text> and <probability> / MODE / MEMORY.

## Last Result

{result}

## Experiment Tree

{tree_view}

## Memory

{memory}"""


# ---------------------------------------------------------------------------
# Parsers (tree-aware)
# ---------------------------------------------------------------------------

def parse_verbalized_responses(text: str) -> list[tuple[str, float]]:
    """Extract <response> blocks with <text> + <probability>. Tolerates loose
    formatting: <text> tag optional (block content used as text if absent),
    <probability> may appear immediately after </response> instead of inside.
    Returns (text, prob) list; unparseable probs default to 0 (uniform fallback).
    """
    pairs: list[tuple[str, float]] = []
    # Find all <response>...</response> blocks with their end positions so we can
    # look for a trailing <probability> after </response>.
    for m in re.finditer(r"<response>(.*?)</response>", text, re.DOTALL):
        block = m.group(1)
        # Prefer inner <text>...</text>; else use the whole block content.
        tm = re.search(r"<text>(.*?)</text>", block, re.DOTALL)
        txt = (tm.group(1) if tm else block).strip()
        if not txt:
            continue
        # Try <probability> inside the block first.
        pm = re.search(r"<probability>\s*([0-9.]+)\s*</probability>", block)
        # If not inside, look in the ~200 chars right after </response>.
        if pm is None:
            tail = text[m.end(): m.end() + 200]
            pm = re.search(r"<probability>\s*([0-9.]+)\s*</probability>", tail)
        try:
            prob = float(pm.group(1)) if pm else 0.0
        except Exception:
            prob = 0.0
        pairs.append((txt, prob))
    return pairs


def sample_verbalized(pairs: list[tuple[str, float]]) -> str | None:
    """Sample one response text proportional to probabilities. If probs are all
    zero or missing, sample uniformly. Returns the chosen text, or None if no
    valid responses were parsed."""
    if not pairs:
        return None
    texts = [t for t, _ in pairs]
    probs = [max(p, 0.0) for _, p in pairs]
    total = sum(probs)
    if total <= 0:
        return random.choice(texts)
    probs = [p / total for p in probs]
    r = random.random()
    cum = 0.0
    for t, p in zip(texts, probs):
        cum += p
        if r < cum:
            return t
    return texts[-1]


def parse_experiment_v12(text: str, valid_ids: set[str]) -> tuple[str, str, list[str]]:
    """Parse v12 single-experiment output.

    Returns (experiment_text, parent_id, combines_list).
    """
    # Extract EXPERIMENT: section (may be multi-sentence now)
    em = re.search(r"EXPERIMENT:\s*(.*?)(?:\n(?:PARENT|COMBINES|MODE|MEMORY):|\Z)",
                   text, re.DOTALL)
    experiment = em.group(1).strip() if em else ""
    # Fallback: if no EXPERIMENT section, use whole text
    if not experiment or len(experiment) < 5:
        # Try extracting from <response> blocks (backward compat)
        pairs = parse_verbalized_responses(text)
        if pairs:
            experiment = pairs[0][0]
        elif len(text.strip()) > 10:
            experiment = text.strip()[:400]

    # Extract PARENT:
    pm = re.search(r"PARENT:\s*([A-Za-z0-9_]+(?:_\d+)*)", text)
    parent_id = "root"
    if pm:
        pid = pm.group(1).strip()
        if pid == "root" or pid in valid_ids:
            parent_id = pid

    # Extract COMBINES:
    combines: list[str] = []
    cm = re.search(r"COMBINES:\s*(.*?)(?:\n(?:MODE|MEMORY|PARENT|EXPERIMENT):|\Z)",
                   text, re.DOTALL)
    if cm:
        raw = cm.group(1).strip()
        if raw.upper() != "NONE" and raw:
            for token in re.split(r"[,\s]+", raw):
                token = token.strip()
                if token in valid_ids and token != parent_id:
                    combines.append(token)

    return experiment, parent_id, combines


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
        r"(\d+)\.\s.*?(?:→|->)\s*PARENT:\s*[\"']?([A-Za-z0-9_]+(?:_\d+)*)[\"']?\s*(?:—|-|$)",
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
    any_parent = re.search(r"PARENT:\s*[\"']?([A-Za-z0-9_]+(?:_\d+)*)[\"']?", text)
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
        # Detect v12 mode from reward_scheme
        self._v12_mode = kwargs.get("reward_scheme", "").startswith("v12")
        # Force tree-aware system prompt even if caller didn't specify.
        kwargs["system_prompt"] = TREE_V12_SYSTEM_PROMPT if self._v12_mode else TREE_SYSTEM_PROMPT
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

    def _format_initial_prompt(self, tree: list[dict] | None = None) -> str:
        tp = self.task_profile
        baseline = self._baseline_score
        if tree is None:
            tree = [{"id": "root", "parent_id": None, "depth": 0,
                      "score": baseline, "strategy": "baseline (no experiment)"}]
        template = TREE_V12_INITIAL_PROMPT if self._v12_mode else TREE_INITIAL_PROMPT
        fmt_kwargs = dict(
            task_description=tp.name,
            task_details=self._scientist_task_blurb(),
            metric_name=tp.primary_metric,
            direction="higher" if tp.higher_is_better else "lower",
            baseline_score=baseline,
            tree_view=self._format_tree(tree),
        )
        if not self._v12_mode:
            fmt_kwargs["memory"] = "No experiments run yet."
            fmt_kwargs["budget_left"] = self.node_budget
            fmt_kwargs["total_budget"] = self.node_budget
        return template.format(**fmt_kwargs)

    def _scientist_task_blurb(self) -> str:
        """Executor-facing task_details has lines like 'Output your first command
        (cat baseline.py):' that confuse the scientist into emitting code.
        Strip executor directives; keep the factual task context.
        """
        raw = (self.task_details or "").strip()
        lines = raw.split("\n")
        kept = []
        bad_markers = (
            "output your first command",
            "output the first command",
            "read baseline.py first",
            "cat baseline.py",
            "modify and train",
            "modify baseline.py",
            "then run ",
            "then 'validate'",
            "then validate",
            "submission:",
            "submission format",
            "files:",
        )
        for ln in lines:
            low = ln.lower().strip()
            if not low:
                kept.append(ln); continue
            if any(m in low for m in bad_markers):
                continue
            kept.append(ln)
        blurb = "\n".join(kept).strip()
        # Remove any trailing colon lines (e.g., "Output your first command:")
        while blurb.endswith(":"):
            blurb = blurb.rsplit("\n", 1)[0].strip()
        return blurb

    def _format_tree(self, tree: list[dict]) -> str:
        """Hierarchical rendering matching eval-time format.

        Shows parent comparison ("better/worse than parent by X") and children
        count so the scientist can see which branches are worth deepening.
        Uses hierarchical node IDs (root_0, root_0_0) that encode lineage.
        """
        if not tree:
            return "  (empty tree)"
        node_by_id = {n["id"]: n for n in tree}
        children_map: dict[str, list[str]] = {}
        for n in tree:
            pid = n.get("parent_id")
            if pid is not None:
                children_map.setdefault(pid, []).append(n["id"])

        higher = tree[0].get("higher_is_better", True) if tree else True
        # Try to get it from a broader context if stored
        lines: list[str] = []

        def _render(node_id: str, prefix: str, is_last: bool, is_root: bool):
            n = node_by_id.get(node_id)
            if n is None:
                return
            sc = n.get("score")
            sc_str = f"{sc:.4f}" if sc is not None else "FAILED"
            strat = (n.get("strategy") or "").replace("\n", " ")[:80]
            n_children = len(children_map.get(node_id, []))

            # Parent comparison
            parent_note = ""
            pid = n.get("parent_id")
            if pid and pid in node_by_id and sc is not None:
                parent_sc = node_by_id[pid].get("score")
                if parent_sc is not None:
                    diff = sc - parent_sc
                    better = (diff > 0) if higher else (diff < 0)
                    parent_note = f" ({'better' if better else 'worse'} than parent by {abs(diff):.4f})"

            if is_root:
                lines.append(f"Node {node_id} [Baseline]")
                lines.append(f"  Score: {sc_str} | Children: {n_children}")
                new_prefix = "  "
            else:
                branch = "└─ " if is_last else "├─ "
                # Show fuller strategy text (up to 200 chars) so the scientist
                # can see what each node actually tried and decide whether to
                # build on it (PARENT: this_node) or try something new from root.
                strat_full = (n.get("strategy") or "").replace("\n", " ")[:200]
                lines.append(f"{prefix}{branch}Node {node_id} [{strat_full}]")
                lines.append(f"{prefix}{'   ' if is_last else '│  '}"
                             f"  Score: {sc_str} | Children: {n_children}{parent_note}")
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
            lines.append(f"\nBest score so far: {best:.4f}")
        return "\n".join(lines)

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        # Seed this rollout's tree from the shared tree (which starts as [root]
        # and grows as prior rollouts' subtrees get merged in with prob ε).
        seed = _SHARED_TREE.get_seed()
        state["tree"] = seed
        state["_baseline_size"] = len(seed)  # nodes already present; rest are "new"
        # Rebuild child_counter from seed so new ids don't collide within this rollout.
        child_counter: dict[str, int] = {}
        for n in seed:
            pid = n.get("parent_id")
            if pid is not None:
                child_counter[pid] = child_counter.get(pid, 0) + 1
        state["_child_counter"] = child_counter
        state["node_counter"] = 0  # fresh budget of node_budget new nodes to add
        if len(seed) > 1:
            logger.info(f"[tree v3] Seeded from shared tree: {len(seed)} nodes")

        # v12: update the initial prompt with the current shared tree so the
        # scientist sees prior experiments on the very first (and only) turn.
        if self._v12_mode and len(seed) > 1:
            new_text = self._format_initial_prompt(tree=seed)
            prompt = state["prompt"]
            if isinstance(prompt, list):
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        msg["content"] = new_text
                        break

        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        last_msg = messages[-1] if messages else None
        if last_msg is None:
            raw_text = ""
        elif isinstance(last_msg, dict):
            raw_text = last_msg.get("content", "") or ""
        elif hasattr(last_msg, "content"):
            raw_text = last_msg.content or ""
        else:
            raw_text = str(last_msg)

        # Debug dump: write the full scientist input+output to disk (rotating, keep last N)
        try:
            import os as _os
            dump_dir = Path(_os.environ.get("ROLLOUT_LOG_DIR", "/tmp")) / "scientist_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%H%M%S")
            # Keep only last 20 dumps to cap disk
            existing = sorted(dump_dir.glob("*.txt"))
            for old in existing[:-19]:
                try: old.unlink()
                except Exception: pass
            dump_path = dump_dir / f"scientist_{ts}_{len(existing):04d}.txt"
            with open(dump_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("INPUT MESSAGES TO SCIENTIST\n")
                f.write("=" * 80 + "\n")
                for i, m in enumerate(messages or []):
                    if isinstance(m, dict):
                        role = m.get("role", "?"); content = m.get("content", "")
                    else:
                        role = getattr(m, "role", "?"); content = getattr(m, "content", str(m))
                    f.write(f"\n--- message {i} (role={role}) ---\n{content}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("SCIENTIST RAW OUTPUT\n")
                f.write("=" * 80 + "\n")
                f.write(raw_text + "\n")
        except Exception as _e:
            logger.warning(f"[tree v3] dump failed: {_e}")

        valid_ids = {n["id"] for n in state.get("tree", [])}

        if self._v12_mode:
            # v12: single experiment with optional multi-parent
            direction, parent_id, combines = parse_experiment_v12(raw_text, valid_ids)
            # Store combines info for logging
            state["_v12_combines"] = combines
        else:
            # Verbalized-sampling path: scientist emits <response> blocks with
            # probabilities; sample one and use its <text> as the direction.
            vs_pairs = parse_verbalized_responses(raw_text)
            sampled_text = sample_verbalized(vs_pairs) if vs_pairs else None
            if sampled_text:
                direction = sampled_text
                # Extract PARENT: from the sampled response's text
                pm = re.search(r"PARENT:\s*([A-Za-z0-9_]+(?:_\d+)*)", sampled_text)
                parent_id = pm.group(1) if (pm and (pm.group(1) == "root" or pm.group(1) in valid_ids)) else "root"
            else:
                # Fallback to legacy DIRECTION/STRATEGIES format
                direction = parse_direction_v3(raw_text)
                parent_id = parse_parent(raw_text, valid_ids)
            combines = []
        memory_update = parse_memory(raw_text)

        # Validate direction
        if not direction or len(direction.strip()) < 5:
            state["last_score"] = None
            state["executor_fault_count"] += 1
            tree_view = self._format_tree(state["tree"])
            memory = "\n".join(state.get("memory_lines", [])) or "No observations yet."
            response = dict(role="user",
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

        # Look up parent's code & score so the executor can edit incrementally.
        parent_node_lookup = next(
            (n for n in state["tree"] if n["id"] == parent_id), None
        )
        parent_code = parent_node_lookup.get("code") if parent_node_lookup else None
        parent_score = parent_node_lookup.get("score") if parent_node_lookup else None

        # v12: if combining nodes, enrich the proposal with context from combined nodes
        if combines:
            combine_context = []
            for cid in combines:
                cnode = next((n for n in state["tree"] if n["id"] == cid), None)
                if cnode:
                    cs = cnode.get("score")
                    cs_str = f"{cs:.4f}" if cs is not None else "FAILED"
                    combine_context.append(
                        f"  - {cid} (score={cs_str}): {(cnode.get('strategy') or '')[:150]}"
                    )
            if combine_context:
                direction += (
                    "\n\nADDITIONAL CONTEXT — ideas to incorporate from other nodes:\n"
                    + "\n".join(combine_context)
                )

        # Execute in container (retry once on fault)
        max_retries = 2
        score = None
        feedback = ""
        executor_fault = False
        final_code = None
        for attempt in range(max_retries):
            score, feedback, executor_fault, final_code = await asyncio.to_thread(
                execute_in_container,
                proposal=direction,
                task_profile=self.task_profile,
                task_config=self.task_cfg["task_config"],
                container_image=self.task_cfg["container_image"],
                executor_url=self.executor_url,
                executor_model=self.executor_model,
                max_actions=self.max_actions,
                env_gpu=self.env_gpu,
                parent_code=parent_code,
                parent_score=parent_score,
            )
            if not executor_fault or score is not None:
                break
            logger.info(f"[tree v3] Executor fault attempt {attempt+1}, retrying")

        # Persist rollout log (tree-aware)
        self._log_rollout_tree(state, direction, score, feedback, executor_fault,
                               raw_text, parent_id)

        # Attach new node to the chosen parent with hierarchical naming
        # (root_0, root_0_0, root_1, etc.) so the scientist can see lineage in the ID.
        state["node_counter"] += 1
        parent_node = next(
            (n for n in state["tree"] if n["id"] == parent_id),
            state["tree"][0],  # fallback: root
        )
        child_counter = state.get("_child_counter", {})
        child_idx = child_counter.get(parent_id, 0)
        child_counter[parent_id] = child_idx + 1
        state["_child_counter"] = child_counter
        node_id = f"{parent_id}_{child_idx}"
        new_depth = parent_node.get("depth", 0) + 1
        state["tree"].append({
            "id": node_id,
            "parent_id": parent_id,
            "depth": new_depth,
            "score": score,
            "strategy": direction[:200],
            "code": final_code,
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
        response = dict(role="user",
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
            # With prob ε, merge this rollout's new nodes into the shared tree.
            try:
                _SHARED_TREE.maybe_merge(
                    state.get("tree", []),
                    baseline_size=state.get("_baseline_size", 1),
                )
            except Exception as e:
                logger.warning(f"[shared_tree] maybe_merge failed: {e}")
            logger.info(f"[tree v3] Budget reached ({self.node_budget}), terminating episode")

        return [response]

    def _log_rollout_tree(self, state, direction, score, feedback, executor_fault,
                          scientist_output, parent_id):
        log_dir = Path(os.environ.get("ROLLOUT_LOG_DIR", "/scratch/jarnav/rollout_logs"))
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
        # Extract reasoning/analysis + hypothesis from the raw scientist output
        analysis_match = re.search(
            r"ANALYSIS:\s*\n(.*?)(?:\n(?:HYPOTHESIS|EXPERIMENT|REASONING|STRATEGIES):|\Z)",
            scientist_output, re.DOTALL,
        )
        if analysis_match:
            entry["scientist_reasoning"] = analysis_match.group(1).strip()[:500]
        else:
            reasoning_match = re.search(
                r"REASONING:\s*\n(.*?)(?:\nSTRATEGIES:|\nEXPERIMENT:|\Z)",
                scientist_output, re.DOTALL,
            )
            if reasoning_match:
                entry["scientist_reasoning"] = reasoning_match.group(1).strip()[:300]
        hypothesis_match = re.search(
            r"HYPOTHESIS:\s*\n(.*?)(?:\n(?:EXPERIMENT|PARENT):|\Z)",
            scientist_output, re.DOTALL,
        )
        if hypothesis_match:
            entry["scientist_hypothesis"] = hypothesis_match.group(1).strip()[:300]
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
