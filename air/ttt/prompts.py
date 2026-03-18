"""Scientist prompts for GRPO training.

Single-turn prompt (no inspect-then-decide two-turn flow) with
structured output fields for reward computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Structured output from the scientist
# ---------------------------------------------------------------------------

@dataclass
class ScientistOutput:
    reasoning: str = ""
    hypotheses_tested: list[str] = field(default_factory=list)
    expected_if_true: str = ""
    expected_if_false: str = ""
    strategies: list[str] = field(default_factory=list)
    chosen_idx: int = 0
    direction: str = ""
    predicted_score_range: tuple[float, float] = (0.0, 0.0)
    mode: str = "explore"
    memory_update: str = ""
    executor_guidance: str = ""
    parent_id: str = "root"
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SCIENTIST_PROMPT = """You are a research scientist guiding a junior coder to solve:

{task_description}

Task: {task_details}
Metric: {metric_name} ({direction} is better). Baseline: {baseline_score}
Budget: {budget_left}/{total_budget} nodes left.

The executor is a small 4B model. It writes code, runs it, validates. {max_actions} actions per attempt.
It CAN do: short scripts, sklearn/XGBoost/LightGBM, simple functions, hyperparameter tuning.
It CANNOT do: PyTorch, complex multi-file code, subtle debugging.

## Search Tree
{tree_view}

## Code Inspections
{code_inspections}

## Knowledge
{memory_section}

## Example Output

REASONING:
Root_0 used RandomForest and got 0.85. Root_1 tried XGBoost and got 0.88. The model choice dimension is partially explored (2 models tested) but feature engineering is untested. I should test whether adding polynomial features improves XGBoost, which would tell me if features are the bottleneck.

DIRECTION:
Use XGBoost with polynomial features (degree=2) on the top 10 numeric columns. Keep the same hyperparameters as root_1.

MODE: explore

HYPOTHESES_TESTED:
- Feature engineering (polynomial) matters more than model choice for this dataset
- XGBoost + polynomial features will exceed 0.88

EXPECTED_IF_TRUE: [0.89, 0.93]
EXPECTED_IF_FALSE: [0.84, 0.88]
PREDICTED_SCORE_RANGE: [0.85, 0.93]

MEMORY:
RandomForest (0.85) < XGBoost (0.88). Model choice gives ~0.03 gain. Testing if features give more.

## Your Turn

Think step by step, then respond in this EXACT format. Every field is REQUIRED.

REASONING:
[What worked, what failed, what is untested. Which hypothesis should you test next?]

DIRECTION:
[Specific instructions for the executor.]

MODE: explore or exploit

HYPOTHESES_TESTED:
[1-3 hypotheses this experiment tests, one per line]

EXPECTED_IF_TRUE: [low, high]
EXPECTED_IF_FALSE: [low, high]
PREDICTED_SCORE_RANGE: [low, high]

MEMORY:
[What you learned. Include scores as evidence. Write NONE if nothing new.]"""


# ---------------------------------------------------------------------------
# Auto-select nodes for code inspection
# ---------------------------------------------------------------------------

def auto_select_inspect_nodes(
    nodes: dict, max_inspect: int = 2
) -> list[str]:
    """Pick the most informative nodes for the scientist to see."""
    if len(nodes) <= 1:
        return []

    candidates = []
    for nid, n in nodes.items():
        if nid == "root":
            continue
        candidates.append(n)

    selected = []

    # 1. Most recent failure (if any)
    failures = [n for n in candidates if n.score is None and n.actions]
    if failures:
        selected.append(failures[-1].node_id)

    # 2. Best scoring node (non-root)
    scored = [n for n in candidates if n.score is not None]
    if scored:
        best = max(scored, key=lambda n: n.score)  # TODO: handle higher_is_better
        if best.node_id not in selected:
            selected.append(best.node_id)

    # 3. Most recent node (if not already selected)
    if candidates:
        latest = candidates[-1]
        if latest.node_id not in selected:
            selected.append(latest.node_id)

    return selected[:max_inspect]


def format_node_code(node_id: str, nodes: dict) -> str:
    """Format a node's executor actions for inspection."""
    if node_id not in nodes:
        return f"Node {node_id} not found."

    node = nodes[node_id]
    if not node.actions:
        return f"Node {node_id}: No actions (baseline node)."

    status_label = node.execution_status or "unknown"
    if node.error_type:
        status_label += f":{node.error_type}"

    lines = [f"=== Node {node_id} (score: {node.score}, status: {status_label}) ==="]
    for i, action in enumerate(node.actions):
        cmd = action.get("action", "")
        obs = action.get("observation", "")
        if len(obs) > 500:
            obs = obs[:500] + "\n... (truncated)"
        lines.append(f"--- Action {i} ---")
        lines.append(f"$ {cmd}")
        lines.append(obs)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_scientist_output(text: str) -> ScientistOutput:
    """Parse the scientist's structured output into a ScientistOutput.

    Handles both old format (REASONING→HYPOTHESES→DIRECTION) and new format
    (REASONING→DIRECTION→MODE→HYPOTHESES→PREDICTED→MEMORY).
    """
    out = ScientistOutput(raw_text=text)

    # All fields are parsed by searching for their header anywhere in the text.
    # This is order-independent.

    # REASONING
    m = re.search(r"REASONING:\s*\n(.*?)(?=\n(?:DIRECTION|HYPOTHESES_TESTED|STRATEGIES|MODE):|$)", text, re.DOTALL)
    if m:
        out.reasoning = m.group(1).strip()

    # DIRECTION
    m = re.search(r"DIRECTION:\s*\n(.*?)(?=\n(?:MODE|EXECUTOR_GUIDANCE|HYPOTHESES_TESTED|PREDICTED_SCORE_RANGE|MEMORY):|$)", text, re.DOTALL)
    if m:
        out.direction = m.group(1).strip()

    # MODE
    m = re.search(r"MODE:\s*(explore|exploit)", text, re.IGNORECASE)
    if m:
        out.mode = m.group(1).lower()

    # HYPOTHESES_TESTED
    m = re.search(r"HYPOTHESES_TESTED:\s*\n(.*?)(?=\n(?:EXPECTED_IF_TRUE|PREDICTED_SCORE_RANGE|MEMORY|DIRECTION|MODE):|$)", text, re.DOTALL)
    if m:
        lines = [l.strip().lstrip("- ") for l in m.group(1).strip().split("\n") if l.strip()]
        out.hypotheses_tested = lines

    # EXPECTED_IF_TRUE
    m = re.search(r"EXPECTED_IF_TRUE:\s*\[?\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\]?", text)
    if m:
        try:
            out.expected_if_true = f"[{m.group(1)}, {m.group(2)}]"
        except ValueError:
            pass

    # EXPECTED_IF_FALSE
    m = re.search(r"EXPECTED_IF_FALSE:\s*\[?\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\]?", text)
    if m:
        try:
            out.expected_if_false = f"[{m.group(1)}, {m.group(2)}]"
        except ValueError:
            pass

    # PREDICTED_SCORE_RANGE
    m = re.search(r"PREDICTED_SCORE_RANGE:\s*\[?\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\]?", text)
    if m:
        try:
            out.predicted_score_range = (float(m.group(1)), float(m.group(2)))
        except ValueError:
            pass

    # MEMORY
    m = re.search(r"MEMORY:\s*\n?(.*?)$", text, re.DOTALL)
    if m:
        mem = m.group(1).strip()
        out.memory_update = "" if mem.upper() == "NONE" else mem

    # STRATEGIES + PARENT (old format, keep for backward compat)
    strat_lines = re.findall(
        r"(\d)\.\s*(.*?)\s*(?:→|->)\s*PARENT:\s*(\S+)", text
    )
    if strat_lines:
        out.strategies = [s[1].strip() for s in strat_lines]
        chosen_m = re.search(r"CHOSEN:\s*(\d)", text)
        if chosen_m:
            out.chosen_idx = int(chosen_m.group(1)) - 1
        if strat_lines and 0 <= out.chosen_idx < len(strat_lines):
            out.parent_id = strat_lines[out.chosen_idx][2].strip().rstrip("—-– ")

    # EXECUTOR_GUIDANCE
    m = re.search(r"EXECUTOR_GUIDANCE:\s*\n(.*?)(?=\n(?:PREDICTED_SCORE_RANGE|HYPOTHESES|MODE|MEMORY):|$)", text, re.DOTALL)
    if m:
        g = m.group(1).strip()
        out.executor_guidance = "" if g.upper() == "NONE" else g

    # If no direction found, use the whole text as direction (model just output code)
    if not out.direction:
        out.direction = text.strip()[:500]

    return out
