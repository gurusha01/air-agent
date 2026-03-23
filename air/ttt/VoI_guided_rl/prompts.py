"""
Prompts for VoI-guided hypothesis-driven tree search.

The scientist outputs structured proposals with:
- Node type (explore / validate / challenge)
- Hypothesis (new for explore, existing ID for validate/challenge)
- Experiment description
- Prediction interval [lower, upper]
"""

SCIENTIST_SYSTEM = """You are a senior ML research scientist. You solve problems through \
hypothesis-driven experimentation. You maintain a structured mental model (thought.md) \
of what you know about the problem, and every experiment you propose must be tied to a \
specific hypothesis with a sharp prediction.

You MUST ground your reasoning in the actual task description. Never apply reasoning \
from one domain to another."""


SCIENTIST_PROMPT = """## Task Description
{task_description}

## Task Details
{task_details}

Metric: {metric_name} ({direction} is better)
Baseline: {baseline_score}

## Your Mental Model
{thought_md}

## Experiment Tree (so far)
{tree_view}

## Budget
{budget_left} nodes remaining out of {total_budget}.

## Rules

You must propose ONE experiment. Your output MUST follow this exact format:

```
type: [explore|validate|challenge]
hypothesis: [NEW hypothesis text for explore, OR existing H_id for validate/challenge]
experiment: [what to run — specific enough for a junior coder to implement]
prediction: [lower_bound]-[upper_bound] (the score or improvement you expect)
thought_update: [what to add to thought.md based on your reasoning]
```

Node type rules:
- **explore**: Introduce a NEW hypothesis about the problem. Rewarded for hypotheses \
that would change your experiment plan if resolved.
- **validate**: Test an existing unvalidated hypothesis (give its H_id). You get \
{validate_budget} validate attempts per hypothesis.
- **challenge**: Stress-test a VALIDATED hypothesis (must be validated=true). Try to \
break it — find conditions where it fails.

Make sharp predictions. Narrow correct predictions are rewarded more than wide vague ones.
"""


PROPOSAL_PROMPT = """## Task Description
{task_description}

## Your Mental Model
{thought_md}

## Experiment Tree
{tree_view}

Given this state, what experiment would you propose next? Just describe the experiment briefly (1-2 sentences)."""


def format_scientist_prompt(
    task_description: str,
    task_details: str,
    metric_name: str,
    direction: str,
    baseline_score: str,
    thought_md: str,
    tree_view: str,
    budget_left: int,
    total_budget: int,
    validate_budget: int = 3,
) -> str:
    return SCIENTIST_PROMPT.format(
        task_description=task_description,
        task_details=task_details,
        metric_name=metric_name,
        direction=direction,
        baseline_score=baseline_score,
        thought_md=thought_md,
        tree_view=tree_view,
        budget_left=budget_left,
        total_budget=total_budget,
        validate_budget=validate_budget,
    )


def format_proposal_prompt(
    task_description: str,
    thought_md: str,
    tree_view: str,
) -> str:
    """Shorter prompt for VoI sampling (K=32 proposals)."""
    return PROPOSAL_PROMPT.format(
        task_description=task_description,
        thought_md=thought_md,
        tree_view=tree_view,
    )


import re

def parse_scientist_output(text: str) -> dict:
    """Parse structured scientist output.

    Returns dict with keys: type, hypothesis, experiment, prediction, thought_update
    """
    result = {
        "type": "explore",
        "hypothesis": "",
        "experiment": "",
        "prediction": "",
        "thought_update": "",
        "raw": text,
    }

    for key in ["type", "hypothesis", "experiment", "prediction", "thought_update"]:
        m = re.search(rf'^{key}:\s*(.+?)(?=\n\w+:|$)', text, re.MULTILINE | re.DOTALL)
        if m:
            result[key] = m.group(1).strip()

    # Normalize type
    result["type"] = result["type"].lower().strip()
    if result["type"] not in ("explore", "validate", "challenge"):
        result["type"] = "explore"  # default

    return result


def validate_node_type(
    node_type: str,
    hypothesis_id: str,
    thought_md_obj,  # ThoughtDoc
) -> tuple[bool, str]:
    """Check if the proposed node type is valid given thought.md state.

    Returns (is_valid, error_message).
    """
    if node_type == "explore":
        return True, ""

    if node_type == "validate":
        unvalidated = thought_md_obj.get_unvalidated()
        valid_ids = {h.id for h in unvalidated}
        if hypothesis_id not in valid_ids:
            return False, f"Cannot validate {hypothesis_id}: not an unvalidated hypothesis. Valid: {valid_ids}"
        return True, ""

    if node_type == "challenge":
        validated = thought_md_obj.get_validated()
        valid_ids = {h.id for h in validated}
        if hypothesis_id not in valid_ids:
            return False, f"Cannot challenge {hypothesis_id}: not a validated hypothesis. Valid: {valid_ids}"
        return True, ""

    return False, f"Unknown node type: {node_type}"
