"""Plot strong model baseline comparison (Experiment 4).

Compares 5 method/model combinations across 4 tasks at 2 budgets:
1. AIRA MCTS + Claude Opus executor
2. AIRA MCTS + Qwen 4B executor (existing Feb22 results)
3. LLM-Guided v2: Claude scientist + Claude executor
4. LLM-Guided v2: Claude scientist + Qwen executor
5. LLM-Guided v2: Qwen scientist + Qwen executor
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Base directories
strong_base = Path("outputs/Strong_Baselines")
feb22_base = Path("outputs/Feb22_Baselines")

tasks = ["titanic", "houseprice", "bos", "mountaincar"]
task_labels = ["Titanic", "House Price", "Battle of Sexes", "Mountain Car"]
budgets = [5, 15]

# Method definitions: (key, label, color, base_dir, file_prefix)
methods = [
    ("aira_claude",       "AIRA MCTS (Claude Opus)",             "#E91E63", strong_base / "aira_claude_opus",          "aira_mcts"),
    ("aira_qwen",         "AIRA MCTS (Qwen 4B)",                 "#FF5722", feb22_base,                                "aira_mcts"),
    ("llm_claude_both",   "LLM-Guided (Claude+Claude)",          "#9C27B0", strong_base / "llm_guided_claude_both",    "llm_guided"),
    ("llm_claude_sci",    "LLM-Guided (Claude+Qwen)",            "#2196F3", strong_base / "llm_guided_claude_scientist","llm_guided"),
    ("llm_qwen_both",     "LLM-Guided (Qwen+Qwen)",             "#4CAF50", strong_base / "llm_guided_qwen_both",      "llm_guided"),
]
n_methods = len(methods)

fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
fig.suptitle("Strong Model Baselines: Method x Model Comparison (best score in tree)",
             fontsize=14, fontweight="bold")

for ti, (task, label) in enumerate(zip(tasks, task_labels)):
    ax = axes[ti]
    for mi, (key, mlabel, color, base_dir, prefix) in enumerate(methods):
        means, stds, xs = [], [], []
        for budget in budgets:
            scores = []
            for r in range(1, 6):
                rpath = base_dir / task / f"{prefix}_n{budget}_r{r}" / "result.json"
                if rpath.exists():
                    data = json.loads(rpath.read_text())
                    scores.append(data.get("best_score", 0) or 0)
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
                xs.append(budget)
                # Spread dots so they don't overlap
                offset = (mi - (n_methods - 1) / 2) * 0.4
                for s in scores:
                    ax.scatter(budget + offset, s, color=color, alpha=0.3, s=20, zorder=3)

        if means:
            ax.errorbar(xs, means, yerr=stds, color=color, marker="o", label=mlabel,
                        capsize=4, linewidth=2, markersize=6, zorder=4)

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Node Budget")
    ax.set_xticks(budgets)
    if ti == 0:
        ax.set_ylabel("Best Score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")

plt.tight_layout()
out = strong_base / "strong_model_baselines.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
