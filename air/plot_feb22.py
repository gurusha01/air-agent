import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path("outputs/Feb22_Baselines")
llm_guided_v1 = Path("outputs/LLM_Guided")
llm_guided_v2 = Path("outputs/LLM_Guided_v2")
tasks = ["titanic", "houseprice", "bos", "mountaincar"]
task_labels = ["Titanic", "House Price", "Battle of Sexes", "Mountain Car"]
# methods = ["softmax", "aira_mcts", "oe", "llm_guided", "llm_guided_v2"]
# method_labels = ["Softmax (ours)", "AIRA MCTS", "Open-Ended", "LLM-Guided v1", "LLM-Guided v2"]
# colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
methods = ["softmax", "aira_mcts", "oe", "llm_guided_v2"]
method_labels = ["Softmax (ours)", "AIRA MCTS", "Open-Ended", "LLM-Guided v2"]
colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
budgets = [5, 15]
n_methods = len(methods)

# Map method name to its base directory
method_bases = {
    "softmax": base,
    "aira_mcts": base,
    "oe": base,
    "llm_guided": llm_guided_v1,
    "llm_guided_v2": llm_guided_v2,
}
# llm_guided_v2 uses same filename prefix "llm_guided" on disk
method_file_prefix = {
    "llm_guided_v2": "llm_guided",
}

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Tree Search Methods Comparison (best score in tree)",
             fontsize=14, fontweight="bold")

for ti, (task, label) in enumerate(zip(tasks, task_labels)):
    ax = axes[ti]
    for mi, (method, mlabel, color) in enumerate(zip(methods, method_labels, colors)):
        method_base = method_bases[method]
        file_prefix = method_file_prefix.get(method, method)

        means, stds, xs = [], [], []
        for budget in budgets:
            scores = []
            for r in range(1, 6):
                rpath = method_base / task / f"{file_prefix}_n{budget}_r{r}" / "result.json"
                if rpath.exists():
                    data = json.loads(rpath.read_text())
                    scores.append(data.get("best_score", 0) or 0)
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
                xs.append(budget)
                # Spread dots so they don't overlap
                offset = (mi - (n_methods - 1) / 2) * 0.5
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
    ax.legend(fontsize=7)

plt.tight_layout()
out = base / "feb22_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
