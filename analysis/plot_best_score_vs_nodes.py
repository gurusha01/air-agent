"""Plot best score vs number of nodes for AIRA vs LLM-Guided on a given task."""
import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_curve(result_path: str, higher_is_better: bool) -> list[float]:
    """Return running-best score at each node (in insertion order)."""
    d = json.load(open(result_path))
    running_best = None
    curve = []
    for ndata in d["nodes"].values():
        score = ndata.get("score")
        if score is not None:
            if running_best is None:
                running_best = score
            elif higher_is_better and score > running_best:
                running_best = score
            elif not higher_is_better and score < running_best:
                running_best = score
        curve.append(running_best)
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outbase", default="outputs/8h_runs_20260408_000747")
    parser.add_argument("--task", default="titanic")
    parser.add_argument("--out", default="analysis/best_score_vs_nodes.png")
    args = parser.parse_args()

    aira_path = os.path.join(args.outbase, f"aira_{args.task}", "result.json")
    llmg_path = os.path.join(args.outbase, f"llmg_{args.task}", "result.json")

    meta = json.load(open(aira_path if os.path.exists(aira_path) else llmg_path))
    higher = meta.get("higher_is_better", True)
    baseline = meta.get("baseline_score", 0)
    metric = meta.get("primary_metric", "score")

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"AIRA (MCTS)": "#e05c5c", "LLM-Guided": "#4a90d9"}

    for label, path in [("AIRA (MCTS)", aira_path), ("LLM-Guided", llmg_path)]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        curve = load_curve(path, higher)
        xs = list(range(1, len(curve) + 1))
        # Forward-fill None values for plotting
        filled = []
        last = baseline
        for v in curve:
            if v is not None:
                last = v
            filled.append(last)
        ax.plot(xs, filled, label=label, color=colors[label], linewidth=2)
        ax.scatter([len(curve)], [filled[-1]], color=colors[label], s=60, zorder=5)

    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline:.4f})")

    ax.set_xlabel("Number of nodes explored", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Best score vs nodes — {args.task}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
