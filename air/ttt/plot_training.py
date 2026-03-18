"""Plot GRPO training curves from episode logs.

Usage:
    python -m air.ttt.plot_training --log-dir outputs/ttt_grpo/battleOfSexes_v1
"""

from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_episodes(log_dir: str) -> list[dict]:
    """Load all episode JSON logs sorted by episode number."""
    logs = sorted(
        glob.glob(str(Path(log_dir) / "logs" / "episode_*.json")),
        key=lambda p: int(Path(p).stem.split("_")[1]),
    )
    episodes = []
    for path in logs:
        with open(path) as f:
            episodes.append(json.load(f))
    return episodes


def plot_training(log_dir: str, save_path: str | None = None):
    """Generate training curves from episode logs."""
    episodes = load_episodes(log_dir)
    if not episodes:
        print(f"No episode logs found in {log_dir}/logs/")
        return

    n = len(episodes)
    ep_nums = list(range(n))

    # --- Extract per-episode metrics ---
    best_scores = [ep["best_score"] for ep in episodes]
    baselines = [ep["baseline"] for ep in episodes]
    improvements = [ep["improvement"] for ep in episodes]
    elapsed = [ep.get("elapsed_s", 0) for ep in episodes]

    # Per-step reward components (averaged over K and steps)
    mean_r_explore = []
    mean_r_exploit = []
    mean_r_memory = []
    mean_total = []
    mean_loss = []

    for ep in episodes:
        explores, exploits, memories, totals, losses = [], [], [], [], []
        for step in ep.get("steps", []):
            for comp in step.get("components", []):
                explores.append(comp.get("r_explore", 0))
                exploits.append(comp.get("r_exploit", 0))
                memories.append(comp.get("r_memory", 0))
                totals.append(comp.get("total", 0))
            losses.append(step.get("loss", 0))

        mean_r_explore.append(np.mean(explores) if explores else 0)
        mean_r_exploit.append(np.mean(exploits) if exploits else 0)
        mean_r_memory.append(np.mean(memories) if memories else 0)
        mean_total.append(np.mean(totals) if totals else 0)
        mean_loss.append(np.mean(losses) if losses else 0)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GRPO Scientist Training — {Path(log_dir).name}", fontsize=14)

    # 1. Best score per episode
    ax = axes[0, 0]
    ax.plot(ep_nums, best_scores, "b-o", markersize=4, label="Best score")
    ax.axhline(baselines[0], color="r", linestyle="--", alpha=0.7, label="Baseline")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Best Score per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Improvement over baseline
    ax = axes[0, 1]
    ax.plot(ep_nums, improvements, "g-o", markersize=4)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Improvement")
    ax.set_title("Improvement over Baseline")
    ax.grid(True, alpha=0.3)

    # 3. Reward components
    ax = axes[1, 0]
    ax.plot(ep_nums, mean_r_explore, "c-", alpha=0.8, label="R_explore")
    ax.plot(ep_nums, mean_r_exploit, "m-", alpha=0.8, label="R_exploit")
    ax.plot(ep_nums, mean_r_memory, "y-", alpha=0.8, label="R_memory")
    ax.plot(ep_nums, mean_total, "k-", linewidth=2, label="Total")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Components (averaged over K×steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss
    ax = axes[1, 1]
    ax.plot(ep_nums, mean_loss, "r-", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Loss")
    ax.set_title("GRPO Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = str(Path(log_dir) / "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()

    # Also print a text summary
    print(f"\n{'Episode':>8} {'Best':>8} {'Improv':>8} {'R_total':>8} {'Loss':>8}")
    print("-" * 44)
    for i in range(n):
        print(f"{i:>8d} {best_scores[i]:>8.4f} {improvements[i]:>+8.4f} "
              f"{mean_total[i]:>8.4f} {mean_loss[i]:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training curves")
    parser.add_argument("--log-dir", required=True, help="Output dir with logs/ subfolder")
    parser.add_argument("--save-path", default=None, help="Where to save the plot")
    args = parser.parse_args()
    plot_training(args.log_dir, args.save_path)


if __name__ == "__main__":
    main()
