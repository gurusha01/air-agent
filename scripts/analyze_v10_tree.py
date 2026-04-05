"""Analyze v10 tree-structured training run.

Reads the tree rollout log (per-rollout entries with parent_id + depth)
and builds a per-training-step summary with:
  - per-episode tree depth, breadth, and max score
  - reward distribution
  - parent selection patterns (root vs deepen)
  - strategy distribution per depth level
  - overall learning curve (max/mean best across episodes)

Also renders one or more example episodes as ASCII trees.

Handles partial data gracefully (v10 still running → show progress so far).
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASELINE = 0.8478
LOG_DIR = Path("/scratch/jarnav/rollout_logs")
OUT_DIR = Path("/home/jarnav/MLScientist/air-agent/outputs")

# v10 uses batch_size=4, rollouts_per_example=4 → 16 rollouts/step
# Each episode has node_budget=5 rollouts
NODE_BUDGET = 5
EPISODES_PER_STEP = 16


def load_tree_rollouts(scheme="v6_binary"):
    f = LOG_DIR / f"fashionMnist_{scheme}_tree_rollouts.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f)]


def group_into_episodes(rollouts):
    """Given a flat list of tree_rollout entries, group them into episodes.

    Each episode has `node_budget` rollouts. The log entries are written in
    sequence within an episode (node_counter 1..K) and episodes are also
    sequential. So we walk the list looking for node_counter=1 as an episode
    start signal.
    """
    episodes = []
    current = []
    for r in rollouts:
        nc = r.get("node_counter", 0)
        if nc == 1 and current:
            episodes.append(current)
            current = []
        current.append(r)
    if current:
        episodes.append(current)
    return episodes


def summarize_episode(ep):
    """Compute per-episode stats: max score, depths, branch structure."""
    valid_scores = [r["score"] for r in ep if r.get("score") is not None]
    max_score = max(valid_scores) if valid_scores else None
    best = max(max_score, BASELINE) if max_score is not None else BASELINE
    depths = [r.get("new_node_depth", 1) for r in ep]
    max_depth = max(depths) if depths else 1
    parents = [r.get("parent_id", "root") for r in ep]
    n_from_root = sum(1 for p in parents if p == "root")
    n_deepen = len(parents) - n_from_root
    n_fault = sum(1 for r in ep if r.get("executor_fault"))
    return {
        "n_rollouts": len(ep),
        "max_score": max_score,
        "best": best,
        "max_depth": max_depth,
        "n_from_root": n_from_root,
        "n_deepen": n_deepen,
        "n_fault": n_fault,
        "scores": valid_scores,
    }


def render_episode_tree(ep):
    """ASCII render of one episode's tree using parent_id links."""
    # Build adjacency: all episode rollouts + implicit root
    node_by_id = {"root": {"score": BASELINE, "strategy": "baseline"}}
    children = defaultdict(list)
    for r in ep:
        nid = f"node_{r['node_counter']}"
        node_by_id[nid] = {
            "score": r.get("score"),
            "strategy": r.get("direction", "")[:80],
            "parent": r.get("parent_id", "root"),
            "fault": r.get("executor_fault", False),
        }
        children[r.get("parent_id", "root")].append(nid)

    lines = []

    def _render(nid, prefix, is_last, is_root):
        node = node_by_id[nid]
        sc = node.get("score")
        sc_str = "FAIL" if sc is None else f"{sc:.4f}"
        strat = node.get("strategy", "") or ""
        strat = strat.replace("\n", " ")[:60]
        if is_root:
            lines.append(f"{nid}: baseline={sc_str}")
            new_prefix = ""
        else:
            branch = "└─ " if is_last else "├─ "
            flag = " [FAULT]" if node.get("fault") else ""
            lines.append(f"{prefix}{branch}{nid} [{sc_str}]{flag} {strat}")
            new_prefix = prefix + ("   " if is_last else "│  ")
        kids = children.get(nid, [])
        for i, cid in enumerate(kids):
            _render(cid, new_prefix, i == len(kids) - 1, is_root=False)

    _render("root", "", True, is_root=True)
    return "\n".join(lines)


def main():
    rollouts = load_tree_rollouts()
    if not rollouts:
        print("No v10 tree rollouts yet — check /scratch/jarnav/rollout_logs/")
        return
    # Filter to latest run only (by timestamp gap)
    # For simplicity, use ALL entries for now (v10 is a single run)
    print(f"Loaded {len(rollouts)} tree rollouts")

    # Episode grouping
    episodes = group_into_episodes(rollouts)
    n_episodes = len(episodes)
    n_steps = n_episodes // EPISODES_PER_STEP
    print(f"Grouped into {n_episodes} episodes ≈ {n_steps} training steps")

    # Per-episode stats
    ep_stats = [summarize_episode(ep) for ep in episodes]

    # Aggregate per-step
    step_stats = []
    for s in range(n_steps):
        ep_slice = ep_stats[s * EPISODES_PER_STEP:(s + 1) * EPISODES_PER_STEP]
        bests = [e["best"] for e in ep_slice]
        max_depths = [e["max_depth"] for e in ep_slice]
        n_deepen = sum(e["n_deepen"] for e in ep_slice)
        n_root = sum(e["n_from_root"] for e in ep_slice)
        n_fault_total = sum(e["n_fault"] for e in ep_slice)
        step_stats.append({
            "step": s,
            "mean_best": float(np.mean(bests)),
            "max_best": max(bests),
            "mean_depth": float(np.mean(max_depths)),
            "max_depth": max(max_depths),
            "n_deepen_total": n_deepen,
            "n_root_total": n_root,
            "n_fault_total": n_fault_total,
            "n_success": sum(1 for b in bests if b > BASELINE),
        })

    # Print summary
    print(f"\n{'step':>4} {'meanBest':>9} {'maxBest':>9} {'meanDepth':>10} {'maxDepth':>9} "
          f"{'deepen':>7} {'root':>5} {'flt':>4} {'succ':>6}")
    for s in step_stats:
        print(f"{s['step']:>4} {s['mean_best']:>9.4f} {s['max_best']:>9.4f} "
              f"{s['mean_depth']:>10.2f} {s['max_depth']:>9} "
              f"{s['n_deepen_total']:>7} {s['n_root_total']:>5} {s['n_fault_total']:>4} "
              f"{s['n_success']:>3}/{EPISODES_PER_STEP}")

    # Overall aggregates
    all_scores = [s for e in ep_stats for s in e["scores"]]
    print(f"\nOverall: {len(rollouts)} rollouts, {n_episodes} episodes")
    if all_scores:
        print(f"  score mean={np.mean(all_scores):.4f} max={max(all_scores):.4f}")
    n_above_baseline = sum(1 for s in all_scores if s > BASELINE)
    n_faults_total = sum(1 for r in rollouts if r.get("executor_fault"))
    print(f"  above baseline: {n_above_baseline}/{len(all_scores)} = {100*n_above_baseline/max(1,len(all_scores)):.1f}%")
    print(f"  fault rate: {n_faults_total}/{len(rollouts)} = {100*n_faults_total/len(rollouts):.1f}%")
    parent_counter = Counter(r.get("parent_id", "root") for r in rollouts)
    root_parents = parent_counter.pop("root", 0)
    non_root = sum(parent_counter.values())
    print(f"  parent selection: root={root_parents} ({100*root_parents/len(rollouts):.1f}%), "
          f"deepen={non_root} ({100*non_root/len(rollouts):.1f}%)")
    depth_counter = Counter(r.get("new_node_depth", 1) for r in rollouts)
    print(f"  depth distribution of new nodes: {dict(sorted(depth_counter.items()))}")

    # Show 3 example episodes: best, median, worst by max_score
    if n_episodes >= 3 and ep_stats:
        indexed = sorted(enumerate(ep_stats),
                         key=lambda x: x[1]["best"] or 0, reverse=True)
        print("\n=== BEST episode ===")
        print(render_episode_tree(episodes[indexed[0][0]]))
        mid = indexed[len(indexed) // 2][0]
        print("\n=== MEDIAN episode ===")
        print(render_episode_tree(episodes[mid]))
        print("\n=== WORST episode ===")
        print(render_episode_tree(episodes[indexed[-1][0]]))

    # Plot learning curve + depth
    if n_steps >= 1:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        xs = [s["step"] for s in step_stats]
        axes[0, 0].plot(xs, [s["max_best"] for s in step_stats], "b-", label="max best")
        axes[0, 0].plot(xs, [s["mean_best"] for s in step_stats], "b--", alpha=0.6, label="mean best")
        axes[0, 0].axhline(BASELINE, color="gray", linestyle=":", label=f"baseline {BASELINE}")
        axes[0, 0].set_title("v10 tree: best accuracy per step")
        axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("accuracy")
        axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(xs, [s["mean_depth"] for s in step_stats], "g-", label="mean max depth")
        axes[0, 1].plot(xs, [s["max_depth"] for s in step_stats], "g--", alpha=0.6, label="max max depth")
        axes[0, 1].set_title("tree depth per episode (per step)")
        axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("depth")
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        total_per_step = EPISODES_PER_STEP * NODE_BUDGET
        axes[1, 0].plot(xs, [s["n_deepen_total"] / total_per_step for s in step_stats],
                        "r-", label="deepen fraction")
        axes[1, 0].plot(xs, [s["n_root_total"] / total_per_step for s in step_stats],
                        "m-", label="explore-from-root fraction")
        axes[1, 0].set_title("scientist choice: deepen vs explore")
        axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("fraction")
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(xs, [s["n_fault_total"] / total_per_step for s in step_stats],
                        "k-", label="fault rate")
        axes[1, 1].plot(xs, [s["n_success"] / EPISODES_PER_STEP for s in step_stats],
                        "c-", label="success rate (any child > baseline)")
        axes[1, 1].set_title("fault and success rates")
        axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("fraction")
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

        plt.suptitle("FashionMNIST v10 tree-structured RL (v6 reward)", fontsize=13)
        plt.tight_layout()
        out = OUT_DIR / "fmnist_v10_tree_analysis.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
