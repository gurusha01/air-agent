"""Unified analysis for v6/v7/v8/v9 FashionMNIST training runs.

Handles the fact that rollouts and rewards are written by different code
paths (env_response vs reward_fn) and can reach the JSONL files out of order
when 8 rollouts run in parallel. Alignment strategy:

1. Filter both files by each scheme's time window (from SLURM sacct).
2. Group into training steps (batch_size=8 consecutive entries).
3. For strategy labelling, match each reward to a rollout in the same step
   by identical score. If multiple candidates, use earliest-timestamp.

Summary statistics (mean/max best, reward, success rate) aggregate over
the step and don't care about intra-step order, so those are unaffected.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASELINE = 0.8478
BATCH_SIZE = 8
LOG_DIR = Path("/scratch/jarnav/rollout_logs")
OUT_DIR = Path("/home/jarnav/MLScientist/air-agent/outputs")


# (scheme_id, pretty_name, rollouts_file, rewards_file, start_ts, end_ts, scheme_tag)
#   scheme_tag: expected value of `scheme` field in rewards; if None, accept
#   rewards that have NO scheme field (used for v6 pre-refactor entries).
SCHEMES = [
    ("v6", "v6 binary",
     LOG_DIR / "fashionMnist_rollouts.jsonl",
     LOG_DIR / "rewards.jsonl",
     "2026-04-04 01:19:00", "2026-04-04 09:19:59",
     None),  # original v6 run (job 4611953), pre-scheme-field
    ("v7", "v7 fixed tier (τ=0.88)",
     LOG_DIR / "fashionMnist_v7_fixed_tier_rollouts.jsonl",
     LOG_DIR / "rewards_v7_fixed_tier.jsonl",
     "2026-04-04 11:59:22", "2026-04-04 19:59:30",
     "v7_fixed_tier"),  # job 4613886
    ("v8", "v8 global best_ever",
     LOG_DIR / "fashionMnist_v8_global_best_rollouts.jsonl",
     LOG_DIR / "rewards_v8_global_best.jsonl",
     "2026-04-05 10:42:00", "2026-04-05 23:59:59",
     "v8_global_best"),  # retry job 4616793
    ("v9", "v9 percentile (N=64,q=70)",
     LOG_DIR / "fashionMnist_v9_percentile_rollouts.jsonl",
     LOG_DIR / "rewards_v9_percentile.jsonl",
     "2026-04-05 10:42:00", "2026-04-05 23:59:59",
     "v9_percentile"),  # retry job 4616794
]


def load_window(path, start, end, scheme_tag=None):
    """Load entries from a JSONL file filtered by time window and scheme tag.

    scheme_tag:
      - None  -> match entries that LACK a `scheme` field (v6 pre-refactor)
      - str   -> match entries whose scheme field == scheme_tag
      - "any" -> no scheme filter (for rollout files)
    """
    if not path.exists():
        return []
    out = []
    for l in open(path):
        try:
            d = json.loads(l)
        except Exception:
            continue
        ts = d.get("timestamp", "")
        if not (start <= ts <= end):
            continue
        if scheme_tag == "any":
            pass
        elif scheme_tag is None:
            if "scheme" in d:
                continue
        else:
            if d.get("scheme") != scheme_tag:
                continue
        out.append(d)
    return out


def categorize(direction: str) -> str:
    d = (direction or "").lower()
    if "cnn" in d or "conv" in d: return "CNN"
    if "mlp" in d or "hidden layer" in d or "dense" in d or "multilayer" in d: return "MLP"
    if "random forest" in d: return "RF"
    if "xgboost" in d: return "XGB"
    if "lightgbm" in d or "lgbm" in d: return "LGBM"
    if "logistic" in d: return "LogReg"
    if "mobilenet" in d or "resnet" in d or "efficientnet" in d: return "PretrCNN"
    if "transformer" in d or "vit " in d: return "ViT"
    if "knn" in d: return "KNN"
    if "svm" in d: return "SVM"
    if "feature engineer" in d or "pca" in d or "augment" in d: return "FE"
    return "Other"


def match_pairs(rollouts, rewards):
    """For each reward, find the rollout in the same step window whose score
    matches. Steps are determined by batch_size=8 consecutive rewards.
    Within a step we have AT MOST 8 rollouts to pair against.
    We iterate step by step and greedily match by score (falling back to any
    unused rollout in the same step if score matching fails).
    Returns list of (reward, rollout_or_None) pairs.
    """
    n = len(rewards)
    n_steps = n // BATCH_SIZE
    pairs = []
    for s in range(n_steps):
        rw_step = rewards[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
        # Rollouts in the same index window. rollouts might be slightly shifted;
        # use [s*8, (s+1)*8 + tolerance) then intersect.
        rl_window = rollouts[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
        used = [False] * len(rl_window)

        def score_of(rw):
            t = rw.get("tree", [])
            return t[1][1] if len(t) > 1 else None

        for rw in rw_step:
            target = score_of(rw)
            match = None
            # Exact score match first
            if target is not None:
                for i, rl in enumerate(rl_window):
                    if used[i]:
                        continue
                    if rl.get("score") == target:
                        match = rl
                        used[i] = True
                        break
            # Fallback: first unused
            if match is None:
                for i, rl in enumerate(rl_window):
                    if not used[i]:
                        match = rl
                        used[i] = True
                        break
            pairs.append((rw, match))
    return pairs, n_steps


def summarize(scheme_id, pretty, pairs, n_steps):
    print(f"\n=== {pretty} ===")
    print(f"  steps={n_steps}  pairs={len(pairs)}")
    steps = []
    for s in range(n_steps):
        step_pairs = pairs[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
        bests = []
        for rw, _ in step_pairs:
            best = BASELINE
            for _, sc in rw.get("tree", []):
                if sc is not None and sc > best:
                    best = sc
            bests.append(best)
        rewards_list = [rw["reward"] for rw, _ in step_pairs]
        n_fault = sum(1 for rw, rl in step_pairs if rl and rl.get("executor_fault"))
        n_success = sum(1 for b in bests if b > BASELINE)
        steps.append({
            "step": s,
            "mean_best": float(np.mean(bests)),
            "max_best": max(bests),
            "mean_reward": float(np.mean(rewards_list)),
            "n_success": n_success,
            "n_fault": n_fault,
        })
    return steps


def print_table(pretty, steps):
    print(f"\n--- {pretty} per-step (first 5 + last 5) ---")
    print(f"{'step':>4} {'meanBest':>9} {'maxBest':>9} {'meanRew':>9} {'succ':>6} {'flt':>4}")
    show_idx = list(range(min(5, len(steps)))) + ([None] if len(steps) > 10 else []) + \
               list(range(max(5, len(steps) - 5), len(steps)))
    for i in show_idx:
        if i is None:
            print("  ...")
            continue
        s = steps[i]
        print(f"{s['step']:>4} {s['mean_best']:>9.4f} {s['max_best']:>9.4f} "
              f"{s['mean_reward']:>+9.3f} {s['n_success']:>3}/{BATCH_SIZE} {s['n_fault']:>4}")


def strategy_counts(pairs):
    cnts = Counter()
    total = 0
    for _, rl in pairs:
        if rl is None:
            continue
        cnts[categorize(rl.get("direction", ""))] += 1
        total += 1
    return cnts, total


def plot_overlay(results):
    if not any(r[2] for r in results):
        print("No steps to plot")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"v6": "#1f77b4", "v7": "#ff7f0e", "v8": "#2ca02c", "v9": "#d62728"}
    for scheme_id, pretty, steps, _ in results:
        if not steps:
            continue
        xs = [s["step"] for s in steps]
        c = colors.get(scheme_id, "gray")
        axes[0, 0].plot(xs, [s["max_best"] for s in steps], label=pretty, color=c, linewidth=1.6)
        axes[0, 1].plot(xs, [s["mean_best"] for s in steps], label=pretty, color=c, linewidth=1.6)
        axes[1, 0].plot(xs, [s["mean_reward"] for s in steps], label=pretty, color=c, linewidth=1.6)
        axes[1, 1].plot(xs, [s["n_success"] / BATCH_SIZE for s in steps], label=pretty, color=c, linewidth=1.6)
    axes[0, 0].axhline(BASELINE, color="gray", linestyle="--", alpha=0.5, label=f"baseline {BASELINE}")
    axes[0, 0].axhline(0.88, color="orange", linestyle=":", alpha=0.4, label="v7 τ=0.88")
    axes[0, 1].axhline(BASELINE, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.3)
    axes[0, 0].set_title("Max best-score per step")
    axes[0, 1].set_title("Mean best-score per step")
    axes[1, 0].set_title("Mean reward per step")
    axes[1, 1].set_title("Success rate (fraction > baseline)")
    for ax in axes.flat:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel("accuracy"); axes[0, 1].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("reward"); axes[1, 1].set_ylabel("fraction")
    plt.suptitle("FashionMNIST reward scheme comparison (v6 vs v7 vs v8 vs v9)", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "fmnist_schemes_overlay.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved overlay plot: {out}")


def main():
    results = []  # list of (scheme_id, pretty, steps, pairs)
    for scheme_id, pretty, r_path, rw_path, start, end, scheme_tag in SCHEMES:
        # Rollouts don't carry a scheme field pre-refactor, and post-refactor they do;
        # either way, time window is enough for v6 (separate file) and per-scheme files
        # for v7+. So always use "any" for rollouts.
        rollouts = load_window(r_path, start, end, scheme_tag="any")
        rewards = load_window(rw_path, start, end, scheme_tag=scheme_tag)
        if not rollouts and not rewards:
            print(f"[skip] {pretty}: no data in window")
            continue
        print(f"\n{pretty}: in-window rollouts={len(rollouts)} rewards={len(rewards)}")
        if not rewards:
            print("  (no rewards in window → skip)")
            continue
        pairs, n_steps = match_pairs(rollouts, rewards)
        steps = summarize(scheme_id, pretty, pairs, n_steps)
        print_table(pretty, steps)
        cnts, total = strategy_counts(pairs)
        print(f"  strategies ({total} paired rollouts):")
        for cat, n in cnts.most_common():
            print(f"    {cat:10s}: {n:4d}  ({100*n/total:5.1f}%)")
        results.append((scheme_id, pretty, steps, pairs))
    plot_overlay(results)
    return results


if __name__ == "__main__":
    main()
