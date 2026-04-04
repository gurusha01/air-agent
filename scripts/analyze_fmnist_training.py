"""Analyze FashionMNIST v6 training run — plot scores, extract strategies per step."""
import json
from pathlib import Path
from collections import defaultdict
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load rollout log (has per-turn details)
rollouts = []
with open("/scratch/jarnav/rollout_logs/fashionMnist_rollouts.jsonl") as f:
    for l in f:
        rollouts.append(json.loads(l))

# Load rewards log (one per episode)
rewards = []
with open("/scratch/jarnav/rollout_logs/rewards.jsonl") as f:
    for l in f:
        rewards.append(json.loads(l))

# Group rollouts/rewards into training steps. batch_size=8 → every 8 episodes = 1 step
BATCH_SIZE = 8
n_steps = len(rewards) // BATCH_SIZE
print(f"Loaded {len(rollouts)} rollout entries, {len(rewards)} episodes")
print(f"Estimated {n_steps} training steps (batch_size={BATCH_SIZE})")

# For each step, compute:
# - mean best_score (across all 8 episodes)
# - max best_score
# - mean reward
# - number of successful rollouts (score > baseline)
steps_data = []
baseline = 0.8478
for step in range(n_steps):
    episodes = rewards[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
    best_scores = [e["best_score"] for e in episodes if e["best_score"] and e["best_score"] > 0]
    all_scores = []
    for e in episodes:
        for _, s in e["tree"]:
            if s is not None and s > 0:
                all_scores.append(s)
    rewards_step = [e["reward"] for e in episodes]
    steps_data.append({
        "step": step,
        "mean_best": np.mean(best_scores) if best_scores else 0,
        "max_best": max(best_scores) if best_scores else 0,
        "mean_score": np.mean(all_scores) if all_scores else 0,
        "mean_reward": np.mean(rewards_step),
        "n_success": sum(1 for e in episodes if e["best_score"] and e["best_score"] > baseline),
    })

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

steps = [d["step"] for d in steps_data]

# 1. Best score per step
ax = axes[0, 0]
ax.plot(steps, [d["max_best"] for d in steps_data], "o-", label="Max (best across 8 rollouts)", color="green")
ax.plot(steps, [d["mean_best"] for d in steps_data], "s-", label="Mean best (avg of best per rollout)", color="blue")
ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline})")
ax.set_xlabel("Train step")
ax.set_ylabel("Accuracy")
ax.set_title("Best accuracy per train step")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Mean score across all turns
ax = axes[0, 1]
ax.plot(steps, [d["mean_score"] for d in steps_data], "o-", color="purple")
ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline})")
ax.set_xlabel("Train step")
ax.set_ylabel("Mean accuracy (all turns)")
ax.set_title("Mean accuracy across all turns per step")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Mean reward per step
ax = axes[1, 0]
ax.plot(steps, [d["mean_reward"] for d in steps_data], "o-", color="orange")
ax.set_xlabel("Train step")
ax.set_ylabel("Mean reward")
ax.set_title("Mean reward per step (discrete: -0.5 / 0 / 1)")
ax.grid(True, alpha=0.3)

# 4. Success rate
ax = axes[1, 1]
ax.plot(steps, [d["n_success"] for d in steps_data], "o-", color="teal")
ax.set_xlabel("Train step")
ax.set_ylabel("# rollouts beating baseline (out of 8)")
ax.set_title("Success rate per step")
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3)

plt.suptitle("FashionMNIST v6 RL Training — Score & Reward Evolution", fontsize=14, y=1.02)
plt.tight_layout()
out_path = Path("/home/jarnav/MLScientist/air-agent/outputs/fmnist_v6_analysis.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=100, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")

# Print table
print("\n=== PER-STEP DATA ===")
print(f"{'Step':>4} {'MeanBest':>10} {'MaxBest':>10} {'MeanScore':>10} {'MeanReward':>10} {'Success':>8}")
for d in steps_data:
    print(f"{d['step']:>4} {d['mean_best']:>10.4f} {d['max_best']:>10.4f} {d['mean_score']:>10.4f} {d['mean_reward']:>10.3f} {d['n_success']:>8}/8")

# Extract strategies per step
print("\n=== STRATEGIES EXPLORED PER STEP ===")
# Group rollouts by time (roughly corresponds to step since we write in order)
# Each turn is logged, and node_counter increments within an episode
# An episode has node_budget=2 turns, so 8 episodes = 16 rollout entries per step
TURNS_PER_STEP = BATCH_SIZE * 2  # 8 episodes × 2 turns

# Categorize each direction
def categorize(direction: str) -> str:
    d = direction.lower()
    if "cnn" in d or "conv" in d: return "CNN"
    if "mlp" in d or "multilayer" in d or "hidden layer" in d or "dense" in d: return "MLP/Dense"
    if "random forest" in d or "randomforest" in d: return "RandomForest"
    if "xgboost" in d or "xgb " in d: return "XGBoost"
    if "logistic" in d: return "Logistic"
    if "svm" in d: return "SVM"
    if "lightgbm" in d or "lgbm" in d: return "LightGBM"
    if "mobilenet" in d or "resnet" in d or "efficientnet" in d: return "PretrainedCNN"
    if "transformer" in d or "vit " in d: return "Transformer"
    if "knn" in d or "k-nearest" in d or "nearest neighbor" in d: return "KNN"
    if "gradient boost" in d: return "GradientBoosting"
    if "feature engineer" in d or "pca" in d or "augment" in d: return "FeatureEng"
    return "Other"

for step in range(min(n_steps, 40)):
    step_turns = rollouts[step * TURNS_PER_STEP : (step + 1) * TURNS_PER_STEP]
    categories = [categorize(t["direction"]) for t in step_turns]
    from collections import Counter
    cnt = Counter(categories)
    summary = ", ".join(f"{k}={v}" for k, v in cnt.most_common())
    print(f"Step {step:>3}: {summary}")
