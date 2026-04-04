"""Build a per-step strategy tree markdown for FashionMNIST v6 training.

For each training step, shows the 8 parallel rollouts as leaves under the
step root, with (accuracy, reward) annotations and a short strategy label.
"""
import json
import re
from pathlib import Path

BATCH_SIZE = 8
BASELINE = 0.8478

rollouts = [json.loads(l) for l in open("/scratch/jarnav/rollout_logs/fashionMnist_rollouts.jsonl")]
rewards = [json.loads(l) for l in open("/scratch/jarnav/rollout_logs/rewards.jsonl")]

assert len(rollouts) == len(rewards), f"{len(rollouts)} vs {len(rewards)}"
n_steps = len(rewards) // BATCH_SIZE


def short_label(direction: str) -> str:
    """Make a ~80-char single-line human label out of a scientist direction."""
    d = direction.strip().replace("\n", " ").replace("  ", " ")
    # Strip leading filler like "Write a train_and_predict.py script that:"
    d = re.sub(r"^(write|modify|update|create|implement)[^.]*?(that|to)[:]?\s*", "", d, flags=re.I)
    d = re.sub(r"\s+", " ", d)
    return d[:90]


def categorize(direction: str) -> str:
    d = direction.lower()
    if "cnn" in d or "conv" in d: return "CNN"
    if "mlp" in d or "hidden layer" in d or "dense" in d or "multilayer" in d: return "MLP"
    if "random forest" in d: return "RF"
    if "xgboost" in d: return "XGB"
    if "logistic" in d: return "LogReg"
    if "lightgbm" in d or "lgbm" in d: return "LGBM"
    if "mobilenet" in d or "resnet" in d or "efficientnet" in d: return "PretrCNN"
    if "transformer" in d or "vit " in d: return "ViT"
    if "knn" in d: return "KNN"
    if "feature engineer" in d or "pca" in d or "augment" in d: return "FE"
    return "Other"


def fmt_score(s):
    if s is None: return "FAIL"
    return f"{s:.4f}"


out = []
out.append("# FashionMNIST v6 — Per-Step Strategy Tree\n")
out.append(f"- Baseline: **{BASELINE}**")
out.append(f"- Batch size: {BATCH_SIZE} (8 parallel rollouts per step)")
out.append(f"- Reward scheme: +1 new best / +0.2 above baseline / 0 below / -0.5 executor fault")
out.append(f"- Structure note: each episode is a **single proposal** from root (node_budget effectively 1), so every \"tree\" is flat — root → one leaf. The 8 parallel rollouts per step are independent siblings under a synthetic step-root below.\n")
out.append(f"Total steps logged: **{n_steps}**\n")
out.append("---\n")

for step in range(n_steps):
    ep_rollouts = rollouts[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
    ep_rewards = rewards[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]

    # Step-level stats
    scores = [e["best_score"] for e in ep_rewards if e.get("best_score") is not None]
    max_best = max(scores) if scores else 0
    rs = [e["reward"] for e in ep_rewards]
    mean_r = sum(rs) / len(rs)
    n_success = sum(1 for e in ep_rewards if e.get("best_score", 0) > BASELINE)

    out.append(f"## Step {step}  ·  max={max_best:.4f}  ·  mean_reward={mean_r:+.3f}  ·  success={n_success}/{BATCH_SIZE}")
    out.append("```")
    out.append(f"root (baseline={BASELINE})")
    for i, (rl, rw) in enumerate(zip(ep_rollouts, ep_rewards)):
        is_last = (i == len(ep_rollouts) - 1)
        branch = "└──" if is_last else "├──"
        cat = categorize(rl["direction"])
        label = short_label(rl["direction"])
        # Prefer tree score (node_1) over rollout score (can differ if tree has the authoritative one)
        node_score = rl.get("score")
        if node_score is None and len(rw["tree"]) > 1:
            node_score = rw["tree"][1][1]
        acc_s = fmt_score(node_score)
        reward = rw["reward"]
        fault = " [FAULT]" if rl.get("executor_fault") else ""
        out.append(f" {branch} [{cat:8s}] acc={acc_s}  r={reward:+.1f}{fault}  │ {label}")
    out.append("```")
    out.append("")

out_path = Path("/home/jarnav/MLScientist/air-agent/outputs/fmnist_v6_strategy_tree.md")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(out))
print(f"Wrote {out_path} ({len(out)} lines)")
