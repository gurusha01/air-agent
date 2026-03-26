"""
Proper GRPO training using TRL's GRPOTrainer.

Uses tiered reward: 0 (no gain), 0.2 (small gain), 1.0 (big jump).
Score simulation via keyword matching to historical experiment trees.
"""

import json
import os
import random
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Score simulation (from historical trees)
# ---------------------------------------------------------------------------

def load_historical_trees(historical_dir: str) -> list[dict]:
    """Load historical experiment trees for score simulation."""
    trees = []
    hist_path = Path(historical_dir)
    for result_file in hist_path.rglob("result.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
            # Also load node data if available
            nodes_dir = result_file.parent / "nodes"
            nodes = {}
            if nodes_dir.exists():
                for nf in nodes_dir.glob("*.json"):
                    with open(nf) as f:
                        node = json.load(f)
                    nodes[nf.stem] = node
            data["nodes"] = nodes
            trees.append(data)
        except Exception:
            continue
    return trees


def simulate_score(experiment_text: str, historical_trees: list[dict], task_name: str) -> float | None:
    """Simulate experiment score from historical data via keyword matching."""
    matching = [t for t in historical_trees if task_name.lower() in str(t).lower()[:200]]
    if not matching:
        matching = historical_trees

    best_match_score = None
    best_overlap = 0
    exp_words = set(experiment_text.lower().split())

    for tree in matching:
        nodes = tree.get("nodes", {})
        for nid, info in nodes.items():
            if nid == "root":
                continue
            score = info.get("score")
            if score is None:
                continue
            strategy = info.get("strategy", "")
            strat_words = set(strategy.lower().split())
            overlap = len(exp_words & strat_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_score = score

    if best_match_score is not None:
        noise = np.random.normal(0, abs(best_match_score) * 0.05)
        return best_match_score + noise
    return None


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "titanic": {
        "task_name": "Titanic Survival Prediction",
        "baseline_score": 0.7655,
        "higher_is_better": True,
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve a Titanic survival "
            "prediction model (binary classification). Current baseline accuracy: 0.7655. "
            "Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. "
            "Challenges: missing Age values, high-cardinality Cabin, interaction effects. "
            "Output format:\n"
            "```\n"
            "type: explore\n"
            "hypothesis: <your hypothesis>\n"
            "experiment: <what to try>\n"
            "prediction: <expected score range, e.g. 0.80-0.85>\n"
            "```"
        ),
    },
    "battleOfSexes": {
        "task_name": "Battle of Sexes",
        "baseline_score": 1.0227,
        "higher_is_better": True,
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve a Battle of Sexes "
            "game agent. Current baseline payoff: 1.0227. "
            "Game: Two players, strategies 0 and 1. Row player prefers (0,0) payoff 2. "
            "Column player prefers (1,1) payoff 2. Miscoordination payoff 0. "
            "Column player copies with ~80% probability. 10 rounds, observable history. "
            "Output format:\n"
            "```\n"
            "type: explore\n"
            "hypothesis: <your hypothesis>\n"
            "experiment: <what to try>\n"
            "prediction: <expected payoff range, e.g. 1.3-1.5>\n"
            "```"
        ),
    },
    "regression": {
        "task_name": "House Price Prediction",
        "baseline_score": 0.88,
        "higher_is_better": True,
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve a house price "
            "prediction model (regression). Current baseline R²: 0.88. "
            "Key features: OverallQual, GrLivArea, GarageCars, TotalBsmtSF. "
            "Challenges: skewed numerical features, mixed categorical/numerical, missing values. "
            "Output format:\n"
            "```\n"
            "type: explore\n"
            "hypothesis: <your hypothesis>\n"
            "experiment: <what to try>\n"
            "prediction: <expected R² range, e.g. 0.89-0.91>\n"
            "```"
        ),
    },
    "mountaincar": {
        "task_name": "Mountain Car Continuous",
        "baseline_score": 33.79,
        "higher_is_better": True,
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve an RL agent for "
            "Mountain Car Continuous (Gymnax). Current baseline reward: 33.79. "
            "Environment: car at bottom of sinusoidal valley, continuous action [-1,1], "
            "goal is to reach position >= 0.45. Reward: -0.1*action² per step, +100 at goal. "
            "Baseline uses PPO with Actor-Critic. You can try different RL algorithms "
            "(SAC, TD3, MBPO), hyperparameter tuning, reward shaping, network architectures. "
            "Must train 5 checkpoints with different seeds. JAX/Gymnax environment. "
            "Output format:\n"
            "```\n"
            "type: explore\n"
            "hypothesis: <your hypothesis>\n"
            "experiment: <what to try>\n"
            "prediction: <expected reward range, e.g. 50-80>\n"
            "```"
        ),
    },
}


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_tiered_reward_fn(task_cfg: dict, historical_trees: list[dict]):
    """Create a tiered reward function for GRPOTrainer.

    Uses keyword-matching to historical trees for score simulation.
    Tiered: 0 (no gain), 0.2 (small gain), 1.0 (big jump over baseline).
    """
    baseline = task_cfg["baseline_score"]
    higher_is_better = task_cfg["higher_is_better"]
    task_name = task_cfg["task_name"]
    threshold = abs(baseline) * 0.05  # 5% = "big jump"

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            # Extract text from completion (may be list of dicts for chat format)
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
            else:
                text = completion

            sim_score = simulate_score(text, historical_trees, task_name)

            if sim_score is not None:
                delta = (sim_score - baseline) if higher_is_better else (baseline - sim_score)
                if delta > threshold:
                    rewards.append(1.0)
                elif delta > 0:
                    rewards.append(0.2)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def make_best_so_far_reward_fn(task_cfg: dict, historical_trees: list[dict]):
    """Create a reward function that tracks the best score seen so far.

    Instead of comparing to a fixed baseline, compares to the running best.
    The sim_score is the best historical score the model has ever "discovered"
    (i.e., the best score from any historical node matched during training).

    Reward:
      - 1.0 if sim_score sets a new best (discovery!)
      - 0.2 if sim_score is within 95% of current best (competitive)
      - 0.0 otherwise
    """
    higher_is_better = task_cfg["higher_is_better"]
    task_name = task_cfg["task_name"]
    initial_baseline = task_cfg["baseline_score"]

    # Mutable state: best score seen so far
    state = {"best": initial_baseline, "n_discoveries": 0}

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        batch_scores = []

        for completion in completions:
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
            else:
                text = completion

            sim_score = simulate_score(text, historical_trees, task_name)
            batch_scores.append(sim_score)

            if sim_score is not None:
                if higher_is_better:
                    is_new_best = sim_score > state["best"]
                    is_competitive = sim_score >= state["best"] * 0.95
                else:
                    is_new_best = sim_score < state["best"]
                    is_competitive = sim_score <= state["best"] * 1.05

                if is_new_best:
                    rewards.append(1.0)
                elif is_competitive:
                    rewards.append(0.2)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)

        # Update running best from this batch
        valid_scores = [s for s in batch_scores if s is not None]
        if valid_scores:
            if higher_is_better:
                batch_best = max(valid_scores)
                if batch_best > state["best"]:
                    state["n_discoveries"] += 1
                    print(f"  [best-so-far] New best: {state['best']:.4f} → {batch_best:.4f} (discovery #{state['n_discoveries']})")
                    state["best"] = batch_best
            else:
                batch_best = min(valid_scores)
                if batch_best < state["best"]:
                    state["n_discoveries"] += 1
                    print(f"  [best-so-far] New best: {state['best']:.4f} → {batch_best:.4f} (discovery #{state['n_discoveries']})")
                    state["best"] = batch_best

        return rewards

    return reward_fn


def make_dense_reward_fn(task_cfg: dict, historical_trees: list[dict]):
    """Dense reward: raw sim_score normalized by baseline.

    reward = (sim_score - baseline) / |baseline|, clipped to [0, 2].
    Every completion gets a different continuous value → max gradient signal.
    """
    baseline = task_cfg["baseline_score"]
    higher_is_better = task_cfg["higher_is_better"]
    task_name = task_cfg["task_name"]

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
            else:
                text = completion

            sim_score = simulate_score(text, historical_trees, task_name)

            if sim_score is not None:
                if higher_is_better:
                    r = (sim_score - baseline) / max(abs(baseline), 1e-6)
                else:
                    r = (baseline - sim_score) / max(abs(baseline), 1e-6)
                rewards.append(max(min(r, 2.0), 0.0))  # clip to [0, 2]
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_prompt_dataset(task_cfg: dict, num_samples: int = 500) -> Dataset:
    """Create a dataset of repeated prompts for GRPO training."""
    # GRPO needs a dataset of prompts to sample from
    # We repeat the same prompt with slight variations
    prompts = []
    base_prompt = task_cfg["prompt"]
    for i in range(num_samples):
        prompts.append({"prompt": base_prompt})
    return Dataset.from_list(prompts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training with TRL")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--historical-dir", default="/home/jarnav/scratch/air-agent/outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="grpo-scientist-v2")
    parser.add_argument("--reward-type", default="tiered", choices=["tiered", "best-so-far", "dense"],
                        help="'tiered': 0/0.2/1.0 vs baseline. 'best-so-far': adaptive threshold. 'dense': continuous normalized score.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_cfg = TASK_CONFIGS[args.task]
    print(f"[GRPO] Task: {task_cfg['task_name']}")
    print(f"[GRPO] Model: {args.model_path}")
    print(f"[GRPO] Beta (KL): {args.beta}, Epsilon (clip): {args.epsilon}")
    print(f"[GRPO] Num generations: {args.num_generations}")
    print(f"[GRPO] Batch size: {args.batch_size}")

    # Load historical trees
    print(f"[GRPO] Loading historical trees from {args.historical_dir}...")
    historical_trees = load_historical_trees(args.historical_dir)
    print(f"[GRPO] Loaded {len(historical_trees)} trees")

    # Create reward function
    print(f"[GRPO] Reward type: {args.reward_type}")
    if args.reward_type == "best-so-far":
        reward_fn = make_best_so_far_reward_fn(task_cfg, historical_trees)
    elif args.reward_type == "dense":
        reward_fn = make_dense_reward_fn(task_cfg, historical_trees)
    else:
        reward_fn = make_tiered_reward_fn(task_cfg, historical_trees)

    # Create dataset
    dataset = create_prompt_dataset(task_cfg, num_samples=max(args.num_steps * 2, 500))
    print(f"[GRPO] Dataset: {len(dataset)} samples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO config
    run_name = f"grpo_{args.task}_beta{args.beta}_eps{args.epsilon}_G{args.num_generations}"
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        # Training
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,  # KL penalty
        epsilon=args.epsilon,  # PPO clip
        temperature=0.9,
        scale_rewards="group",  # normalize rewards within each group
        # Logging
        logging_steps=1,
        save_steps=50,
        report_to="wandb",
        # Misc
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Set wandb project
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"[GRPO] Starting training: {args.num_steps} steps")
    print(f"[GRPO] Wandb project: {args.wandb_project}, run: {run_name}")

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print(f"[GRPO] Done! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
