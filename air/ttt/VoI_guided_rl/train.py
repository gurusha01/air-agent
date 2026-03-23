"""
GRPO training with VoI-guided rewards for the scientist model.

Phase 1: SFT on grounded data (already done in v3 experiments)
Phase 2: GRPO with hypothesis-driven tree search and VoI rewards

Training loop:
1. For each training problem, sample K experiment proposals per node
2. Compute VoI for explore proposals (centroid cosine distance)
3. Simulate/run the selected experiment
4. Compute per-node rewards (R1 + R2) and end-of-tree R3
5. GRPO update: reinforce proposals with above-mean reward

Usage:
    python -m air.ttt.VoI_guided_rl.train \
        --model-path /scratch/jarnav/scientist_v3_sft/merged_epoch1 \
        --output-dir /scratch/jarnav/voi_grpo/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .scientist_env import VoIScientistEnv, VoIState
from .rewards import NodeReward


# ---------------------------------------------------------------------------
# Historical data loader (simulate experiments from past tree search results)
# ---------------------------------------------------------------------------

def load_historical_trees(outputs_dir: str) -> list[dict]:
    """Load past tree search results for simulated training.

    Instead of running the executor in a container (expensive), we replay
    historical experiment outcomes. The scientist proposes experiments,
    and we look up what score similar experiments got in the past.
    """
    trees = []
    outputs_path = Path(outputs_dir)

    for result_file in outputs_path.rglob("result.json"):
        try:
            data = json.loads(result_file.read_text())
            if data.get("total_nodes", 0) >= 3:
                trees.append(data)
        except Exception:
            continue

    print(f"[train] Loaded {len(trees)} historical trees for simulation")
    return trees


def simulate_score(experiment_text: str, historical_tree: dict) -> float | None:
    """Simulate an experiment score from historical data.

    Simple heuristic: find the most similar node in the historical tree
    and return its score (with some noise).
    """
    nodes = historical_tree.get("nodes", historical_tree.get("tree_shape", {}))
    baseline = historical_tree.get("baseline_score", 0)

    # Find best matching historical node
    best_match_score = None
    for nid, info in nodes.items():
        if nid == "root":
            continue
        node_score = info.get("score")
        if node_score is None:
            continue
        strategy = info.get("strategy", "")
        # Simple keyword overlap as similarity
        exp_words = set(experiment_text.lower().split())
        strat_words = set(strategy.lower().split())
        overlap = len(exp_words & strat_words)
        if overlap > 2 or best_match_score is None:
            best_match_score = node_score

    if best_match_score is not None:
        # Add noise
        noise = np.random.normal(0, abs(best_match_score - baseline) * 0.1)
        return best_match_score + noise
    return None


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "titanic": {
        "task_config": "tasks/titanic.yaml",
        "task_name": "Titanic Survival Prediction",
        "metric_name": "accuracy",
        "higher_is_better": True,
        "baseline_score": 0.7655,
        "description": "Binary classification: predict passenger survival on the Titanic. "
                       "Features include Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. "
                       "Class imbalance: ~38% survived. Key challenges: missing Age values, "
                       "high-cardinality Cabin feature, interaction effects (Pclass × Sex).",
    },
    "regression": {
        "task_config": "tasks/regressionKaggleHousePrice.yaml",
        "task_name": "House Price Prediction (Kaggle)",
        "metric_name": "r2",
        "higher_is_better": True,
        "baseline_score": 0.88,
        "description": "Regression: predict house sale prices from 79 features. "
                       "Key features: OverallQual, GrLivArea, GarageCars, TotalBsmtSF. "
                       "Challenges: skewed numerical features, mixed categorical/numerical, "
                       "missing values in multiple columns.",
    },
    "battleOfSexes": {
        "task_config": "tasks/battleOfSexes.yaml",
        "task_name": "Battle of Sexes",
        "metric_name": "score",
        "higher_is_better": True,
        "baseline_score": 1.0227,
        "description": "Game theory: Battle of the Sexes. Two strategies 0 and 1. "
                       "Row player prefers (0,0) → payoff 2. Column player prefers (1,1) → payoff 2. "
                       "Miscoordination → payoff 0. Column player copies with ~80% probability. "
                       "10 rounds with observable history.",
    },
}


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def grpo_step(
    model,
    tokenizer,
    ref_model,
    optimizer,
    env: VoIScientistEnv,
    historical_tree: dict,
    K: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    kl_coeff: float = 0.1,
) -> dict:
    """One GRPO training step on a single tree.

    1. For each node position, sample K proposals
    2. Compute rewards for each proposal
    3. Compute advantages (reward - mean_reward)
    4. GRPO policy gradient update
    """
    device = next(model.parameters()).device
    state = env.reset()
    all_losses = []

    for node_idx in range(env.node_budget):
        if state.budget_left <= 0:
            break

        # Sample K proposals
        prompt = state.prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        proposals = []
        log_probs_list = []

        model.eval()
        for k in range(K):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            proposals.append(text)

            # Compute log prob of this generation under current policy
            with torch.no_grad():
                full_ids = outputs.sequences[0].unsqueeze(0)
                model_out = model(full_ids)
                logits = model_out.logits[0, inputs["input_ids"].shape[1]-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                total_log_prob = token_log_probs.sum()
                log_probs_list.append(total_log_prob)

        # Compute rewards for each proposal
        rewards = []
        for k, proposal_text in enumerate(proposals):
            from .prompts import parse_scientist_output
            parsed = parse_scientist_output(proposal_text)

            # Simulate the experiment
            sim_score = simulate_score(parsed.get("experiment", ""), historical_tree)

            # Step the environment (for the first valid proposal only)
            if k == 0:
                result = env.step(
                    proposal_text,
                    actual_score=sim_score,
                    model=model if parsed["type"] == "explore" else None,
                    tokenizer=tokenizer if parsed["type"] == "explore" else None,
                )
                rewards.append(result.reward.r_node if result.valid else 0.0)
            else:
                # For other proposals, compute R2 (VoI) only if explore
                # Skip full VoI computation for speed — use R1 estimate only
                prediction = None
                from .rewards import parse_prediction
                prediction = parse_prediction(parsed.get("prediction", ""))
                if prediction and sim_score is not None:
                    from .rewards import compute_r1
                    r1 = compute_r1(prediction[0], prediction[1], sim_score)
                    rewards.append(env.alpha * r1)
                else:
                    rewards.append(0.0)

        # GRPO: compute advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_reward = rewards_tensor.mean()
        advantages = rewards_tensor - mean_reward

        # Policy gradient loss
        model.train()
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        for k in range(K):
            if abs(advantages[k].item()) < 1e-8:
                continue

            # Recompute log prob with gradients
            generated_ids = tokenizer(proposals[k], return_tensors="pt", truncation=True,
                                      max_length=max_new_tokens).to(device)["input_ids"][0]
            full_ids = torch.cat([inputs["input_ids"][0], generated_ids]).unsqueeze(0)

            model_out = model(full_ids)
            logits = model_out.logits[0, inputs["input_ids"].shape[1]-1:-1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            if generated_ids.shape[0] <= logits.shape[0]:
                token_log_probs = log_probs[:generated_ids.shape[0]].gather(
                    1, generated_ids.unsqueeze(1)
                ).squeeze(1)
                total_log_prob = token_log_probs.sum()

                # KL penalty vs reference model
                with torch.no_grad():
                    ref_out = ref_model(full_ids)
                    ref_logits = ref_out.logits[0, inputs["input_ids"].shape[1]-1:-1, :]
                    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    if generated_ids.shape[0] <= ref_logits.shape[0]:
                        ref_token_log_probs = ref_log_probs[:generated_ids.shape[0]].gather(
                            1, generated_ids.unsqueeze(1)
                        ).squeeze(1)
                        kl = (token_log_probs - ref_token_log_probs).mean()
                    else:
                        kl = torch.tensor(0.0, device=device)

                # GRPO loss: -advantage * log_prob + kl_coeff * kl
                proposal_loss = -advantages[k] * total_log_prob + kl_coeff * kl
                loss = loss + proposal_loss / K

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        all_losses.append(loss.item())

        # Get next state
        state = env._get_state()

    # End of tree: compute R3
    tree_rewards = env.get_tree_rewards()

    return {
        "mean_loss": np.mean(all_losses) if all_losses else 0.0,
        "mean_reward": float(mean_reward.item()) if len(rewards) > 0 else 0.0,
        "best_score": env.best_score,
        "improvement": env.best_score - env.baseline_score,
        "n_hypotheses": len(env.thought.hypotheses),
        "n_validated": len(env.thought.get_validated()),
        "summary": env.get_summary(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    print(f"[VoI-GRPO] Loading model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (with LoRA)
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=args.target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Load historical trees for simulation
    historical = load_historical_trees(args.historical_dir)
    if not historical:
        print("[VoI-GRPO] WARNING: No historical trees found. Training will use random scores.")

    # Task pool
    task_pool = [TASK_CONFIGS[t] for t in args.tasks.split(",") if t in TASK_CONFIGS]
    if not task_pool:
        task_pool = list(TASK_CONFIGS.values())
    print(f"[VoI-GRPO] Training on {len(task_pool)} tasks: {[t['task_name'] for t in task_pool]}")

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "task", "loss", "mean_reward", "best_score",
                         "improvement", "n_hypotheses", "n_validated", "elapsed"])
    log_file.flush()

    print(f"[VoI-GRPO] Starting training: {args.steps} steps")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # Pick a random task
        task_cfg = random.choice(task_pool)

        # Pick a random historical tree for this task (for score simulation)
        task_name = task_cfg["task_name"]
        matching_trees = [t for t in historical if t.get("task", "") == task_name]
        hist_tree = random.choice(matching_trees) if matching_trees else {"nodes": {}}

        # Create environment
        env = VoIScientistEnv(
            task_config=task_cfg["task_config"],
            task_description=task_cfg["description"],
            task_name=task_cfg["task_name"],
            metric_name=task_cfg["metric_name"],
            higher_is_better=task_cfg["higher_is_better"],
            baseline_score=task_cfg["baseline_score"],
            node_budget=args.node_budget,
            voi_K=args.voi_K,
        )

        # GRPO step
        result = grpo_step(
            model=model,
            tokenizer=tokenizer,
            ref_model=ref_model,
            optimizer=optimizer,
            env=env,
            historical_tree=hist_tree,
            K=args.grpo_K,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            kl_coeff=args.kl_coeff,
        )

        elapsed = time.time() - t0
        log_writer.writerow([
            step, task_cfg["task_name"], f"{result['mean_loss']:.6f}",
            f"{result['mean_reward']:.6f}", f"{result['best_score']:.4f}",
            f"{result['improvement']:.4f}",
            result["n_hypotheses"], result["n_validated"],
            f"{elapsed:.0f}",
        ])
        log_file.flush()

        if step % args.log_every == 0:
            print(f"[step {step}/{args.steps}] task={task_cfg['task_name'][:20]} "
                  f"loss={result['mean_loss']:.4f} reward={result['mean_reward']:.4f} "
                  f"best={result['best_score']:.4f} hyp={result['n_hypotheses']} "
                  f"val={result['n_validated']} ({elapsed:.0f}s)")

        if step % args.save_every == 0:
            save_dir = output_dir / f"step_{step}"
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            # Save example tree
            with open(save_dir / "example_tree.json", "w") as f:
                json.dump(result["summary"], f, indent=2)
            print(f"  Saved checkpoint to {save_dir}")

    # Final save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log_file.close()
    print(f"[VoI-GRPO] Training complete. {args.steps} steps in {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="VoI-guided GRPO for scientist")
    parser.add_argument("--model-path", default="/scratch/jarnav/scientist_v3_sft/merged_epoch1")
    parser.add_argument("--output-dir", default="/scratch/jarnav/voi_grpo/checkpoints")
    parser.add_argument("--historical-dir", default="/home/jarnav/scratch/air-agent/outputs")
    parser.add_argument("--tasks", default="titanic,regression,battleOfSexes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--node-budget", type=int, default=8)
    parser.add_argument("--grpo-K", type=int, default=8)
    parser.add_argument("--voi-K", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
