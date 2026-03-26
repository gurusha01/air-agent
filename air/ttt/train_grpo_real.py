"""
GRPO training with REAL MLGym execution rewards.

Each proposal is actually executed by an executor LLM in an MLGym container.
Reward = (actual_score - baseline) / |baseline|.

Requires:
- GPU 0: Policy model (GRPO trainer)
- GPU 1: Executor vLLM server
- CPU: MLGym Apptainer containers (parallelized)
"""

import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add air-agent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from air.tree_search import (
    ContainerManager, LLMClient, TaskProfile, get_task_profile, extract_command,
    MLGYM_PATH,
)


# ---------------------------------------------------------------------------
# Task configs (maps our task names to MLGym task configs)
# ---------------------------------------------------------------------------

TASK_MAP = {
    "titanic": {
        "task_config": "tasks/titanic.yaml",
        "baseline": 0.7655,
        "higher_is_better": True,
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve a Titanic survival "
            "prediction model. Current baseline accuracy: 0.7655. "
            "Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. "
            "Output a specific, actionable experiment direction in 2-3 sentences."
        ),
    },
    "battleOfSexes": {
        "task_config": "tasks/battleOfSexes.yaml",
        "baseline": 1.0227,
        "higher_is_better": True,
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve a Battle of Sexes "
            "game agent. Current baseline payoff: 1.0227. "
            "Column player copies with ~80% probability. 10 rounds. "
            "Output a specific, actionable experiment direction in 2-3 sentences."
        ),
    },
    "regression": {
        "task_config": "tasks/regressionKaggleHousePrice.yaml",
        "baseline": 0.88,
        "higher_is_better": True,
        "container_image": os.environ.get("MLGYM_APPTAINER_IMAGE", "/scratch/jarnav/mlgym_sandbox"),
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve house price prediction. "
            "Current baseline R²: 0.88. Key features: OverallQual, GrLivArea, GarageCars. "
            "Output a specific, actionable experiment direction in 2-3 sentences."
        ),
    },
    "mountaincar": {
        "task_config": "tasks/rlMountainCarContinuous.yaml",
        "baseline": 33.79,
        "higher_is_better": True,
        "container_image": os.environ.get("MLGYM_RL_IMAGE", "/scratch/jarnav/mlgym_rl.sif"),
        "prompt": (
            "You are an ML scientist. Propose ONE experiment to improve an RL agent for "
            "Mountain Car Continuous. Current baseline reward: 33.79. "
            "Baseline uses PPO. Try different algorithms, hyperparameters, or architectures. "
            "Output a specific, actionable experiment direction in 2-3 sentences."
        ),
    },
}


# ---------------------------------------------------------------------------
# Single proposal execution
# ---------------------------------------------------------------------------

def execute_proposal(
    proposal_text: str,
    task_config: str,
    task_profile: TaskProfile,
    executor_url: str,
    executor_model: str,
    container_image: str,
    env_gpu: str = "cpu",
    max_actions: int = 15,
) -> float | None:
    """Execute a single proposal in an MLGym container and return the score.

    1. Create container
    2. Build initial messages with the proposal as the direction
    3. Multi-turn executor loop: generate command → run → check score
    4. Cleanup and return score
    """
    container = None
    try:
        # Create container
        container = ContainerManager(
            task_config=task_config,
            env_gpu=env_gpu,
            image_name=container_image,
            task_profile=task_profile,
        )
        container.create()

        # Create executor client
        executor = LLMClient(
            base_url=executor_url,
            model=executor_model,
            temperature=0.9,
        )

        # Build initial messages
        data_head = ""
        if task_profile.data_head_cmd:
            try:
                data_head = container.env.communicate(task_profile.data_head_cmd, timeout_duration=10)
            except Exception:
                data_head = ""

        task_desc = task_profile.root_task_desc.format(
            baseline_score=container.baseline_score,
            data_head=data_head,
        )

        messages = [
            {"role": "system", "content": task_profile.system_prompt},
            {"role": "user", "content": task_desc},
        ]

        # Add the proposal as a direction
        direction_msg = (
            f"Strategy to try: {proposal_text}\n\n"
            f"{task_profile.branch_write_instruction}"
        )
        messages.append({"role": "user", "content": direction_msg})

        # Multi-turn executor loop
        score = None
        for step in range(max_actions):
            try:
                raw = executor.chat(messages)
            except Exception as e:
                print(f"    Executor error at step {step}: {e}")
                break

            action, _ = extract_command(raw)
            if not action:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "No command detected. Output a valid command."})
                continue

            if action.strip().lower() == "submit":
                action = "validate"

            obs, info = container.step(action)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": obs})

            # Check for score
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(
                        task_profile.primary_metric, list(score_data.values())[0]
                    )
                else:
                    score = score_data
                break
        else:
            # Force validate
            obs, info = container.step("validate")
            if info.get("score"):
                score_data = info["score"][-1]
                if isinstance(score_data, dict):
                    score = score_data.get(
                        task_profile.primary_metric, list(score_data.values())[0]
                    )
                else:
                    score = score_data

        return score

    except Exception as e:
        print(f"    Container execution failed: {e}")
        return None
    finally:
        if container and container.env:
            try:
                container.env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Reward function with real execution
# ---------------------------------------------------------------------------

def make_real_execution_reward_fn(
    task_cfg: dict,
    task_profile: TaskProfile,
    executor_url: str,
    executor_model: str,
    max_parallel: int = 4,
    max_actions: int = 15,
    reward_scale: str = "tanh",
):
    """Create a reward function that executes proposals in real MLGym containers."""
    baseline = task_cfg["baseline"]
    higher_is_better = task_cfg["higher_is_better"]
    task_config = task_cfg["task_config"]
    threshold = abs(baseline) * 0.05  # for tiered
    container_image = task_cfg["container_image"]

    state = {"step": 0, "total_executions": 0, "total_successes": 0, "best_score": baseline}

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        state["step"] += 1
        n = len(completions)
        print(f"\n  [real-exec] Step {state['step']}: executing {n} proposals ({max_parallel}-way parallel)...")

        # Extract text from completions
        texts = []
        for c in completions:
            if isinstance(c, list):
                texts.append(" ".join(m.get("content", "") for m in c if isinstance(m, dict)))
            else:
                texts.append(c)

        # Execute in parallel
        scores = [None] * n

        def run_one(idx):
            return idx, execute_proposal(
                proposal_text=texts[idx],
                task_config=task_config,
                task_profile=task_profile,
                executor_url=executor_url,
                executor_model=executor_model,
                container_image=container_image,
                max_actions=max_actions,
            )

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(run_one, i) for i in range(n)]
            for future in as_completed(futures):
                try:
                    idx, score = future.result()
                    scores[idx] = score
                except Exception as e:
                    print(f"    Execution error: {e}")

        # Compute rewards
        import math
        rewards = []
        for i, score in enumerate(scores):
            state["total_executions"] += 1
            if score is not None:
                state["total_successes"] += 1
                if higher_is_better:
                    delta = score - baseline
                else:
                    delta = baseline - score

                if reward_scale == "tiered":
                    if delta > threshold:
                        r = 1.0
                    elif delta > 0:
                        r = 0.2
                    else:
                        r = 0.0
                else:  # tanh
                    normalized = delta / max(abs(baseline), 1e-6)
                    r = math.tanh(normalized)
                rewards.append(r)

                # Track best
                if (higher_is_better and score > state["best_score"]) or \
                   (not higher_is_better and score < state["best_score"]):
                    print(f"    NEW BEST: {state['best_score']:.4f} → {score:.4f}")
                    state["best_score"] = score
            else:
                rewards.append(0.0)

        success_rate = state["total_successes"] / max(state["total_executions"], 1)
        valid = [s for s in scores if s is not None]
        score_str = f"scores={[f'{s:.4f}' for s in valid]}" if valid else "no valid scores"
        print(f"    [real-exec] rewards={[f'{r:.3f}' for r in rewards]} | {score_str} | "
              f"success_rate={success_rate:.0%} | best={state['best_score']:.4f}")

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_server(model_path: str, port: int, gpu_id: int) -> subprocess.Popen:
    """Start a vLLM server on a specific GPU."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "16384",
        "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.40",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    log_path = f"/home/jarnav/MLScientist/air-agent/outputs/vllm_executor_{port}.log"
    log_file = open(log_path, "w")
    print(f"[vLLM] Starting executor on GPU {gpu_id}, port {port}...")
    print(f"[vLLM] Log: {log_path}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    # Wait for ready
    import urllib.request
    for i in range(60):
        time.sleep(5)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            print(f"[vLLM] Ready after {(i+1)*5}s")
            return proc
        except Exception:
            pass

    raise RuntimeError(f"vLLM failed to start on port {port} after 300s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO with real MLGym execution rewards")
    parser.add_argument("--model-path", required=True, help="Policy model (scientist)")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task", required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--executor-port", type=int, default=8001)
    parser.add_argument("--executor-gpu", type=int, default=1)
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="grpo-scientist-real")
    parser.add_argument("--full-ft", action="store_true", help="Full finetuning instead of LoRA")
    parser.add_argument("--reward-scale", default="tanh", choices=["tanh", "tiered"],
                        help="'tanh': continuous [-1,1]. 'tiered': 0/0.2/1.0 based on improvement over baseline.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_cfg = TASK_MAP[args.task]
    task_profile = get_task_profile(task_cfg["task_config"])

    print(f"[GRPO-Real] Task: {args.task}")
    print(f"[GRPO-Real] Policy model: {args.model_path} (GPU {args.policy_gpu})")
    print(f"[GRPO-Real] Executor: {args.executor_model} (GPU {args.executor_gpu})")
    print(f"[GRPO-Real] Steps: {args.num_steps}, G={args.num_generations}, parallel={args.max_parallel}")
    print(f"[GRPO-Real] Total executions: {args.num_steps * args.num_generations}")

    # Start executor vLLM server
    vllm_proc = start_vllm_server(args.executor_model, args.executor_port, args.executor_gpu)

    try:
        executor_url = f"http://localhost:{args.executor_port}/v1"

        # Create reward function
        reward_fn = make_real_execution_reward_fn(
            task_cfg=task_cfg,
            task_profile=task_profile,
            executor_url=executor_url,
            executor_model=args.executor_model,
            max_parallel=args.max_parallel,
            max_actions=args.max_actions,
            reward_scale=args.reward_scale,
        )

        # Dataset
        dataset = Dataset.from_list([{"prompt": task_cfg["prompt"]}] * max(args.num_steps * 2, 100))

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA or full FT
        if args.full_ft:
            peft_config = None
            print("[GRPO-Real] Full finetuning (no LoRA)")
        else:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )

        # GRPO config
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.policy_gpu)
        os.environ["WANDB_PROJECT"] = args.wandb_project

        ft_tag = "fullFT" if args.full_ft else "LoRA"
        run_name = f"real_{args.task}_{ft_tag}_G{args.num_generations}_p{args.max_parallel}"
        # Use Adafactor for full FT (much less memory than AdamW)
        optim = "adafactor" if args.full_ft else "adamw_torch"

        grpo_config = GRPOConfig(
            output_dir=args.output_dir,
            run_name=run_name,
            max_steps=args.num_steps,
            per_device_train_batch_size=args.num_generations,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            optim=optim,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            epsilon=args.epsilon,
            temperature=0.9,
            scale_rewards="group",
            logging_steps=1,
            save_steps=25,
            report_to="wandb",
            seed=args.seed,
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=args.model_path,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        print(f"[GRPO-Real] Starting training...")
        trainer.train()
        trainer.save_model(args.output_dir + "/final")
        print(f"[GRPO-Real] Done! Model saved to {args.output_dir}/final")

    finally:
        print("[GRPO-Real] Stopping executor vLLM...")
        vllm_proc.terminate()
        vllm_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
