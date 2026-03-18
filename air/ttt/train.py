"""CLI entry point for scientist GRPO training.

Usage:
    # Terminal 1: Start vLLM for the executor
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
        --max-model-len 16384 --max-num-seqs 8 \
        --gpu-memory-utilization 0.90

    # Terminal 2: GRPO training (scientist model on second GPU)
    cd /home/jarnav/MLScientist/MLGym
    CUDA_VISIBLE_DEVICES=1 python -m air.ttt.train \
        --task-config tasks/battleOfSexes.yaml \
        --scientist-model Qwen/Qwen3-4B-Instruct-2507 \
        --executor-url http://localhost:8000/v1 \
        --K 4 --steps-per-episode 5 --max-episodes 15 \
        --env-gpu 0 --output-dir outputs/ttt_grpo/battleOfSexes_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load .env for API keys (if any)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from air.tree_search import LLMClient
from air.ttt.grpo_trainer import GRPOTrainer, ScientistModel
from air.ttt.scientist_env import ParallelScientistEnv


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for the scientist model (step-level)",
    )

    # Task
    parser.add_argument("--task-config", default="tasks/battleOfSexes.yaml",
                        help="Task YAML (relative to MLGym/configs/)")

    # Models
    parser.add_argument("--scientist-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="HuggingFace model ID for the scientist (loaded locally with LoRA)")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name for the executor (served via vLLM)")
    parser.add_argument("--executor-url", default="http://localhost:8000/v1",
                        help="vLLM endpoint for the executor")

    # GRPO
    parser.add_argument("--K", type=int, default=4, help="Group size for GRPO")
    parser.add_argument("--steps-per-episode", type=int, default=5,
                        help="Scientist decisions per episode (node budget)")
    parser.add_argument("--max-episodes", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--epsilon-greedy", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=4)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="QLoRA: load base model in 4-bit (fits in 20GB MIG)")

    # Environment
    parser.add_argument("--max-actions", type=int, default=15,
                        help="Max executor actions per node")
    parser.add_argument("--env-gpu", default="0",
                        help="GPU for MLGym containers (training environment)")
    parser.add_argument("--scientist-device", default="cuda:0",
                        help="Device for scientist model (LoRA training)")
    parser.add_argument("--image-name", default="aigym/mlgym-agent:latest")

    # Reward weights
    parser.add_argument("--w-explore", type=float, default=0.3)
    parser.add_argument("--w-exploit", type=float, default=0.5)
    parser.add_argument("--w-memory", type=float, default=0.2)
    parser.add_argument("--reward-mode", default="granular",
                        choices=["granular", "binary"],
                        help="Reward mode: 'granular' (weighted components) or 'binary' (1 if above baseline+eps)")
    parser.add_argument("--reward-epsilon", type=float, default=0.0,
                        help="Epsilon for binary reward: score must exceed baseline + eps")
    parser.add_argument("--verbalized-sampling", action="store_true",
                        help="Use verbalized sampling for diverse strategy generation")

    # Output
    parser.add_argument("--output-dir", default="outputs/ttt_grpo")
    parser.add_argument("--resume-from", default="",
                        help="Path to LoRA checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Scientist Training")
    print(f"  Task: {args.task_config}")
    print(f"  Scientist: {args.scientist_model} (LoRA r={args.lora_r})")
    print(f"  Executor: {args.executor_model} @ {args.executor_url}")
    print(f"  K={args.K}, episodes={args.max_episodes}, steps={args.steps_per_episode}")
    print(f"  LR={args.lr}, KL={args.kl_coeff}, ε-greedy={args.epsilon_greedy}")
    print(f"  Reward weights: explore={args.w_explore}, exploit={args.w_exploit}, memory={args.w_memory}")
    print("=" * 60)

    # 1. Load scientist model with LoRA
    scientist = ScientistModel(
        model_name=args.scientist_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.scientist_device,
        load_in_4bit=args.load_in_4bit,
    )
    if args.resume_from:
        scientist.load_lora(args.resume_from)

    # 2. Create executor client (via vLLM)
    executor = LLMClient(
        base_url=args.executor_url,
        model=args.executor_model,
        temperature=0.7,
    )

    # 3. Create parallel environment
    env = ParallelScientistEnv(
        K=args.K,
        task_config=args.task_config,
        env_gpu=args.env_gpu,
        image_name=args.image_name,
        executor=executor,
        max_actions=args.max_actions,
        node_budget=args.steps_per_episode,
        verbose=args.verbose,
    )
    env.create_all()

    # 4. Create trainer
    trainer = GRPOTrainer(
        scientist_model=scientist,
        env=env,
        K=args.K,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        epsilon_greedy=args.epsilon_greedy,
        reward_weights=(args.w_explore, args.w_exploit, args.w_memory),
        reward_mode=args.reward_mode,
        reward_epsilon=args.reward_epsilon,
        use_verbalized_sampling=args.verbalized_sampling,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        steps_per_episode=args.steps_per_episode,
        gradient_accumulation_steps=args.grad_accum,
    )

    # 5. Train
    try:
        summary = trainer.train()
        print(f"\nTraining complete. Results in {args.output_dir}/")
    finally:
        env.close_all()


if __name__ == "__main__":
    main()
