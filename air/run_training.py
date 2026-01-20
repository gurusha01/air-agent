#!/usr/bin/env python
"""Run MLGym RL training with prime-rl.

This script demonstrates how to run the full training pipeline:
1. Start the vLLM inference server
2. Run the orchestrator to collect rollouts
3. Run the trainer to update the model

For a simpler standalone run, use:
    uv run python -m air.prime_orchestrator --task battleOfSexes

For full distributed training with prime-rl:
    # Terminal 1: Start inference server
    uv run inference @ configs/mlgym/infer.toml

    # Terminal 2: Start trainer
    uv run trainer @ configs/mlgym/train.toml

    # Terminal 3: Start orchestrator
    uv run orchestrator @ configs/mlgym/orch.toml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path


def check_environment():
    """Check that required environment variables and dependencies are set."""
    issues = []

    # Check MLGYM_CONFIG_ROOT
    mlgym_config_root = os.environ.get("MLGYM_CONFIG_ROOT")
    if not mlgym_config_root:
        # Try to auto-detect
        possible_paths = [
            "/data4/parth/MLGym/configs",
            "../MLGym/configs",
            os.path.expanduser("~/MLGym/configs"),
        ]
        for path in possible_paths:
            if Path(path).exists():
                os.environ["MLGYM_CONFIG_ROOT"] = str(Path(path).resolve())
                print(f"Auto-detected MLGYM_CONFIG_ROOT: {os.environ['MLGYM_CONFIG_ROOT']}")
                break
        else:
            issues.append("MLGYM_CONFIG_ROOT not set and MLGym configs not found")

    # Check for GPU
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU required for training")
        else:
            print(f"Found {torch.cuda.device_count()} GPU(s)")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check for mlgym
    try:
        import mlgym
        print(f"MLGym found: {mlgym.__file__}")
    except ImportError:
        issues.append("MLGym not installed - run 'uv sync' first")

    if issues:
        print("\n[ERROR] Environment issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nEnvironment check passed!")
    return True


def start_vllm_server(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    port: int = 8000,
    gpu_ids: str = "0",
) -> subprocess.Popen:
    """Start the vLLM inference server."""
    print(f"\nStarting vLLM server on port {port}...")

    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "8192",
        "--trust-remote-code",
    ]

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    print("Waiting for server to start...")
    time.sleep(60)  # Give it time to load the model

    return process


async def run_standalone_training(
    task: str = "battleOfSexes",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    num_steps: int = 10,
    batch_size: int = 16,
    output_dir: str = "outputs",
    base_url: str = "http://localhost:8000/v1",
):
    """Run standalone training without the full prime-rl infrastructure.

    This is useful for testing and debugging the MLGym integration.
    """
    from air.prime_orchestrator import run_orchestrator

    print(f"\nRunning standalone training on task: {task}")
    print(f"Model: {model}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")

    await run_orchestrator(
        task=task,
        model_name=model,
        base_url=base_url,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=output_dir,
    )


def run_prime_rl_training(config_dir: str = "configs/mlgym"):
    """Run full prime-rl distributed training.

    This requires starting the inference server, trainer, and orchestrator
    in separate terminals. This function prints instructions.
    """
    print("\n" + "=" * 60)
    print("Prime-RL Distributed Training Setup")
    print("=" * 60)

    print("""
To run full distributed training with prime-rl, open three terminals:

Terminal 1 - Start vLLM Inference Server:
    cd /data4/parth/air-agent
    uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 8192 --trust-remote-code

Terminal 2 - Start RL Trainer:
    cd /data4/parth/air-agent
    uv run trainer @ configs/mlgym/train.toml

Terminal 3 - Start Orchestrator:
    cd /data4/parth/air-agent
    uv run orchestrator @ configs/mlgym/orch.toml

Note: Ensure MLGYM_CONFIG_ROOT is set:
    export MLGYM_CONFIG_ROOT="/data4/parth/MLGym/configs"
""")


def main():
    parser = argparse.ArgumentParser(
        description="Run MLGym RL training with prime-rl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check environment setup")

    # Standalone training command
    standalone_parser = subparsers.add_parser(
        "standalone",
        help="Run standalone training (without full prime-rl)",
    )
    standalone_parser.add_argument("--task", default="battleOfSexes", help="MLGym task")
    standalone_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name")
    standalone_parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    standalone_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    standalone_parser.add_argument("--output-dir", default="outputs", help="Output directory")
    standalone_parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM URL")

    # Distributed training command
    distributed_parser = subparsers.add_parser(
        "distributed",
        help="Show instructions for distributed training",
    )

    # Start server command
    server_parser = subparsers.add_parser("server", help="Start vLLM server")
    server_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    server_parser.add_argument("--gpu-ids", default="0", help="GPU IDs to use")

    args = parser.parse_args()

    if args.command == "check":
        check_environment()

    elif args.command == "standalone":
        if not check_environment():
            sys.exit(1)
        asyncio.run(run_standalone_training(
            task=args.task,
            model=args.model,
            num_steps=args.steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            base_url=args.base_url,
        ))

    elif args.command == "distributed":
        run_prime_rl_training()

    elif args.command == "server":
        process = start_vllm_server(
            model=args.model,
            port=args.port,
            gpu_ids=args.gpu_ids,
        )
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
