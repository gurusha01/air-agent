"""
Evaluate the counterfactual-SFT fine-tuned model on tree search tasks.

Loads Qwen3-4B + LoRA adapter, serves it via vLLM (with merged weights),
and runs llm_guided_tree_search on regressionKaggleHousePrice.

Usage:
    python evaluate.py \
        --adapter-path /scratch/jarnav/counterfactual_sft/checkpoints/final \
        --task regressionKaggleHousePrice \
        --node-budget 15 \
        --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save(model_name: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model and save for vLLM serving."""
    print(f"[eval] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Merge on CPU to keep GPU free for vLLM
    )

    print(f"[eval] Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("[eval] Merging weights...")
    model = model.merge_and_unload()

    print(f"[eval] Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("[eval] Merge complete.")
    # Free CPU memory from the merged model
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def start_vllm(model_path: str, port: int = 8000, gpu_util: float = 0.90):
    """Start a vLLM server with the given model and return the process."""
    python = sys.executable
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "32768",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", str(gpu_util),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
    ]
    print(f"[eval] Starting vLLM: {' '.join(cmd)}")
    vllm_log_path = f"/home/jarnav/MLScientist/air-agent/outputs/vllm_eval_{port}_{os.getpid()}.log"
    log_file = open(vllm_log_path, "w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print(f"[eval] vLLM log: {vllm_log_path}, VLLM_ENABLE_V1_MULTIPROCESSING=0")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    # Wait for vLLM to be ready
    import urllib.request
    for i in range(30):
        time.sleep(10)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            print(f"[eval] vLLM ready after {(i+1)*10}s")
            return proc
        except Exception:
            if proc.poll() is not None:
                print("[eval] vLLM process died!")
                log_file.close()
                with open(vllm_log_path) as f:
                    print(f.read()[-2000:])
                sys.exit(1)

    print("[eval] vLLM failed to start within 300s")
    proc.kill()
    sys.exit(1)


def run_tree_search(
    task: str,
    scientist_model: str,
    scientist_url: str,
    executor_model: str,
    executor_url: str,
    node_budget: int,
    output_dir: str,
    container_type: str = "podman",
    env_gpu: str = "cpu",
):
    """Run llm_guided_tree_search as a subprocess."""
    python = sys.executable
    llmg_script = str(
        Path(__file__).resolve().parents[1] / "llm_guided_tree_search.py"
    )
    mlgym_dir = "/home/jarnav/MLScientist/MLGym"

    cmd = [
        python, llmg_script,
        "--task-config", f"tasks/{task}.yaml",
        "--node-budget", str(node_budget),
        "--max-actions", "20",
        "--scientist-model", scientist_model,
        "--scientist-url", scientist_url,
        "--executor-model", executor_model,
        "--executor-url", executor_url,
        "--temperature", "0.9",
        "--env-gpu", env_gpu,
        "--image-name", os.environ.get("MLGYM_APPTAINER_IMAGE", "aigym/mlgym-agent:latest"),
        "--output-dir", output_dir,
    ]
    print(f"[eval] Running tree search: {task}")
    print(f"[eval] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=mlgym_dir)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Evaluate counterfactual SFT model")
    parser.add_argument(
        "--adapter-path", type=str,
        default="/scratch/jarnav/counterfactual_sft/checkpoints/final",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name",
    )
    parser.add_argument(
        "--merged-model-dir", type=str,
        default="/scratch/jarnav/counterfactual_sft/merged_model",
        help="Directory to save the merged model for vLLM",
    )
    parser.add_argument("--task", type=str, default="regressionKaggleHousePrice")
    parser.add_argument("--node-budget", type=int, default=15)
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/jarnav/MLScientist/air-agent/outputs/counterfactual_eval",
        help="Directory for evaluation results",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--container-type", default="podman", choices=["docker", "podman", "apptainer"])
    parser.add_argument("--env-gpu", default="cpu")
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip merging if merged model already exists",
    )
    parser.add_argument(
        "--run-baseline", action="store_true",
        help="Also run baseline (no fine-tuning) for comparison",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of independent tree searches to run and average",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Merge LoRA into base model ---
    merged_dir = Path(args.merged_model_dir)
    if args.skip_merge and merged_dir.exists():
        print(f"[eval] Using existing merged model at {merged_dir}")
    else:
        merge_and_save(args.base_model, args.adapter_path, str(merged_dir))

    # --- Step 2: Start vLLM with fine-tuned model ---
    # Kill any existing vLLM
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"],
                   capture_output=True)
    time.sleep(3)

    # Start two vLLM servers: fine-tuned (scientist) on port, base (executor) on port+1
    scientist_port = args.port
    executor_port = args.port + 1

    vllm_scientist = start_vllm(str(merged_dir), port=scientist_port, gpu_util=0.35)
    vllm_executor = start_vllm(args.base_model, port=executor_port, gpu_util=0.35)

    try:
        ft_scores = []
        baseline_scores = []

        for run_idx in range(1, args.num_runs + 1):
            print(f"\n[eval] === Run {run_idx}/{args.num_runs} ===")

            # --- Fine-tuned scientist ---
            ft_output = str(output_dir / f"finetuned_run{run_idx}")
            Path(ft_output).mkdir(parents=True, exist_ok=True)

            ret = run_tree_search(
                task=args.task,
                scientist_model=str(merged_dir),
                scientist_url=f"http://localhost:{scientist_port}/v1",
                executor_model=args.base_model,
                executor_url=f"http://localhost:{executor_port}/v1",
                node_budget=args.node_budget,
                output_dir=ft_output,
                container_type=args.container_type,
                env_gpu=args.env_gpu,
            )
            print(f"[eval] Fine-tuned run {run_idx} exit code: {ret}")

            ft_result_path = Path(ft_output) / "result.json"
            if ft_result_path.exists():
                with open(ft_result_path) as f:
                    ft_result = json.load(f)
                    s = ft_result.get("best_score")
                    if s is not None:
                        ft_scores.append(s)
                        print(f"[eval] Fine-tuned run {run_idx} score: {s}")

            # --- Baseline ---
            if args.run_baseline:
                bl_output = str(output_dir / f"baseline_run{run_idx}")
                Path(bl_output).mkdir(parents=True, exist_ok=True)

                ret = run_tree_search(
                    task=args.task,
                    scientist_model=args.base_model,
                    scientist_url=f"http://localhost:{executor_port}/v1",
                    executor_model=args.base_model,
                    executor_url=f"http://localhost:{executor_port}/v1",
                    node_budget=args.node_budget,
                    output_dir=bl_output,
                    container_type=args.container_type,
                    env_gpu=args.env_gpu,
                )
                print(f"[eval] Baseline run {run_idx} exit code: {ret}")

                bl_result_path = Path(bl_output) / "result.json"
                if bl_result_path.exists():
                    with open(bl_result_path) as f:
                        bl_result = json.load(f)
                        s = bl_result.get("best_score")
                        if s is not None:
                            baseline_scores.append(s)
                            print(f"[eval] Baseline run {run_idx} score: {s}")

        # --- Summary ---
        import numpy as np
        ft_mean = float(np.mean(ft_scores)) if ft_scores else None
        ft_std = float(np.std(ft_scores)) if len(ft_scores) > 1 else 0.0
        bl_mean = float(np.mean(baseline_scores)) if baseline_scores else None
        bl_std = float(np.std(baseline_scores)) if len(baseline_scores) > 1 else 0.0

        summary = {
            "task": args.task,
            "node_budget": args.node_budget,
            "num_runs": args.num_runs,
            "adapter_path": args.adapter_path,
            "finetuned_scores": ft_scores,
            "finetuned_mean": ft_mean,
            "finetuned_std": ft_std,
            "baseline_scores": baseline_scores,
            "baseline_mean": bl_mean,
            "baseline_std": bl_std,
        }
        if ft_mean is not None and bl_mean is not None:
            summary["improvement"] = ft_mean - bl_mean
            print(f"\n[eval] RESULTS ({args.num_runs} runs):")
            print(f"[eval]   Fine-tuned: {ft_mean:.4f} ± {ft_std:.4f} ({ft_scores})")
            print(f"[eval]   Baseline:   {bl_mean:.4f} ± {bl_std:.4f} ({baseline_scores})")
            print(f"[eval]   Improvement: {summary['improvement']:+.4f}")

        summary_path = output_dir / "comparison.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] Summary saved to {summary_path}")

    finally:
        vllm_scientist.kill()
        vllm_scientist.wait()
        vllm_executor.kill()
        vllm_executor.wait()
        print("[eval] vLLM servers stopped.")


if __name__ == "__main__":
    main()
