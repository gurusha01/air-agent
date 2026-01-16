"""Minimal orchestrator for iterative model training and trajectory collection."""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from air.run import Main, ScriptArguments
from mlgym.agent.base import AgentArguments
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments
from mlgym.utils.config import load_environment_variables


class ModelVersion:
    """Track model version, base model, and LoRA adapter path."""

    def __init__(self, version_file: Path, base_model: str):
        self.version_file = version_file
        self.base_model = base_model
        if not version_file.exists():
            # Start with base model, no LoRA adapter
            self._save({"version": 0, "base_model": base_model, "lora_adapter": None})

    def get(self) -> dict:
        return json.loads(self.version_file.read_text())

    def increment(self, new_lora_adapter: str) -> None:
        """Increment version and set new LoRA adapter path."""
        current = self.get()
        self._save({
            "version": current["version"] + 1,
            "base_model": self.base_model,
            "lora_adapter": new_lora_adapter,
        })

    def _save(self, data: dict) -> None:
        self.version_file.write_text(json.dumps(data, indent=2))


class VLLMServer:
    """Manage vLLM server lifecycle with LoRA support."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):  # noqa: S104
        self.host = host
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(
        self,
        base_model: str,
        lora_adapter: str | None = None,
        gpu_ids: str | None = None,  # e.g., "0" or "0,1"
    ) -> None:
        """Start vLLM server with base model and optional LoRA adapter, optionally specifying GPU(s) via vllm CLI."""
        print(f"Starting vLLM server with base model: {base_model}")
        cmd = [
            "vllm", "serve", base_model,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", "8192",
        ]
        if lora_adapter is not None:
            print(f"  + LoRA adapter: {lora_adapter}")
            cmd.extend([
                "--enable-lora",
                "--lora-modules", f"default={lora_adapter}",
            ])
        if gpu_ids is not None:
            print(f"  + Limiting vLLM to GPU(s): {gpu_ids}")
            cmd.extend(["--gpu-ids", str(gpu_ids)])
        self.process = subprocess.Popen(cmd)
        time.sleep(120)  # Wait for server startup
        print(f"vLLM server started (PID: {self.process.pid})")

    def stop(self) -> None:
        """Kill vLLM server."""
        if self.process:
            print(f"Stopping vLLM server (PID: {self.process.pid})")
            self.process.terminate()
            self.process.wait(timeout=10)
            self.process = None
            time.sleep(120)
            print("vLLM server stopped")



def collect_trajectories(
    num_trajectories: int,
    model_version: int,
    task: str,
    exp_name: str,
    trajectories_base: Path,
) -> Path:
    """Collect trajectories and save to organized directory structure."""
    # Create directory structure: {base}/{exp_name}/v{version}/{task}/
    save_dir = trajectories_base / exp_name / f"v{model_version}" / task
    save_dir.mkdir(parents=True, exist_ok=True)

    args = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path=Path(f"/home/ubuntu/MLScientist/MLGym/configs/tasks/{task}.yaml"),
            container_type="docker",
            max_steps=50,
            seed=42,
            verbose=True,
            aliases_file="/home/ubuntu/MLScientist/MLGym/dockerfiles/aliases.sh",
        ),
        agent=AgentArguments(
            model=ModelArguments(
                model_name="litellm:hosted_vllm/Qwen/Qwen3-4B-Instruct-2507",
                per_instance_cost_limit=4.0,
                temperature=1.0,
                top_p=0.95,
                host_url="http://0.0.0.0:8000/v1",
            ),
            agent_config_path=Path("/home/ubuntu/MLScientist/MLGym/configs/agents/default.yaml"),
            log_verbose_to_console=True,
        ),
        num_agents=num_trajectories,
        gpus_per_agent=1,
        gpus=[0, 1, 2, 3, 4, 5, 6, 7],
        trajectory_dir=save_dir,  # Pass custom trajectory directory
    )

    print(f"======Generating Trajectories (v{model_version})=======")
    print(f"Save directory: {save_dir}")

    # Run trajectory collection
    asyncio.run(Main(args).main())

    return save_dir



def get_trajectory_reward(traj_file: Path) -> float:
    """Get reward for a trajectory based on task success."""
    try:
        # Check if results.json exists in the same directory
        results_file = traj_file.parent / "results.json"
        if results_file.exists():
            results = json.loads(results_file.read_text())
            # Extract agent score if available
            agent_score = results.get("agent", [])
            if agent_score:
                return float(agent_score[0].get("Score", 0.0))
        # Fallback: random reward
        return float(random.randint(0, 1))
    except Exception:
        return 0.0


def parse_trajectories(trajectories_dir: Path) -> tuple[list[str], list[str], list[float]]:
    """Parse trajectory files and extract queries, responses, and trajectory-level rewards."""
    queries, responses, trajectory_rewards = [], [], []

    # Find all .traj files
    traj_files = list(trajectories_dir.rglob("*.traj"))
    print(f"Found {len(traj_files)} trajectory files")

    for traj_file in traj_files:
        try:
            data = json.loads(traj_file.read_text())
            trajectory = data.get("trajectory", [])
            
            # Get overall trajectory reward
            traj_reward = get_trajectory_reward(traj_file)

            # Extract thought-action pairs as training data
            for step in trajectory:
                thought = step.get("thought", "").strip()
                action = step.get("action", "").strip()

                if thought and action:
                    queries.append(thought)
                    responses.append(action)
                    # Use trajectory-level reward for all steps
                    trajectory_rewards.append(traj_reward)
        except Exception as e:
            print(f"Error parsing {traj_file}: {e}")

    print(f"Extracted {len(queries)} training examples")
    return queries, responses, trajectory_rewards


def reward_function(responses: list[str], stored_rewards: list[float]) -> list[torch.Tensor]:
    """
    Reward function for PPO.
    Takes responses and returns reward tensors.
    
    In real use: this would analyze response quality.
    For now: uses pre-computed rewards from trajectory results.
    """
    return [torch.tensor(r) for r in stored_rewards]


def train_model(
    ppo_trainer: PPOTrainer,
    trajectories_dir: Path,
    version: int,
    exp_name: str,
) -> Path:
    """Train LoRA adapter using PPO on collected trajectories."""
    print(f"Training LoRA adapter on trajectories from {trajectories_dir}...")

    # Parse trajectories
    queries, responses, rewards = parse_trajectories(trajectories_dir)

    if len(queries) == 0:
        print("No training data found, skipping training")
        return None

    # Convert to tensors
    tokenizer = ppo_trainer.tokenizer
    query_tensors = [tokenizer.encode(q, return_tensors="pt")[0] for q in queries]
    response_tensors = [tokenizer.encode(r, return_tensors="pt")[0] for r in responses]

    # Get rewards using reward function
    reward_tensors = reward_function(responses, rewards)

    # Training step
    print(f"Running PPO step on {len(queries)} examples...")
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    print(f"Training stats: {stats}")

    # Save adapter
    output_dir = Path(f"checkpoints/{exp_name}/v{version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_trainer.model.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")

    return output_dir




def main() -> None:
    """Main orchestration loop."""
    load_environment_variables()

    # Configuration
    base_model = "Qwen/Qwen3-4B-Instruct-2507"  # HuggingFace model
    version_file = Path("model_version.json")
    trajectories_base = Path("trajectories")  # Base directory for all trajectories
    task = "battleOfSexes"  # Task name
    num_trajectories = 5
    num_iterations = 3
    exp_name = "base"

    # Initialize version tracker and vLLM server
    version_tracker = ModelVersion(version_file, base_model)
    vllm_server = VLLMServer()

    # Initialize PPO Trainer (once at the start)
    print("Initializing PPO Trainer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Reference model (frozen copy for KL divergence)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )

    # Create PPO config and trainer
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=2,
    )
    
    # Empty dataset (we'll use step() directly)
    from datasets import Dataset
    dummy_dataset = Dataset.from_dict({"query": [""], "response": [""]})
    
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        reward_model = reward_function,
        train_dataset=dummy_dataset,
        value_model=None,
    )
    print("PPO Trainer initialized")

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Get current model info
        model_info = version_tracker.get()
        model_version = model_info["version"]
        base_model_path = model_info["base_model"]
        lora_adapter = model_info["lora_adapter"]

        # 1. Start vLLM with base model + optional LoRA adapter
        vllm_server.start(base_model_path, lora_adapter, gpu_ids="1")

        # 2. Collect N trajectories with organized directory structure
        traj_save_dir = collect_trajectories(
            num_trajectories, model_version, task, exp_name, trajectories_base
        )

        # 3. Stop trajectory collection (implicit)
        print("Stopping trajectory collection")

        # 4. Kill vLLM server
        vllm_server.stop()

        # 5. Train LoRA adapter using PPO
        new_adapter = train_model(ppo_trainer, traj_save_dir, model_version + 1, exp_name)

        # 6. Save new adapter path with incremented version
        if new_adapter:
            version_tracker.increment(str(new_adapter))

        # 7. Restart vLLM with new adapter
        # (will happen in next iteration's step 1)

        # 8. Verify model (dummy - would happen after restart)
        # verify_model(vllm_server.host, vllm_server.port)

        # 9. Resume trajectory collection (happens in next iteration)

        print(f"\nIteration {iteration + 1} complete. New version: {version_tracker.get()['version']}")

    print(f"\n{'='*60}")
    print("All iterations complete!")


if __name__ == "__main__":
    main()

