"""veRL PPO training orchestrator for MLGym environments."""

from __future__ import annotations

from pathlib import Path

import ray
from mlgym.environment.env import EnvironmentArguments
from mlgym.environment.registration import register_task
from mlgym.utils.config import load_environment_variables
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from air.rollout import MLGymDataset, load_agent_config


class MLGymTaskRunner(BaseTaskRunner):
    """Custom TaskRunner that uses MLGym environments for PPO training."""

    def __init__(self):
        super().__init__()
        self.mlgym_config = None
        self.agent_config = None

    def run(self, config):
        """Execute PPO training with MLGym environments."""
        # Standard veRL setup
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        from verl.utils.fs import copy_to_local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path, config.actor_rollout_ref.model, trust_remote_code=True)
        processor = hf_processor(local_path, config.actor_rollout_ref.model, trust_remote_code=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Load MLGym config
        self.mlgym_config = config.get("mlgym", {})
        self.agent_config = load_agent_config(Path(self.mlgym_config["agent_config_path"]))

        # Register MLGym task
        env_args = EnvironmentArguments(
            task_config_path=Path(self.mlgym_config["task_config_path"]),
            container_type="docker",
            max_steps=self.mlgym_config.get("max_steps", 50),
            seed=42,
            verbose=False,
        )
        register_task(env_args)

        # Create MLGym dataset
        train_dataset = MLGymDataset(
            task=self.mlgym_config["task"],
            agent_config=self.agent_config,
            tokenizer=tokenizer,
            num_episodes=config.data.train_files,  # Reuse this config field
        )

        # Simple reward function (already computed in dataset)
        def reward_fn(data):
            return data

        # Initialize trainer
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=None,
            train_dataset=train_dataset,
            val_dataset=None,
            collate_fn=None,
            train_sampler=None,
        )

        trainer.init_workers()
        trainer.fit()


def main():
    """Main training loop."""
    load_environment_variables()

    # Create veRL config with MLGym-specific fields
    config = OmegaConf.create({
        # MLGym-specific config
        "mlgym": {
            "task": "battleOfSexes",
            "task_config_path": "/home/ubuntu/MLScientist/MLGym/configs/tasks/battleOfSexes.yaml",
            "agent_config_path": "/home/ubuntu/MLScientist/MLGym/configs/agents/default.yaml",
            "max_steps": 50,
        },

        # Standard veRL PPO config (minimal)
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen3-4B-Instruct-2507",
                "use_shm": False,
            },
            "actor": {
                "strategy": "fsdp",
            },
            "rollout": {
                "log_prob_micro_batch_size": 4,
                "tensor_model_parallel_size": 1,
            },
        },

        "critic": {
            "strategy": "fsdp",
        },

        "reward_model": {
            "enable": False,
            "enable_resource_pool": False,
        },

        "trainer": {
            "total_epochs": 10,
            "n_gpus_per_node": 8,
            "nnodes": 1,
            "use_legacy_worker_impl": "auto",
        },

        "algorithm": {
            "kl_ctrl": {
                "kl_coef": 0.1,
            },
        },

        "data": {
            "train_files": 16,  # Number of episodes to collect
        },

        "ray_kwargs": {
            "ray_init": {
                "num_cpus": 32,
            },
        },

        "transfer_queue": {
            "enable": False,
        },
    })

    # Create custom task runner
    task_runner_class = ray.remote(num_cpus=1)(MLGymTaskRunner)

    # Run PPO with custom runner
    print("Starting PPO training with MLGym...")
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
