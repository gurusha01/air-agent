"""Minimal script to sample one trajectory for a task."""

from __future__ import annotations

import asyncio
from pathlib import Path

from air.run import Main, ScriptArguments
from mlgym.agent.base import AgentArguments
from mlgym.backend.base import ModelArguments
from mlgym.environment.env import EnvironmentArguments
from mlgym.utils.config import load_environment_variables

# Load environment variables (.env file)
load_environment_variables()

# Construct arguments directly using dataclasses
args = ScriptArguments(
    environment=EnvironmentArguments(
        task_config_path=Path("/home/ubuntu/MLScientist/MLGym/configs/tasks/battleOfSexes.yaml"),
        container_type="docker",
        max_steps=50,
        seed=42,
        verbose=True,
        aliases_file="/home/ubuntu/MLScientist/MLGym/dockerfiles/aliases.sh",
    ),
    agent=AgentArguments(
        model=ModelArguments(
            model_name="litellm:hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
            per_instance_cost_limit=4.0,
            temperature=1.0,
            top_p=0.95,
            host_url="http://0.0.0.0:8000/v1",
        ),
        agent_config_path=Path("/home/ubuntu/MLScientist/MLGym/configs/agents/default.yaml"),
        log_verbose_to_console=True,
    ),
    num_agents=5,
    gpus_per_agent=1,
    gpus=[0, 1, 2, 3, 4, 5, 6, 7],
)

# Run the main function
asyncio.run(Main(args).main())

# To Dos: 
# 1. Make a function to update the model and re-serve it at the same local-host.
# 2. Write a reward function based on the trajectories and the results. 
# 3. make a function to extract log probs.
# 4. write a model-update function. 
