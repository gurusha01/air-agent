"""air-agent: AI Research Agents with Reinforcement Learning.

This package provides tools for training AI research agents using
reinforcement learning on MLGym environments with prime-rl.
"""

from air.mlgym_env import MLGymEnvironment, load_environment
from air.prime_orchestrator import MLGymOrchestrator, run_orchestrator

__all__ = [
    "MLGymEnvironment",
    "load_environment",
    "MLGymOrchestrator",
    "run_orchestrator",
]

__version__ = "0.0.1"
