"""
air_mlgym - MLGym environment for prime-rl training.

This module provides a verifiers-compatible entry point for loading MLGym environments.
Use with: env_id = "air_mlgym" in prime-rl configs.
"""

import sys
import traceback

# Debug logging for worker subprocess crashes
_debug_log_path = "/tmp/air_mlgym_import.log"

try:
    with open(_debug_log_path, "a") as f:
        f.write(f"\n=== air_mlgym import started (pid={__import__('os').getpid()}) ===\n")
        f.flush()

    from air.mlgym_env import (
        MLGymEnvironment,
        compute_delta_reward,
        compute_final_improvement_reward,
        load_environment,
    )

    with open(_debug_log_path, "a") as f:
        f.write("air_mlgym import successful\n")
        f.flush()

except Exception as e:
    with open(_debug_log_path, "a") as f:
        f.write(f"air_mlgym import FAILED: {e}\n")
        f.write(traceback.format_exc())
        f.flush()
    raise

__all__ = [
    "MLGymEnvironment",
    "compute_delta_reward",
    "compute_final_improvement_reward",
    "load_environment",
]
