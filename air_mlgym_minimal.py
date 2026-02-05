"""
Minimal MLGym environment for testing - no Docker, just echoes responses.
"""

import os
import sys
import traceback

# Debug logging to file
_debug_log = "/tmp/air_mlgym_minimal_debug.log"
with open(_debug_log, "a") as f:
    f.write(f"\n=== Module import (pid={os.getpid()}) ===\n")
    f.write(f"sys.path: {sys.path[:5]}...\n")
    f.flush()

try:
    from datasets import Dataset
    import verifiers as vf
    with open(_debug_log, "a") as f:
        f.write("Imports successful\n")
except Exception as e:
    with open(_debug_log, "a") as f:
        f.write(f"Import error: {e}\n")
        f.write(traceback.format_exc())
    raise


def load_environment(
    task_configs: list[str] | None = None,
    max_turns: int = 15,
    num_examples_per_task: int = 10,
    **kwargs,
) -> vf.Environment:
    """Load minimal test environment."""
    with open(_debug_log, "a") as f:
        f.write(f"\nload_environment called (pid={os.getpid()})\n")
        f.write(f"  task_configs={task_configs}\n")
        f.write(f"  max_turns={max_turns}\n")
        f.write(f"  num_examples_per_task={num_examples_per_task}\n")
        f.flush()

    try:
        if task_configs is None:
            task_configs = ["titanic.yaml"]

        # Create simple dataset
        data = []
        for i in range(num_examples_per_task):
            data.append({
                "prompt": [
                    {"role": "system", "content": "You are an ML research agent."},
                    {"role": "user", "content": f"Complete the titanic task (example {i})."},
                ],
                "task": "titanic",  # Must match env config name
                "info": {"task_name": "titanic"},
            })

        dataset = Dataset.from_list(data)

        class MinimalMLGymEnv(vf.MultiTurnEnv):
            """Minimal environment that just echoes responses."""

            @vf.stop
            async def check_done(self, state: vf.State) -> bool:
                """Stop after 3 turns or on 'submit'."""
                if not state.get("trajectory"):
                    return False

                last_step = state["trajectory"][-1]
                last_response = last_step.get("completion", "")

                if "submit" in last_response.lower():
                    return True

                return len(state["trajectory"]) >= 3

            async def env_response(
                self,
                messages: vf.Messages,
                state: vf.State,
                **kwargs,
            ) -> vf.Messages:
                """Just echo a simple observation."""
                turn = len(state.get("trajectory", []))
                return [{"role": "user", "content": f"Observation: Turn {turn} completed. Type 'submit' when done."}]

        def simple_reward(completion, state, **kwargs):
            """Simple reward function."""
            return 0.5

        rubric = vf.Rubric(funcs=[simple_reward], weights=[1.0])

        env = MinimalMLGymEnv(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
        )

        with open(_debug_log, "a") as f:
            f.write(f"Environment created successfully (pid={os.getpid()})\n")
            f.write(f"  Dataset size: {len(dataset)}\n")
            f.flush()

        return env

    except Exception as e:
        with open(_debug_log, "a") as f:
            f.write(f"Error in load_environment: {e}\n")
            f.write(traceback.format_exc())
            f.flush()
        raise
