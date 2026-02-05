"""
Integration test for MLGym + prime-rl.

This test verifies the full pipeline:
1. Environment loading
2. Docker container creation
3. Action execution
4. Reward computation

Run with: cd air-agent && uv run python tests/test_integration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from air.mlgym_env import load_environment, MLGymEnvironment


def test_environment_with_docker():
    """Test MLGym environment with real Docker container."""
    print("=" * 60)
    print("MLGym Integration Test")
    print("=" * 60)

    # Load environment with titanic task (simple, fast)
    print("\n1. Loading environment...")
    env = load_environment(
        task_configs=["titanic.yaml"],
        max_turns=5,
        num_examples_per_task=1,
        env_gpu="0",
        save_trajectories=True,
        trajectory_dir="outputs/test_trajectories",
    )
    print(f"   Environment loaded: {type(env).__name__}")
    print(f"   Dataset size: {len(env.dataset)}")
    print(f"   Max turns: {env.max_turns}")

    # Get first example
    example = env.dataset[0]
    print(f"\n2. Test example:")
    print(f"   Task: {example['task']}")
    print(f"   Baseline: {example['info'].get('baseline_scores', {})}")

    # Create a mock state for testing
    state = {
        "example_id": "test_001",
        "info": example["info"],
        "trajectory": [],
    }

    print("\n3. Creating Docker container...")
    try:
        mlgym_env = env._get_or_create_env(state)
        print(f"   Container created successfully!")
        print(f"   Container type: {type(mlgym_env).__name__}")
    except Exception as e:
        print(f"   ERROR creating container: {e}")
        print("\n   Make sure Docker is running and the image is pulled:")
        print("   docker pull aigym/mlgym-agent:latest")
        return False

    # Test a simple action
    print("\n4. Testing action execution...")
    try:
        # Try opening a file (safe action)
        obs, reward, done, info = mlgym_env.step("ls")
        print(f"   Action: ls")
        print(f"   Observation (first 200 chars): {obs[:200] if obs else 'None'}...")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
    except Exception as e:
        print(f"   ERROR executing action: {e}")
        return False

    # Clean up
    print("\n5. Cleaning up...")
    try:
        env.close_all_containers()
        print("   Containers closed successfully!")
    except Exception as e:
        print(f"   Warning during cleanup: {e}")

    print("\n" + "=" * 60)
    print("Integration test PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_environment_with_docker()
    sys.exit(0 if success else 1)
