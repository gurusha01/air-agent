"""
Tests for MLGym environment integration with prime-rl.

Run with: cd air-agent && uv run pytest tests/test_mlgym_env.py -v
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDatasetBuilder:
    """Test dataset generation."""

    def test_dataset_builder_creates_dataset(self):
        """Test that dataset builder creates valid dataset."""
        from air.mlgym_env import get_dataset_builder

        builder = get_dataset_builder(
            task_configs=["titanic.yaml"],
            num_examples_per_task=5,
            seed=42,
        )
        dataset = builder()

        assert len(dataset) == 5
        assert "prompt" in dataset[0]
        assert "task" in dataset[0]
        assert "info" in dataset[0]

    def test_dataset_builder_multiple_tasks(self):
        """Test dataset builder with multiple tasks."""
        from air.mlgym_env import get_dataset_builder

        builder = get_dataset_builder(
            task_configs=["titanic.yaml", "prisonersDilemma.yaml"],
            num_examples_per_task=3,
            seed=42,
        )
        dataset = builder()

        assert len(dataset) == 6  # 3 per task
        tasks = set(d["task"] for d in dataset)
        assert "titanic" in tasks or "titanicSurvival" in tasks
        assert "prisonersDilemma" in tasks

    def test_dataset_prompt_format(self):
        """Test that prompts have correct format."""
        from air.mlgym_env import get_dataset_builder

        builder = get_dataset_builder(
            task_configs=["titanic.yaml"],
            num_examples_per_task=1,
            seed=42,
        )
        dataset = builder()

        prompt = dataset[0]["prompt"]
        assert isinstance(prompt, list)
        assert len(prompt) >= 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert "validate" in prompt[0]["content"].lower()


class TestRewardFunctions:
    """Test reward computation functions."""

    def test_delta_reward_positive(self):
        """Test delta reward for improvement."""
        from air.mlgym_env import compute_delta_reward

        state = {"last_delta_reward": 0.1}
        reward = compute_delta_reward([], state)

        assert reward > 0
        assert reward <= 1.0

    def test_delta_reward_negative(self):
        """Test delta reward for regression."""
        from air.mlgym_env import compute_delta_reward

        state = {"last_delta_reward": -0.1}
        reward = compute_delta_reward([], state)

        assert reward < 0
        assert reward >= -1.0

    def test_delta_reward_zero(self):
        """Test delta reward when no change."""
        from air.mlgym_env import compute_delta_reward

        state = {}  # No delta stored
        reward = compute_delta_reward([], state)

        assert reward == 0.0

    def test_final_improvement_reward(self):
        """Test final improvement reward calculation."""
        from air.mlgym_env import compute_final_improvement_reward

        state = {
            "last_validation_score": 0.85,
            "baseline_score": 0.70,
        }
        reward = compute_final_improvement_reward([], state)

        assert reward > 0  # Improvement over baseline
        assert reward <= 1.0


class TestTaskMetrics:
    """Test TaskMetrics dataclass."""

    def test_task_metrics_initialization(self):
        """Test TaskMetrics default values."""
        from air.mlgym_env import TaskMetrics

        metrics = TaskMetrics()

        assert metrics.baseline_score == 0.0
        assert metrics.current_score == 0.0
        assert metrics.previous_score == 0.0
        assert metrics.validation_scores == []
        assert metrics.validation_steps == []

    def test_task_metrics_tracking(self):
        """Test TaskMetrics score tracking."""
        from air.mlgym_env import TaskMetrics

        metrics = TaskMetrics(baseline_score=0.7)
        metrics.previous_score = 0.7
        metrics.current_score = 0.75
        metrics.validation_scores.append(0.75)
        metrics.validation_steps.append(5)

        assert metrics.current_score - metrics.previous_score == pytest.approx(0.05)
        assert len(metrics.validation_scores) == 1


class TestEnvironmentLoading:
    """Test environment loading function."""

    def test_load_environment_returns_env(self):
        """Test that load_environment returns valid environment."""
        from air.mlgym_env import MLGymEnvironment, load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                max_turns=10,
                num_examples_per_task=2,
                trajectory_dir=tmpdir,
            )

            assert isinstance(env, MLGymEnvironment)
            assert env.max_turns == 10

    def test_load_environment_default_tasks(self):
        """Test load_environment with default tasks."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=None,  # Should default to titanic
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            assert env.dataset is not None
            assert len(env.dataset) >= 1


class TestMLGymEnvironment:
    """Test MLGymEnvironment class."""

    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                max_turns=20,
                num_examples_per_task=2,
                trajectory_dir=tmpdir,
                save_trajectories=True,
            )

            assert env.max_turns == 20
            assert env.save_trajectories is True
            assert env.trajectory_dir == Path(tmpdir)

    def test_check_done_submit(self):
        """Test check_done detects submit action."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            state = {
                "trajectory": [
                    {"completion": "Let me submit the solution.\n<<SUBMISSION||submission.csv||SUBMISSION>>"}
                ]
            }

            done = run_async(env.check_done(state))
            assert done is True

    def test_check_done_max_turns(self):
        """Test check_done at max turns."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                max_turns=3,
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            state = {
                "trajectory": [
                    {"completion": "action 1"},
                    {"completion": "action 2"},
                    {"completion": "action 3"},
                ]
            }

            done = run_async(env.check_done(state))
            assert done is True

    def test_check_done_not_done(self):
        """Test check_done when not done."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                max_turns=10,
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            state = {
                "trajectory": [
                    {"completion": "open baseline.py"},
                ]
            }

            done = run_async(env.check_done(state))
            assert done is False


class TestTrajectoryVisualization:
    """Test trajectory saving for visualization."""

    def test_save_trajectory_creates_file(self):
        """Test that trajectories are saved correctly."""
        from air.mlgym_env import MLGymEnvironment, TaskMetrics, load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
                save_trajectories=True,
            )

            # Manually set up state and metrics for testing
            state = {
                "example_id": "test_123",
                "info": {"task_name": "titanic"},
                "trajectory": [
                    {"prompt": "Start", "completion": "open baseline.py"},
                    {"prompt": "File contents...", "completion": "validate"},
                ],
            }

            env._metrics["test_123"] = TaskMetrics(
                baseline_score=0.76,
                current_score=0.82,
                previous_score=0.76,
                validation_scores=[0.82],
                validation_steps=[2],
            )

            # Save trajectory
            env._save_trajectory(state, {"exit_status": "submitted"})

            # Check file was created
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            # Check file contents
            with open(files[0]) as f:
                data = json.load(f)

            assert data["task"] == "titanic"
            assert data["metrics"]["baseline_score"] == 0.76
            assert data["metrics"]["final_score"] == 0.82
            assert data["metrics"]["improvement"] == pytest.approx(0.06)


class TestIntegrationWithVerifiers:
    """Test integration with verifiers library."""

    def test_environment_has_required_attributes(self):
        """Test environment has attributes required by verifiers."""
        from air.mlgym_env import load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            # Check required attributes from vf.MultiTurnEnv
            assert hasattr(env, "dataset")
            assert hasattr(env, "rubric")
            assert hasattr(env, "max_turns")
            assert hasattr(env, "env_response")
            assert hasattr(env, "check_done")

    def test_rubric_has_correct_functions(self):
        """Test rubric is set up with delta reward."""
        from air.mlgym_env import compute_delta_reward, load_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            env = load_environment(
                task_configs=["titanic.yaml"],
                num_examples_per_task=1,
                trajectory_dir=tmpdir,
            )

            assert env.rubric is not None
            # MultiTurnEnv wraps rubric in RubricGroup with additional monitor rubric
            # Our rubric is at rubrics[0]
            original_rubric = env.rubric.rubrics[0]
            assert len(original_rubric.funcs) == 1
            assert original_rubric.funcs[0] == compute_delta_reward


class TestModuleImports:
    """Test that all modules import correctly."""

    def test_air_mlgym_import(self):
        """Test air_mlgym module imports."""
        import air_mlgym

        assert hasattr(air_mlgym, "load_environment")
        assert hasattr(air_mlgym, "MLGymEnvironment")
        assert hasattr(air_mlgym, "compute_delta_reward")

    def test_air_package_import(self):
        """Test air package imports."""
        import air

        assert hasattr(air, "load_environment")
        assert hasattr(air, "MLGymEnvironment")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
