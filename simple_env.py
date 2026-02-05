"""
Simple test environment - mirrors alphabet_sort structure exactly.
"""

from datasets import Dataset
import verifiers as vf


def get_dataset_builder(num_examples: int = 10, seed: int = 42):
    """Returns a DatasetBuilder that lazily builds the dataset."""

    def build() -> Dataset:
        data = []
        for i in range(num_examples):
            data.append({
                "prompt": [{"role": "user", "content": f"Say 'done' to finish. Example {i}."}],
                "task": "simple-task",
                "info": {"example_idx": i},
            })
        return Dataset.from_list(data)

    return build


def load_environment(
    num_examples: int = 10,
    max_turns: int = 3,
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """Load simple test environment."""

    class SimpleEnv(vf.MultiTurnEnv):
        @vf.stop
        async def check_done(self, state: vf.State) -> bool:
            if not state.get("trajectory"):
                return False
            # completion is a list of messages, get the last assistant message content
            completion = state["trajectory"][-1].get("completion", [])
            if completion:
                last_msg = [m for m in completion if m.get("role") == "assistant"]
                if last_msg and "done" in last_msg[-1].get("content", "").lower():
                    return True
            return len(state["trajectory"]) >= max_turns

        async def env_response(
            self, messages: vf.Messages, state: vf.State, **kwargs
        ) -> vf.Messages:
            turn = len(state.get("trajectory", []))
            return [{"role": "user", "content": f"Turn {turn}. Say 'done' to finish."}]

    def reward_fn(completion, state, **kwargs):
        # Simple reward - 1.0 if said "done", 0.0 otherwise
        if completion:
            last_msg = [m for m in completion if m.get("role") == "assistant"]
            if last_msg and "done" in last_msg[-1].get("content", "").lower():
                return 1.0
        return 0.0

    rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0])
    dataset_builder = get_dataset_builder(num_examples=num_examples, seed=seed)
    dataset = dataset_builder()

    return SimpleEnv(dataset=dataset, rubric=rubric, max_turns=max_turns)
