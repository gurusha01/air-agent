"""
Simple Trajectory Comparator

Compare two trajectories side-by-side with condensed views.

Usage:
    cd /home/ubuntu/MLScientist/air-agent
    uv run streamlit run air/compare_trajectories.py

Then SSH port forward: ssh -L 8501:localhost:8501 user@server
Open http://localhost:8501 in your browser
"""

import json
from pathlib import Path

import streamlit as st

# Default trajectory directory
DEFAULT_DIR = "/home/ubuntu/MLScientist/MLGym/outputs/trajectories_exp1.4"


def load_trajectories(trajectory_dir: str) -> dict[str, dict]:
    """Load all trajectories as {filename: data}."""
    trajectory_dir = Path(trajectory_dir)
    trajectories = {}

    if not trajectory_dir.exists():
        return trajectories

    for file in sorted(trajectory_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(file) as f:
                trajectories[file.name] = json.load(f)
        except json.JSONDecodeError as e:
            # Skip corrupted files (e.g., from mid-write kills)
            print(f"Skipping corrupted file {file.name}: {e}")
            continue

    return trajectories


def extract_action_short(completion: list) -> str:
    """Extract short action from completion (for display in expander title)."""
    if not completion:
        return "(no action)"

    for msg in completion:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Remove thinking tags
            if "<think>" in content:
                content = content.split("</think>")[-1]
            # Truncate for title
            content = content.strip()[:80]
            if len(content) == 80:
                content += "..."
            return content or "(empty)"

    return "(no assistant msg)"


def extract_action_full(completion: list) -> str:
    """Extract full action from completion (for display in expander body)."""
    if not completion:
        return "(no action)"

    for msg in completion:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Remove thinking tags but keep the rest
            if "<think>" in content:
                content = content.split("</think>")[-1]
            return content.strip() or "(empty)"

    return "(no assistant msg)"


def extract_feedback_from_next_turn(trajectory: list, turn_idx: int) -> str:
    """Extract env feedback from the next turn's prompt (last user message)."""
    if turn_idx + 1 >= len(trajectory):
        return "(end of episode)"

    next_turn = trajectory[turn_idx + 1]
    prompt = next_turn.get("prompt", [])
    if not prompt:
        return "(no prompt)"

    # Find the last user message - that's the env response
    user_msgs = [m for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
    if user_msgs:
        return user_msgs[-1].get("content", "(empty)")

    return "(no user message)"


def display_trajectory(traj: dict, col):
    """Display trajectory in a column."""
    metrics = traj.get("metrics", {})
    trajectory = traj.get("trajectory", [])

    col.markdown(f"**Task:** {traj.get('task', 'unknown')}")
    col.markdown(f"**Policy:** π_{traj.get('policy_step', '?')}")
    col.markdown(f"**Example ID:** {traj.get('example_id', '?')}")
    col.markdown(f"**Turns:** {len(trajectory)}")
    col.markdown(f"**Baseline:** {metrics.get('baseline_score', 'N/A')}")
    col.markdown(f"**Final:** {metrics.get('final_score', 'N/A')}")
    col.markdown(f"**Improvement:** {metrics.get('improvement', 'N/A')}")

    col.markdown("---")
    col.markdown("### Steps")

    for i, turn in enumerate(trajectory):
        completion = turn.get("completion", [])
        reward = turn.get("reward", None)

        action_short = extract_action_short(completion)
        action_full = extract_action_full(completion)
        feedback = extract_feedback_from_next_turn(trajectory, i)

        with col.expander(f"Step {i+1}: {action_short[:40]}{'...' if len(action_short) > 40 else ''}"):
            st.markdown("**Action:**")
            st.code(action_full, language=None)
            st.markdown("**Environment Feedback:**")
            st.code(feedback, language=None)
            if reward is not None:
                st.markdown(f"**Reward:** `{reward}`")


def main():
    st.set_page_config(page_title="Trajectory Comparator", layout="wide")
    st.title("Trajectory Comparator")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        trajectory_dir = st.text_input("Trajectory Directory", DEFAULT_DIR)

        if st.button("Refresh"):
            st.rerun()

    # Load trajectories
    trajectories = load_trajectories(trajectory_dir)

    if not trajectories:
        st.warning(f"No trajectories found in {trajectory_dir}")
        return

    st.success(f"Loaded {len(trajectories)} trajectories")

    # Trajectory selection dropdowns
    traj_names = list(trajectories.keys())

    col1, col2 = st.columns(2)

    with col1:
        selected1 = st.selectbox("Trajectory 1", traj_names, index=0, key="traj1")

    with col2:
        default_idx = min(1, len(traj_names) - 1)
        selected2 = st.selectbox("Trajectory 2", traj_names, index=default_idx, key="traj2")

    st.markdown("---")

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Trajectory 1")
        st.caption(selected1)
        display_trajectory(trajectories[selected1], col1)

    with col2:
        st.header(f"Trajectory 2")
        st.caption(selected2)
        display_trajectory(trajectories[selected2], col2)


if __name__ == "__main__":
    main()
