"""
MLGym RL Training Visualization

This module provides visualization tools for:
1. Streamlit trajectory viewer (individual episode inspection)
2. W&B integration helpers (training curves)
3. Trajectory analysis utilities

Usage:
    # Run Streamlit visualizer
    cd air-agent && uv run streamlit run air/visualize.py -- --trajectory_dir outputs/trajectories
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st


def load_trajectories(trajectory_dir: str | Path) -> list[dict]:
    """
    Load all trajectory files from a directory.

    Args:
        trajectory_dir: Directory containing trajectory JSON files

    Returns:
        List of trajectory dictionaries
    """
    trajectory_dir = Path(trajectory_dir)
    trajectories = []

    for file in sorted(trajectory_dir.glob("*.json")):
        with open(file) as f:
            traj = json.load(f)
            traj["_filename"] = file.name
            trajectories.append(traj)

    return trajectories


def configure_page_style():
    """Configure Streamlit page style."""
    st.set_page_config(
        page_title="MLGym RL Training Visualizer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #60a5fa;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #94a3b8;
        }
        .step-box {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .agent-action {
            background: #0f172a;
            border-left: 3px solid #3b82f6;
            padding: 10px 15px;
            margin: 5px 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .env-response {
            background: #0f172a;
            border-left: 3px solid #22c55e;
            padding: 10px 15px;
            margin: 5px 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .reward-positive { color: #22c55e; }
        .reward-negative { color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)


def show_metrics_summary(trajectories: list[dict]):
    """Display summary metrics across all trajectories."""
    st.header("Training Summary")

    # Group by task
    task_metrics: dict[str, list] = defaultdict(list)
    for traj in trajectories:
        task = traj.get("task", "unknown")
        metrics = traj.get("metrics", {})
        task_metrics[task].append(metrics)

    # Display per-task metrics
    cols = st.columns(len(task_metrics) if task_metrics else 1)

    for i, (task, metrics_list) in enumerate(task_metrics.items()):
        with cols[i % len(cols)]:
            st.subheader(f"📊 {task}")

            improvements = [m.get("improvement", 0) for m in metrics_list]
            final_scores = [m.get("final_score", 0) for m in metrics_list]
            baseline_scores = [m.get("baseline_score", 0) for m in metrics_list]

            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                avg_final = sum(final_scores) / len(final_scores)
                avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

                st.metric("Avg Improvement", f"{avg_improvement:.4f}",
                         delta=f"+{avg_improvement:.4f}" if avg_improvement > 0 else f"{avg_improvement:.4f}")
                st.metric("Avg Final Score", f"{avg_final:.4f}")
                st.metric("Baseline", f"{avg_baseline:.4f}")
                st.metric("Episodes", len(metrics_list))


def show_improvement_chart(trajectories: list[dict]):
    """Display improvement over time chart."""
    import pandas as pd

    st.header("Improvement Over Episodes")

    data = []
    for i, traj in enumerate(trajectories):
        metrics = traj.get("metrics", {})
        data.append({
            "episode": i,
            "improvement": metrics.get("improvement", 0),
            "final_score": metrics.get("final_score", 0),
            "task": traj.get("task", "unknown"),
        })

    if data:
        df = pd.DataFrame(data)

        # Improvement chart
        st.line_chart(df.set_index("episode")["improvement"], use_container_width=True)


def show_validation_history(trajectories: list[dict]):
    """Display validation score history within episodes."""
    st.header("Validation History (Within Episodes)")

    # Select episode
    episode_options = [f"{traj.get('task', 'unknown')} - {traj.get('_filename', i)}"
                      for i, traj in enumerate(trajectories)]

    if not episode_options:
        st.info("No trajectories found.")
        return

    selected_idx = st.selectbox("Select Episode", range(len(episode_options)),
                                format_func=lambda x: episode_options[x])

    traj = trajectories[selected_idx]
    validation_history = traj.get("metrics", {}).get("validation_history", [])

    if validation_history:
        import pandas as pd

        steps, scores = zip(*validation_history)
        df = pd.DataFrame({"step": steps, "validation_score": scores})
        st.line_chart(df.set_index("step"), use_container_width=True)

        st.write(f"**Final Score:** {traj.get('metrics', {}).get('final_score', 'N/A')}")
        st.write(f"**Baseline:** {traj.get('metrics', {}).get('baseline_score', 'N/A')}")
        st.write(f"**Improvement:** {traj.get('metrics', {}).get('improvement', 'N/A')}")
    else:
        st.info("No validation history for this episode.")


def show_trajectory_viewer(trajectories: list[dict]):
    """Display detailed trajectory viewer."""
    st.header("Trajectory Viewer")

    episode_options = [f"{traj.get('task', 'unknown')} - {traj.get('_filename', i)}"
                      for i, traj in enumerate(trajectories)]

    if not episode_options:
        st.info("No trajectories found.")
        return

    selected_idx = st.selectbox("Select Episode", range(len(episode_options)),
                                format_func=lambda x: episode_options[x],
                                key="trajectory_viewer_select")

    traj = trajectories[selected_idx]
    trajectory_steps = traj.get("trajectory", [])

    # Show episode info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Task", traj.get("task", "unknown"))
    with col2:
        st.metric("Steps", len(trajectory_steps))
    with col3:
        improvement = traj.get("metrics", {}).get("improvement", 0)
        st.metric("Improvement", f"{improvement:.4f}")

    # Step slider
    if trajectory_steps:
        step_idx = st.slider("Step", 0, len(trajectory_steps) - 1, 0)
        step = trajectory_steps[step_idx]

        st.subheader(f"Step {step_idx + 1} / {len(trajectory_steps)}")

        # Show prompt (environment message)
        with st.expander("🔵 Environment Message", expanded=True):
            prompt = step.get("prompt", "")
            if isinstance(prompt, list):
                for msg in prompt:
                    st.markdown(f"**{msg.get('role', 'unknown')}:** {msg.get('content', '')[:500]}...")
            else:
                st.code(prompt[:2000] if len(str(prompt)) > 2000 else prompt)

        # Show completion (agent response)
        with st.expander("🟢 Agent Response", expanded=True):
            completion = step.get("completion", "")
            st.code(completion[:3000] if len(str(completion)) > 3000 else completion)


def main():
    """Main Streamlit application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_dir", type=str, default="outputs/trajectories",
                       help="Directory containing trajectory JSON files")
    args, _ = parser.parse_known_args()

    configure_page_style()

    st.title("🔬 MLGym RL Training Visualizer")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        trajectory_dir = st.text_input("Trajectory Directory", args.trajectory_dir)

        if st.button("Refresh"):
            st.rerun()

    # Load trajectories
    trajectory_path = Path(trajectory_dir)
    if not trajectory_path.exists():
        st.warning(f"Trajectory directory not found: {trajectory_dir}")
        st.info("Run training first to generate trajectories, then refresh.")
        return

    trajectories = load_trajectories(trajectory_path)

    if not trajectories:
        st.warning("No trajectory files found.")
        st.info("Trajectories are saved during training. Run training first.")
        return

    st.success(f"Loaded {len(trajectories)} trajectories")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary",
        "📈 Improvement Chart",
        "🎯 Validation History",
        "🔍 Trajectory Viewer"
    ])

    with tab1:
        show_metrics_summary(trajectories)

    with tab2:
        show_improvement_chart(trajectories)

    with tab3:
        show_validation_history(trajectories)

    with tab4:
        show_trajectory_viewer(trajectories)


if __name__ == "__main__":
    main()
