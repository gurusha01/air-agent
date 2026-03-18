"""Test RL baseline training with distrax fix."""
import os
import sys

os.chdir("/home/jarnav/MLScientist/MLGym")
sys.path.insert(0, "/home/jarnav/MLScientist/air-agent")

from air.tree_search import ContainerManager, get_task_profile

task = os.environ.get("TASK", "rlMountainCarContinuous")
print(f"=== Testing {task} baseline ===")

tp = get_task_profile(f"tasks/{task}.yaml")
cm = ContainerManager(f"tasks/{task}.yaml", "0", "aigym/mlgym-agent:latest", task_profile=tp)
cm.create()
print(f"Baseline: {cm.baseline_score}")

# Check workspace
obs = cm.communicate("cd /home/agent/workspace && ls src/", timeout=10)
print(f"Workspace src/: {obs.strip()}")

obs = cm.communicate("cd /home/agent/workspace && cat requirements.txt", timeout=10)
print(f"requirements.txt:\n{obs.strip()}")

# Install deps
print("\nInstalling requirements...")
obs = cm.communicate(
    "cd /home/agent/workspace && pip install -r requirements.txt 2>&1 | tail -10",
    timeout=300,
)
print(f"Install output:\n{obs.strip()}")

# Run training
print("\nRunning train.py...")
obs = cm.communicate(
    "cd /home/agent/workspace && python src/train.py 2>&1",
    timeout=1800,
)
# Print last 30 lines
lines = obs.strip().split("\n")
print(f"Training output ({len(lines)} lines, last 30):")
for line in lines[-30:]:
    print(f"  {line}")

# Evaluate
print("\nValidating...")
obs, info = cm.step("validate")
print(f"Validate output: {obs[:500]}")
score = info.get("score")
print(f"Score: {score}")

cm.close()
print(f"\n=== {task} baseline test complete ===")
