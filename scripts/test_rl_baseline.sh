#!/bin/bash
# Test RL baseline training with distrax fix
# Runs the baseline code directly in a container for one task

#SBATCH --job-name=test_rl_base
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=/scratch/jarnav/logs/test_rl_%j.log
#SBATCH --error=/scratch/jarnav/logs/test_rl_%j.log

TASK=${TASK:-rlMountainCarContinuous}
PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3

mkdir -p /scratch/jarnav/logs
echo "[test] Node: $(hostname), Task: $TASK"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/jarnav/hf_cache

source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh
cd /home/jarnav/MLScientist/MLGym

echo "[test] Creating container and running baseline..."
$PYTHON -u -c "
import os, sys
os.chdir('$PWD')

from air.tree_search import ContainerManager, get_task_profile

task = '$TASK'
tp = get_task_profile(f'tasks/{task}.yaml')
cm = ContainerManager(f'tasks/{task}.yaml', '0', 'aigym/mlgym-agent:latest', task_profile=tp)
cm.create()

print(f'Baseline: {cm.baseline_score}')
print()

# Run the baseline training script
print('Running baseline training...')
obs = cm.communicate('cd /home/agent/workspace && ls src/', timeout=10)
print(f'Workspace: {obs}')

# Install deps
print('Installing requirements...')
obs = cm.communicate('cd /home/agent/workspace && pip install -r requirements.txt 2>&1 | tail -5', timeout=300)
print(f'Install: {obs}')

# Run training
print('Training...')
obs = cm.communicate('cd /home/agent/workspace && python src/train.py 2>&1 | tail -20', timeout=1800)
print(f'Training output: {obs}')

# Evaluate
print('Evaluating...')
obs, info = cm.step('validate')
print(f'Validate: {obs}')
print(f'Info: {info}')

cm.close()
" 2>&1

echo "[test] Done."
