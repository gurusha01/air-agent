#!/bin/bash
#SBATCH --job-name=voi_grpo
#SBATCH --account=aip-irina
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/jarnav/MLScientist/air-agent/outputs/voi_grpo_%j.log
#SBATCH --error=/home/jarnav/MLScientist/air-agent/outputs/voi_grpo_%j.log

set -euo pipefail

export HF_HOME=/scratch/jarnav/hf_cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3
MODEL_PATH=${MODEL_PATH:-/scratch/jarnav/scientist_v3_sft/merged_epoch1}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/jarnav/voi_grpo/checkpoints}
STEPS=${STEPS:-200}
NODE_BUDGET=${NODE_BUDGET:-8}

echo "[voi_grpo] Started at $(date) on $(hostname)"
echo "[voi_grpo] Model: $MODEL_PATH"
echo "[voi_grpo] Steps: $STEPS, Node budget: $NODE_BUDGET"

cd /home/jarnav/MLScientist/air-agent

$PYTHON -m air.ttt.VoI_guided_rl.train \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --historical-dir /home/jarnav/scratch/air-agent/outputs \
    --tasks titanic,regression,battleOfSexes \
    --steps $STEPS \
    --node-budget $NODE_BUDGET \
    --grpo-K 8 \
    --voi-K 32 \
    --lr 5e-6 \
    --kl-coeff 0.1 \
    --log-every 5 \
    --save-every 50

echo "[voi_grpo] Done at $(date)"
