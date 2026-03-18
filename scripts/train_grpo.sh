#!/bin/bash
# GRPO scientist training — parameterized.
# Pass: TASK (yaml name without ext), OUTNAME (output subdir), MAX_EPISODES, MAX_ACTIONS
# Defaults: battleOfSexes, 50 episodes, 10 max_actions

#SBATCH --job-name=grpo_train
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:2
#SBATCH --output=/scratch/jarnav/logs/grpo_%x_%j.log
#SBATCH --error=/scratch/jarnav/logs/grpo_%x_%j.log

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3

TASK=${TASK:-battleOfSexes}
OUTNAME=${OUTNAME:-${TASK}_v2}
MAX_EPISODES=${MAX_EPISODES:-50}
MAX_ACTIONS=${MAX_ACTIONS:-10}
STEPS_PER_EP=${STEPS_PER_EP:-5}
K=${K:-4}
SCI_MODEL=${SCI_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

OUTDIR=/home/jarnav/MLScientist/air-agent/outputs/ttt_grpo/$OUTNAME

mkdir -p /scratch/jarnav/logs
echo "[batch] Node: $(hostname), Job: $SLURM_JOB_ID"
echo "[batch] Task=$TASK, Episodes=$MAX_EPISODES, MaxActions=$MAX_ACTIONS, K=$K, Steps=$STEPS_PER_EP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

set -a; source /home/jarnav/MLScientist/air-agent/.env; set +a
export HF_HOME=/scratch/jarnav/hf_cache

# --- vLLM on GPU 0 ---
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /scratch/jarnav/logs/vllm_grpo_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "[batch] Waiting for vLLM..."
for i in $(seq 1 30); do
    sleep 10
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && echo "[batch] vLLM ready" && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "[batch] vLLM died"; tail -20 /scratch/jarnav/logs/vllm_grpo_${SLURM_JOB_ID}.log; exit 1; }
done

source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh
cd /home/jarnav/MLScientist/MLGym
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "[batch] Starting GRPO training..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u -m air.ttt.train \
    --task-config "tasks/${TASK}.yaml" \
    --scientist-model "$SCI_MODEL" \
    --executor-model Qwen/Qwen3-4B-Instruct-2507 \
    --executor-url http://localhost:8000/v1 \
    --K "$K" --steps-per-episode "$STEPS_PER_EP" --max-episodes "$MAX_EPISODES" \
    --lr 1e-5 --kl-coeff 0.01 --epsilon-greedy 0.1 \
    --lora-r 16 --lora-alpha 32 \
    --max-actions "$MAX_ACTIONS" \
    --env-gpu 0 --scientist-device cuda:0 \
    --image-name aigym/mlgym-agent:latest \
    --output-dir "$OUTDIR" \
    --verbose $EXTRA_FLAGS 2>&1

echo "[batch] Training exit code: $?"
ls -la "$OUTDIR/" 2>/dev/null
