#!/bin/bash
# GRPO scientist training on battleOfSexes.
# GPU 0: vLLM (executor), GPU 1: Scientist LoRA training

#SBATCH --job-name=grpo_bos
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:2
#SBATCH --output=/scratch/jarnav/logs/grpo_bos_%j.log
#SBATCH --error=/scratch/jarnav/logs/grpo_bos_%j.log

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3
OUTDIR=/home/jarnav/MLScientist/air-agent/outputs/ttt_grpo/battleOfSexes_v1

mkdir -p /scratch/jarnav/logs
echo "[batch] Node: $(hostname), Job: $SLURM_JOB_ID, GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null | head -20 || echo "nvidia-smi not available"

set -a; source /home/jarnav/MLScientist/air-agent/.env; set +a
export HF_HOME=/scratch/jarnav/hf_cache

# --- Start vLLM on GPU 0 ---
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /scratch/jarnav/logs/vllm_grpo_bos_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "[batch] Waiting for vLLM (PID=$VLLM_PID)..."
for i in $(seq 1 30); do
    sleep 10
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[batch] vLLM ready after ${i}x10s"; break
    fi
    kill -0 $VLLM_PID 2>/dev/null || {
        echo "[batch] vLLM died"; tail -30 /scratch/jarnav/logs/vllm_grpo_bos_${SLURM_JOB_ID}.log; exit 1
    }
done

if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[batch] vLLM not ready after 300s"; exit 1
fi

# --- Apptainer shim ---
source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh

cd /home/jarnav/MLScientist/MLGym
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "[batch] Starting GRPO training..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u -m air.ttt.train \
    --task-config tasks/battleOfSexes.yaml \
    --scientist-model Qwen/Qwen3-4B-Instruct-2507 \
    --executor-model Qwen/Qwen3-4B-Instruct-2507 \
    --executor-url http://localhost:8000/v1 \
    --K 4 --steps-per-episode 5 --max-episodes 15 \
    --lr 1e-5 --kl-coeff 0.01 --epsilon-greedy 0.1 \
    --lora-r 16 --lora-alpha 32 \
    --load-in-4bit \
    --max-actions 15 \
    --env-gpu 0 --scientist-device cuda:0 \
    --image-name aigym/mlgym-agent:latest \
    --output-dir "$OUTDIR" \
    --verbose 2>&1

echo "[batch] Training exit code: $?"
ls -la "$OUTDIR/" 2>/dev/null
