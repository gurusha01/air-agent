#!/bin/bash
# Run 2 GRPO experiments on the same node with 4 GPUs.
# Experiment A: GPU 0 (vLLM) + GPU 1 (scientist)
# Experiment B: GPU 2 (vLLM) + GPU 3 (scientist)

#SBATCH --job-name=grpo_dual
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:4
#SBATCH --output=/scratch/jarnav/logs/grpo_dual_%j.log
#SBATCH --error=/scratch/jarnav/logs/grpo_dual_%j.log

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3

# Experiment A params
TASK_A=${TASK_A:-battleOfSexes}
OUTNAME_A=${OUTNAME_A:-bos_4b_exp_A}
MAX_ACTIONS_A=${MAX_ACTIONS_A:-5}
EXTRA_A=${EXTRA_A:-}

# Experiment B params
TASK_B=${TASK_B:-regressionKaggleHousePrice}
OUTNAME_B=${OUTNAME_B:-regr_4b_exp_B}
MAX_ACTIONS_B=${MAX_ACTIONS_B:-10}
EXTRA_B=${EXTRA_B:-}

# Shared params
MAX_EPISODES=${MAX_EPISODES:-50}
K=${K:-16}
STEPS_PER_EP=${STEPS_PER_EP:-3}
SCI_MODEL=${SCI_MODEL:-Qwen/Qwen3-4B-Instruct-2507}

OUTDIR_A=/home/jarnav/MLScientist/air-agent/outputs/ttt_grpo/$OUTNAME_A
OUTDIR_B=/home/jarnav/MLScientist/air-agent/outputs/ttt_grpo/$OUTNAME_B

mkdir -p /scratch/jarnav/logs
echo "[batch] Node: $(hostname), Job: $SLURM_JOB_ID"
echo "[batch] Exp A: $TASK_A ($OUTNAME_A) | Exp B: $TASK_B ($OUTNAME_B)"
echo "[batch] K=$K, Episodes=$MAX_EPISODES, Steps=$STEPS_PER_EP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

set -a; source /home/jarnav/MLScientist/air-agent/.env; set +a
export HF_HOME=/scratch/jarnav/hf_cache

# --- Start 2 vLLM instances ---
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# vLLM A on GPU 0, port 8000
CUDA_VISIBLE_DEVICES=0 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /scratch/jarnav/logs/vllm_dual_A_${SLURM_JOB_ID}.log 2>&1 &

# vLLM B on GPU 2, port 8001
CUDA_VISIBLE_DEVICES=2 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 16384 --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /scratch/jarnav/logs/vllm_dual_B_${SLURM_JOB_ID}.log 2>&1 &

echo "[batch] Waiting for both vLLM instances..."
for i in $(seq 1 30); do
    sleep 10
    A_OK=$(curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && echo 1 || echo 0)
    B_OK=$(curl -sf http://localhost:8001/v1/models > /dev/null 2>&1 && echo 1 || echo 0)
    [ "$A_OK" = "1" ] && [ "$B_OK" = "1" ] && echo "[batch] Both vLLM ready after ${i}x10s" && break
done

source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh
cd /home/jarnav/MLScientist/MLGym
rm -rf "$OUTDIR_A" "$OUTDIR_B"
mkdir -p "$OUTDIR_A" "$OUTDIR_B"

# --- Run experiment A on GPU 1 (background) ---
echo "[batch] Starting Experiment A: $TASK_A"
CUDA_VISIBLE_DEVICES=1 $PYTHON -u -m air.ttt.train \
    --task-config "tasks/${TASK_A}.yaml" \
    --scientist-model "$SCI_MODEL" \
    --executor-model Qwen/Qwen3-4B-Instruct-2507 \
    --executor-url http://localhost:8000/v1 \
    --K "$K" --steps-per-episode "$STEPS_PER_EP" --max-episodes "$MAX_EPISODES" \
    --lr 1e-5 --kl-coeff 0.01 --epsilon-greedy 0.1 \
    --lora-r 16 --lora-alpha 32 \
    --max-actions "$MAX_ACTIONS_A" \
    --env-gpu 0 --scientist-device cuda:0 \
    --image-name aigym/mlgym-agent:latest \
    --output-dir "$OUTDIR_A" \
    --verbose $EXTRA_A \
    > /scratch/jarnav/logs/grpo_expA_${SLURM_JOB_ID}.log 2>&1 &
PID_A=$!

# --- Run experiment B on GPU 3 (background) ---
echo "[batch] Starting Experiment B: $TASK_B"
CUDA_VISIBLE_DEVICES=3 $PYTHON -u -m air.ttt.train \
    --task-config "tasks/${TASK_B}.yaml" \
    --scientist-model "$SCI_MODEL" \
    --executor-model Qwen/Qwen3-4B-Instruct-2507 \
    --executor-url http://localhost:8001/v1 \
    --K "$K" --steps-per-episode "$STEPS_PER_EP" --max-episodes "$MAX_EPISODES" \
    --lr 1e-5 --kl-coeff 0.01 --epsilon-greedy 0.1 \
    --lora-r 16 --lora-alpha 32 \
    --max-actions "$MAX_ACTIONS_B" \
    --env-gpu 0 --scientist-device cuda:0 \
    --image-name aigym/mlgym-agent:latest \
    --output-dir "$OUTDIR_B" \
    --verbose $EXTRA_B \
    > /scratch/jarnav/logs/grpo_expB_${SLURM_JOB_ID}.log 2>&1 &
PID_B=$!

echo "[batch] Both experiments running (PID_A=$PID_A, PID_B=$PID_B)"
wait $PID_A; echo "[batch] Exp A exit: $?"
wait $PID_B; echo "[batch] Exp B exit: $?"
echo "[batch] Both done."
ls -la "$OUTDIR_A/" "$OUTDIR_B/" 2>/dev/null
