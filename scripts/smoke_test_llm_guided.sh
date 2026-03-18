#!/bin/bash
#SBATCH --job-name=llm_smoke
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=/tmp/llm_guided_smoke_%j.log
#SBATCH --error=/tmp/llm_guided_smoke_%j.log

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3
LLMG=/home/jarnav/MLScientist/air-agent/air/llm_guided_tree_search.py
OUTBASE=/home/jarnav/MLScientist/air-agent/outputs/Feb28_Experiments/llm_guided/smoke_test

echo "[smoke] Node: $(hostname), Job: $SLURM_JOB_ID"
echo "[smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load API keys
set -a; source /home/jarnav/MLScientist/air-agent/.env; set +a

# --- Start vLLM ---
export HF_HOME=/scratch/jarnav/hf_cache
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /tmp/vllm_smoke.log 2>&1 &
VLLM_PID=$!
echo "[smoke] vLLM PID=$VLLM_PID"

for i in $(seq 1 30); do
    sleep 10
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[smoke] vLLM ready after ${i}x10s"
        break
    fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "[smoke] vLLM died"; tail -20 /tmp/vllm_smoke.log; exit 1; }
    echo "[smoke]   waiting for vLLM... ($i/30)"
done

# --- Activate Apptainer shim ---
source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh
cd /home/jarnav/MLScientist/MLGym
mkdir -p "$OUTBASE"

COMMON=(
    --task-config tasks/battleOfSexes.yaml
    --node-budget 2 --max-actions 5
    --executor-model Qwen/Qwen3-4B-Instruct-2507
    --executor-url http://localhost:8000/v1
    --temperature 0.9
    --env-gpu 1
)

# --- Test A: Qwen3-4B as scientist ---
echo ""
echo "=== SMOKE TEST A: Qwen3-4B scientist + Qwen3-4B executor ==="
$PYTHON $LLMG "${COMMON[@]}" \
    --scientist-model Qwen/Qwen3-4B-Instruct-2507 \
    --scientist-url http://localhost:8000/v1 \
    --output-dir "$OUTBASE/battleofsexes_qwen" \
    > /tmp/smoke_qwen.log 2>&1
RC_A=$?
echo "[smoke] Test A exit=$RC_A"
if [ -f "$OUTBASE/battleofsexes_qwen/result.json" ]; then
    $PYTHON -c "import json,sys; d=json.load(open(sys.argv[1])); print('PASS: score='+str(d['best_score'])+' nodes='+str(d['total_nodes']))" \
        "$OUTBASE/battleofsexes_qwen/result.json"
else
    echo "FAIL: no result.json — last 40 lines of log:"
    tail -40 /tmp/smoke_qwen.log
fi

# --- Test B: o3 as scientist ---
echo ""
echo "=== SMOKE TEST B: o3 scientist + Qwen3-4B executor ==="
$PYTHON $LLMG "${COMMON[@]}" \
    --scientist-model o3 \
    --output-dir "$OUTBASE/battleofsexes_o3" \
    > /tmp/smoke_o3.log 2>&1
RC_B=$?
echo "[smoke] Test B exit=$RC_B"
if [ -f "$OUTBASE/battleofsexes_o3/result.json" ]; then
    $PYTHON -c "import json,sys; d=json.load(open(sys.argv[1])); print('PASS: score='+str(d['best_score'])+' nodes='+str(d['total_nodes']))" \
        "$OUTBASE/battleofsexes_o3/result.json"
else
    echo "FAIL: no result.json — last 40 lines of log:"
    tail -40 /tmp/smoke_o3.log
fi

echo ""
echo "[smoke] Summary: A=$RC_A B=$RC_B"
[ $RC_A -eq 0 ] && [ $RC_B -eq 0 ] && echo "[smoke] ALL PASS" || echo "[smoke] SOME FAILURES"
