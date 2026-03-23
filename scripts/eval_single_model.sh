#!/bin/bash
# Usage: SCIENTIST_MODEL=... SCIENTIST_LABEL=... EVAL_TASKS="task1 task2" sbatch --job-name=ev_LABEL ...
#SBATCH --account=aip-irina
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
#SBATCH --output=/home/jarnav/MLScientist/air-agent/outputs/eval_%x_%j.log
#SBATCH --error=/home/jarnav/MLScientist/air-agent/outputs/eval_%x_%j.log

set -euo pipefail
export HF_HOME=/scratch/jarnav/hf_cache
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export APPTAINER_TMPDIR=/tmp
export MLGYM_APPTAINER_IMAGE=/scratch/jarnav/mlgym_sandbox
export MLGYM_CONTAINER_TYPE=apptainer

PYTHON=/home/jarnav/MLScientist/air-agent/.venv/bin/python3
LLMG=/home/jarnav/MLScientist/air-agent/air/llm_guided_tree_search.py
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
NODE_BUDGET=${NODE_BUDGET:-5}
N_RUNS=${N_RUNS:-3}

module load apptainer
source /home/jarnav/MLScientist/MLGym/apptainer/activate.sh 2>/dev/null || true

OUTBASE=/home/jarnav/MLScientist/air-agent/outputs/eval_v3_${SCIENTIST_LABEL}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTBASE"
RESULTS="$OUTBASE/results.tsv"

log() { echo "[eval $(date +%H:%M:%S)] $*"; }
log "$SCIENTIST_LABEL | model=$SCIENTIST_MODEL | tasks=$EVAL_TASKS"

# Executor on GPU 0
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" --host 0.0.0.0 --port 8001 \
    --max-model-len 16384 --max-num-seqs 4 --gpu-memory-utilization 0.8 \
    --trust-remote-code --dtype bfloat16 --enforce-eager \
    > "$OUTBASE/vllm_exec.log" 2>&1 &

# Scientist on GPU 1
CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$SCIENTIST_MODEL" --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 --max-num-seqs 4 --gpu-memory-utilization 0.8 \
    --trust-remote-code --dtype bfloat16 --enforce-eager \
    > "$OUTBASE/vllm_sci.log" 2>&1 &

trap "pkill -f 'vllm.entrypoints' 2>/dev/null || true" EXIT

for port in 8000 8001; do
    for i in $(seq 1 60); do
        sleep 10
        curl -sf http://localhost:${port}/v1/models > /dev/null 2>&1 && log "Port $port ready" && break
        [ $i -eq 60 ] && log "ERROR: port $port timeout" && exit 1
    done
done

cd /home/jarnav/MLScientist/MLGym

for task in $EVAL_TASKS; do
    log "--- $SCIENTIST_LABEL on $task ---"
    for run in $(seq 1 $N_RUNS); do
        outdir="$OUTBASE/$task/run_$run"
        mkdir -p "$outdir"
        $PYTHON $LLMG \
            --task-config "tasks/${task}.yaml" \
            --node-budget $NODE_BUDGET --max-actions 15 \
            --scientist-model "$SCIENTIST_MODEL" --scientist-url http://localhost:8000/v1 \
            --executor-model "$BASE_MODEL" --executor-url http://localhost:8001/v1 \
            --temperature 0.9 \
            --output-dir "$outdir" > "$outdir/stdout.log" 2>&1 || true
        if [ -f "$outdir/result.json" ]; then
            score=$($PYTHON -c "import json; print(json.load(open('$outdir/result.json'))['best_score'])")
            log "  $task run $run: $score"
        else
            score="NaN"
            log "  $task run $run: FAILED"
        fi
        echo -e "$SCIENTIST_LABEL\t$task\t$run\t$score" >> "$RESULTS"
    done
done
log "Done"
