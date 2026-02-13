#!/bin/bash
# Experiment 3: Adaptive tree search ablation batch runner.
# Runs each signal configuration on titanic, 1 run each (directional signal).
# Use from MLGym directory:
#   bash /home/ubuntu/MLScientist/air-agent/air/run_exp3_batch.sh

set -e

PROJECT="/home/ubuntu/MLScientist/air-agent"
SCRIPT="$PROJECT/air/adaptive_tree_search.py"
TASK="tasks/titanic.yaml"
BASE_OUT="outputs/adaptive_search/titanic"
BUDGET=12
BREADTH=3

# Check vLLM
echo "Checking vLLM server..."
curl -s http://localhost:8000/v1/models > /dev/null || { echo "vLLM not running!"; exit 1; }
echo "vLLM is ready."

# Common args
COMMON="--node-budget $BUDGET --initial-breadth $BREADTH --max-actions 15 --temperature 0.9 --verbose --task-config $TASK"

# Define experiments: name | extra flags
declare -a EXPS=(
    "3.0.p|--context parent"
    "3.0.g|--context global"
    "3.1a.p|--use-variance --context parent"
    "3.1a.g|--use-variance --context global"
    "3.1b.p|--use-regret --context parent"
    "3.1b.g|--use-regret --context global"
    "3.1d.p|--use-coverage --context parent"
    "3.1d.g|--use-coverage --context global"
    "3.1e.p|--use-depth --context parent"
    "3.1e.g|--use-depth --context global"
    "3.2.p|--use-regret --use-depth --context parent"
    "3.2.g|--use-regret --use-depth --context global"
    "3.3.p|--use-variance --use-regret --use-coverage --use-depth --context parent"
    "3.3.g|--use-variance --use-regret --use-coverage --use-depth --context global"
)

TOTAL=${#EXPS[@]}
echo "============================================================"
echo "BATCH: $TOTAL experiments on titanic"
echo "Started: $(date)"
echo "============================================================"

for i in "${!EXPS[@]}"; do
    IFS='|' read -r NAME FLAGS <<< "${EXPS[$i]}"
    OUTDIR="$BASE_OUT/$NAME/run1"

    # Skip if already done
    if [ -f "$OUTDIR/result.json" ]; then
        echo "[$(($i+1))/$TOTAL] $NAME — SKIPPING (already done)"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[$(($i+1))/$TOTAL] $NAME"
    echo "Flags: $FLAGS"
    echo "Output: $OUTDIR"
    echo "Time: $(date)"
    echo "============================================================"

    uv run --project "$PROJECT" python "$SCRIPT" \
        $COMMON $FLAGS \
        --output-dir "$OUTDIR" \
        2>&1

    echo "[$(($i+1))/$TOTAL] $NAME — DONE"
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"

# Summary table
echo ""
echo "RESULTS SUMMARY"
echo "============================================================"
printf "%-12s %10s %8s %6s\n" "Experiment" "Best" "Nodes" "Time"
echo "--------------------------------------------"
for i in "${!EXPS[@]}"; do
    IFS='|' read -r NAME FLAGS <<< "${EXPS[$i]}"
    RESULT="$BASE_OUT/$NAME/run1/result.json"
    if [ -f "$RESULT" ]; then
        BEST=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"best_score\"]:.4f}')")
        NODES=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d['total_nodes'])")
        TIME=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"elapsed_seconds\"]:.0f}s')")
        printf "%-12s %10s %8s %6s\n" "$NAME" "$BEST" "$NODES" "$TIME"
    else
        printf "%-12s %10s\n" "$NAME" "FAILED"
    fi
done
