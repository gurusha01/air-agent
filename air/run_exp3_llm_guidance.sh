#!/bin/bash
# Experiment 3: LLM guidance ablation (add-on to main batch).
# Runs LLM guidance signal on titanic with parent and global context.
# Use from MLGym directory:
#   bash /home/ubuntu/MLScientist/air-agent/air/run_exp3_llm_guidance.sh

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

COMMON="--node-budget $BUDGET --initial-breadth $BREADTH --max-actions 15 --temperature 0.9 --verbose --task-config $TASK"

declare -a EXPS=(
    "3.1c.p|--use-llm-guidance --context parent"
    "3.1c.g|--use-llm-guidance --context global"
)

TOTAL=${#EXPS[@]}
echo "============================================================"
echo "BATCH: LLM Guidance experiments on titanic"
echo "Started: $(date)"
echo "============================================================"

for i in "${!EXPS[@]}"; do
    IFS='|' read -r NAME FLAGS <<< "${EXPS[$i]}"
    OUTDIR="$BASE_OUT/$NAME/run1"

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
echo "LLM GUIDANCE EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"

# Summary
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
