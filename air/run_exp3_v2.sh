#!/bin/bash
# Experiment 3 v2: Re-run experiments affected by the root-selection bug fix
# and new LLM guidance prompt (interestingness + depth potential).
#
# Bug fix: root was always selected by regret+depth because depth=0 gets max
# bonus and score=0.0 gets max regret. Now root is excluded after initial_breadth
# children are created.
#
# LLM guidance: now rates nodes on "interestingness" (explore) vs "depth potential"
# (exploit) instead of generic "exploration potential". The LLM's recommendation
# also drives the explore/exploit decision (replaces percentile threshold).

set -e

PROJECT="/home/ubuntu/MLScientist/air-agent"
SCRIPT="$PROJECT/air/adaptive_tree_search.py"
TASK="tasks/titanic.yaml"
BASE_OUT="outputs/adaptive_search_v2/titanic"
BUDGET=12
BREADTH=3

echo "Checking vLLM server..."
curl -s http://localhost:8000/v1/models > /dev/null || { echo "vLLM not running!"; exit 1; }
echo "vLLM is ready."

COMMON="--node-budget $BUDGET --initial-breadth $BREADTH --max-actions 15 --temperature 0.9 --verbose --task-config $TASK"

# Re-run ALL experiments affected by the root-selection bug fix.
# The fix excludes root from candidates once it has >= initial_breadth children.
# This matters for any signal that systematically favored root:
#   - regret: root score=0.0 → regret=global_best (always max)
#   - depth: root depth=0 → bonus=1.0 (always max)
# Variance/coverage are also slightly affected since root was always a candidate.
declare -a EXPS=(
    # Regret only — was deep in v1 (root regret drops as children improve),
    # but root exclusion still changes behavior slightly
    "3.1bv2.p|--use-regret --context parent"
    "3.1bv2.g|--use-regret --context global"
    # LLM guidance — new interestingness/depth-potential prompt + LLM-driven explore/exploit
    "3.1cv2.p|--use-llm-guidance --context parent"
    "3.1cv2.g|--use-llm-guidance --context global"
    # Depth only — was flat (depth 1) in v1, root always won
    "3.1ev2.p|--use-depth --context parent"
    "3.1ev2.g|--use-depth --context global"
    # Regret+depth — most affected (root always selected → flat GBM monoculture)
    "3.2v2.p|--use-regret --use-depth --context parent"
    "3.2v2.g|--use-regret --use-depth --context global"
    # All cheap signals — has regret+depth so affected
    "3.3v2.p|--use-variance --use-regret --use-coverage --use-depth --context parent"
    "3.3v2.g|--use-variance --use-regret --use-coverage --use-depth --context global"
)

TOTAL=${#EXPS[@]}
echo "============================================================"
echo "BATCH v2: $TOTAL experiments on titanic (with bug fixes)"
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
echo "ALL v2 EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"

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
