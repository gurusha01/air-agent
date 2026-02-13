#!/bin/bash
# Experiment 3 v3: UCB and Open-Ended selection strategies
# Compares vanilla UCB vs open-ended (UCB + trend + commitment) vs best signal-based (regret)
#
# Run from MLGym directory:
#   cd /home/ubuntu/MLScientist/MLGym
#   bash /home/ubuntu/MLScientist/air-agent/air/run_exp3_v3.sh

set -e

SCRIPT="/home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py"
BASE_OUT="outputs/adaptive_search_v3/titanic"
TASK="tasks/titanic.yaml"

run_experiment() {
    local name="$1"
    shift
    local out_dir="${BASE_OUT}/${name}/run1"

    if [ -f "${out_dir}/result.json" ]; then
        echo "=== SKIP ${name} (already completed) ==="
        return
    fi

    echo ""
    echo "============================================================"
    echo "  RUNNING: ${name}"
    echo "  Output:  ${out_dir}"
    echo "============================================================"
    echo ""

    uv run --project /home/ubuntu/MLScientist/air-agent \
        python "$SCRIPT" \
        --task-config "$TASK" \
        --output-dir "$out_dir" \
        --node-budget 12 \
        --initial-breadth 3 \
        --max-actions 15 \
        "$@"

    echo ""
    echo "=== DONE: ${name} ==="
    echo ""
}

# 1. UCB with parent context, varying C
run_experiment "ucb_c1.0.p" --selection-strategy ucb --ucb-c 1.0 --context parent
run_experiment "ucb_c1.41.p" --selection-strategy ucb --ucb-c 1.41 --context parent
run_experiment "ucb_c2.0.p" --selection-strategy ucb --ucb-c 2.0 --context parent

# 2. UCB with global context (best C from parent, plus standard)
run_experiment "ucb_c1.41.g" --selection-strategy ucb --ucb-c 1.41 --context global
run_experiment "ucb_c2.0.g" --selection-strategy ucb --ucb-c 2.0 --context global

# 3. Open-ended with parent context (sweep trend_weight and commitment)
run_experiment "oe_t0.3_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.3 --commitment-threshold 2 --context parent
run_experiment "oe_t0.5_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 2 --context parent
run_experiment "oe_t0.5_k3.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 3 --context parent
run_experiment "oe_t1.0_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 1.0 --commitment-threshold 2 --context parent

# 4. Open-ended with global context
run_experiment "oe_t0.5_k2.g" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 2 --context global
run_experiment "oe_t1.0_k2.g" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 1.0 --commitment-threshold 2 --context global

# 5. Baselines for comparison (re-run regret with same output dir convention)
run_experiment "regret.p" --selection-strategy signals --use-regret --context parent
run_experiment "regret.g" --selection-strategy signals --use-regret --context global
run_experiment "regret_depth.g" --selection-strategy signals --use-regret --use-depth --context global

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""

# Print summary
echo "Results:"
for dir in ${BASE_OUT}/*/run1; do
    exp=$(echo "$dir" | sed "s|${BASE_OUT}/||;s|/run1||")
    if [ -f "${dir}/result.json" ]; then
        score=$(python3 -c "import json; d=json.load(open('${dir}/result.json')); print(f\"{d['best_score']:.4f}\")")
        echo "  ${exp}: ${score}"
    fi
done
