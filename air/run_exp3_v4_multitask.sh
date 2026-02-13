#!/bin/bash
# Experiment 3 v4: UCB and Open-Ended — Multi-task evaluation
# Runs the full v4 config set on houseprice and battleofsexes
#
# Run from MLGym directory:
#   cd /home/ubuntu/MLScientist/MLGym
#   bash /home/ubuntu/MLScientist/air-agent/air/run_exp3_v4_multitask.sh

set -e

SCRIPT="/home/ubuntu/MLScientist/air-agent/air/adaptive_tree_search.py"
BASE_OUT="outputs/adaptive_search_v3"

run_experiment() {
    local task_key="$1"
    local task_config="$2"
    local name="$3"
    shift 3
    local out_dir="${BASE_OUT}/${task_key}/${name}/run1"

    if [ -f "${out_dir}/result.json" ]; then
        echo "=== SKIP ${task_key}/${name} (already completed) ==="
        return
    fi

    echo ""
    echo "============================================================"
    echo "  RUNNING: ${task_key}/${name}"
    echo "  Output:  ${out_dir}"
    echo "============================================================"
    echo ""

    uv run --project /home/ubuntu/MLScientist/air-agent \
        python "$SCRIPT" \
        --task-config "$task_config" \
        --output-dir "$out_dir" \
        --node-budget 12 \
        --initial-breadth 3 \
        --max-actions 15 \
        "$@"

    echo ""
    echo "=== DONE: ${task_key}/${name} ==="
    echo ""
}

run_all_configs() {
    local task_key="$1"
    local task_config="$2"

    echo ""
    echo "########################################################"
    echo "  TASK: ${task_key}"
    echo "  Config: ${task_config}"
    echo "########################################################"
    echo ""

    # --- UCB ---
    run_experiment "$task_key" "$task_config" "ucb_c1.0.p" --selection-strategy ucb --ucb-c 1.0 --context parent
    run_experiment "$task_key" "$task_config" "ucb_c1.41.p" --selection-strategy ucb --ucb-c 1.41 --context parent
    run_experiment "$task_key" "$task_config" "ucb_c2.0.p" --selection-strategy ucb --ucb-c 2.0 --context parent
    run_experiment "$task_key" "$task_config" "ucb_c1.41.g" --selection-strategy ucb --ucb-c 1.41 --context global
    run_experiment "$task_key" "$task_config" "ucb_c2.0.g" --selection-strategy ucb --ucb-c 2.0 --context global

    # --- Open-ended ---
    run_experiment "$task_key" "$task_config" "oe_t0.3_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.3 --commitment-threshold 2 --context parent
    run_experiment "$task_key" "$task_config" "oe_t0.5_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 2 --context parent
    run_experiment "$task_key" "$task_config" "oe_t0.5_k3.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 3 --context parent
    run_experiment "$task_key" "$task_config" "oe_t1.0_k2.p" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 1.0 --commitment-threshold 2 --context parent
    run_experiment "$task_key" "$task_config" "oe_t0.5_k2.g" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 0.5 --commitment-threshold 2 --context global
    run_experiment "$task_key" "$task_config" "oe_t1.0_k2.g" --selection-strategy open-ended --ucb-c 1.41 --trend-weight 1.0 --commitment-threshold 2 --context global

    # --- Baselines ---
    run_experiment "$task_key" "$task_config" "regret.p" --selection-strategy signals --use-regret --context parent
    run_experiment "$task_key" "$task_config" "regret.g" --selection-strategy signals --use-regret --context global
    run_experiment "$task_key" "$task_config" "regret_depth.g" --selection-strategy signals --use-regret --use-depth --context global
}

# --- Run on both remaining tasks ---
run_all_configs "houseprice" "configs/tasks/regressionKaggleHousePrice.yaml"
run_all_configs "battleofsexes" "configs/tasks/battleOfSexes.yaml"

echo ""
echo "============================================================"
echo "  ALL MULTI-TASK EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""

# Print summary for all tasks
for task in houseprice battleofsexes; do
    echo "=== ${task} ==="
    for dir in ${BASE_OUT}/${task}/*/run1; do
        exp=$(echo "$dir" | sed "s|${BASE_OUT}/${task}/||;s|/run1||")
        if [ -f "${dir}/result.json" ]; then
            score=$(python3 -c "import json; d=json.load(open('${dir}/result.json')); print(f\"{d['best_score']:.4f}\")")
            echo "  ${exp}: ${score}"
        fi
    done
    echo ""
done
