#!/bin/bash
# Run tree search experiments across tasks x modes x runs
# Tasks: titanic, houseprice, battleofsexes
# Modes: 2.0 (no VS), 2.1 (tail VS), 2.2 (uniform VS), 2.3 (local VS)
#
# Usage:
#   bash run_multitask_experiments.sh                    # all 3 tasks
#   bash run_multitask_experiments.sh houseprice         # single task
#   bash run_multitask_experiments.sh titanic houseprice # two tasks
#
# Results saved to outputs/tree_search/{task}/exp{X.Y}/run{I}/

set -e

cd /home/ubuntu/MLScientist/MLGym

PYTHON_CMD="uv run --project /home/ubuntu/MLScientist/air-agent python /home/ubuntu/MLScientist/air-agent/air/tree_search.py"
COMMON_ARGS="--branching-factor 3 --max-depth 2 --max-actions 15 --temperature 0.9 --verbose"
N_RUNS=5

# --- Ensure datasets exist ---
if [ ! -d "data/regressionKaggleHousePrice/data" ]; then
    echo "Generating houseprice dataset splits..."
    cd data/regressionKaggleHousePrice/_helpers
    uv run --project /home/ubuntu/MLScientist/air-agent python _generate_split.py
    cd /home/ubuntu/MLScientist/MLGym
    echo "Done."
fi

# --- Wait for vLLM ---
echo "Checking vLLM server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM is ready."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM not ready after 10 minutes. Aborting."
        exit 1
    fi
    echo "Waiting for vLLM... ($i/60)"
    sleep 10
done

# --- Define tasks ---
declare -A TASK_CONFIGS
TASK_CONFIGS[titanic]="tasks/titanic.yaml"
TASK_CONFIGS[houseprice]="tasks/regressionKaggleHousePrice.yaml"
TASK_CONFIGS[battleofsexes]="tasks/battleOfSexes.yaml"

ALL_TASKS="titanic houseprice battleofsexes"

# Filter tasks from CLI args (default: all)
if [ $# -gt 0 ]; then
    TASK_ORDER="$*"
else
    TASK_ORDER="$ALL_TASKS"
fi

# Count tasks
N_TASKS=$(echo $TASK_ORDER | wc -w)
TOTAL=$((N_TASKS * 4 * N_RUNS))
RUN_NUM=0
START_TIME=$(date +%s)

echo "============================================================"
echo "BATCH: $N_TASKS tasks x 4 modes x $N_RUNS runs = $TOTAL total"
echo "Tasks: $TASK_ORDER"
echo "Started: $(date)"
echo "============================================================"

for task in $TASK_ORDER; do
    TASK_CFG="${TASK_CONFIGS[$task]}"
    if [ -z "$TASK_CFG" ]; then
        echo "ERROR: Unknown task '$task'. Valid: titanic houseprice battleofsexes"
        continue
    fi

    for run in $(seq 1 $N_RUNS); do
        for exp in "2.0" "2.1" "2.2" "2.3"; do
            RUN_NUM=$((RUN_NUM + 1))
            OUTDIR="outputs/tree_search/${task}/exp${exp}/run${run}"

            # Skip if already completed
            if [ -f "$OUTDIR/result.json" ]; then
                BEST=$(python3 -c "import json; d=json.load(open('$OUTDIR/result.json')); print(f\"best={d['best_score']:.4f}\")")
                echo "[$RUN_NUM/$TOTAL] $task exp$exp run$run: SKIPPING (already done, $BEST)"
                continue
            fi

            case $exp in
                "2.0") EXP_ARGS="--no-verbalized-sampling" ;;
                "2.1") EXP_ARGS="--sampling-mode tail" ;;
                "2.2") EXP_ARGS="--sampling-mode uniform" ;;
                "2.3") EXP_ARGS="--sampling-mode local" ;;
            esac

            echo ""
            echo "============================================================"
            echo "[$RUN_NUM/$TOTAL] Task=$task, Exp $exp, Run $run"
            echo "Config: $TASK_CFG"
            echo "Output: $OUTDIR"
            echo "Time: $(date)"
            echo "============================================================"

            $PYTHON_CMD $COMMON_ARGS $EXP_ARGS \
                --task-config "$TASK_CFG" \
                --output-dir "$OUTDIR" 2>&1 || {
                echo "WARNING: $task exp $exp run $run failed, continuing..."
            }

            if [ -f "$OUTDIR/result.json" ]; then
                BEST=$(python3 -c "import json; d=json.load(open('$OUTDIR/result.json')); print(f\"best={d['best_score']:.4f} ({d['best_node_id']}), nodes={d['total_nodes']}, time={d['elapsed_seconds']:.0f}s\")")
                echo ">>> RESULT: $task exp$exp run$run: $BEST"
            else
                echo ">>> RESULT: $task exp$exp run$run: NO RESULT FILE"
            fi
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE"
echo "Total time: ${ELAPSED} minutes"
echo "Finished: $(date)"
echo "============================================================"

# --- Summary tables ---
python3 << 'PYEOF'
import json
from pathlib import Path

tasks = ["titanic", "houseprice", "battleofsexes"]
exps = ["2.0", "2.1", "2.2", "2.3"]
n_runs = 5
base = Path("outputs/tree_search")

for task in tasks:
    task_dir = base / task
    if not task_dir.exists():
        continue
    print(f"\n{'='*70}")
    print(f"TASK: {task}")
    print(f"{'='*70}")
    print(f"{'Exp':<8}", end="")
    for r in range(1, n_runs+1):
        print(f"{'Run'+str(r):<10}", end="")
    print(f"{'Mean':<10}{'Max':<10}{'Min':<10}")
    print("-" * 70)

    for exp in exps:
        scores = []
        print(f"{exp:<8}", end="")
        for r in range(1, n_runs+1):
            rfile = base / task / f"exp{exp}" / f"run{r}" / "result.json"
            if rfile.exists():
                d = json.loads(rfile.read_text())
                s = d["best_score"]
                scores.append(s)
                print(f"{s:<10.4f}", end="")
            else:
                print(f"{'---':<10}", end="")

        if scores:
            valid = [s for s in scores if s > 0]
            if valid:
                print(f"{sum(valid)/len(valid):<10.4f}{max(valid):<10.4f}{min(valid):<10.4f}", end="")
        print()

# Cross-task summary
available = [t for t in tasks if (base / t).exists()]
if len(available) > 1:
    print(f"\n{'='*70}")
    print("CROSS-TASK SUMMARY (best score per experiment, averaged over runs)")
    print(f"{'='*70}")
    print(f"{'Exp':<8}", end="")
    for task in available:
        print(f"{task:<18}", end="")
    print()
    print("-" * (8 + 18 * len(available)))
    for exp in exps:
        print(f"{exp:<8}", end="")
        for task in available:
            scores = []
            for r in range(1, n_runs+1):
                rfile = base / task / f"exp{exp}" / f"run{r}" / "result.json"
                if rfile.exists():
                    d = json.loads(rfile.read_text())
                    if d["best_score"] > 0:
                        scores.append(d["best_score"])
            if scores:
                print(f"{sum(scores)/len(scores):<18.4f}", end="")
            else:
                print(f"{'N/A':<18}", end="")
        print()
PYEOF
