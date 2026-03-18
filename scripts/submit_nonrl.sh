#!/bin/bash
# Submit all non-RL tasks across all 5 methods.
# Methods: aira_vanilla (qwen MCTS x5), aira_o3 (o3 MCTS x2),
#          llm_guided_qwen, llm_guided_o3, llm_guided_o3o3 (each x2)
# Budgets: n5 and n15 for all methods.
#
# Usage: bash submit_nonrl.sh [--missing-only]
#   --missing-only: skip runs that already have result.json

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTBASE=/home/jarnav/MLScientist/air-agent/outputs/Feb28_Experiments
MISSING_ONLY=0
[[ "$1" == "--missing-only" ]] && MISSING_ONLY=1

TASKS="titanic battleOfSexes blotto prisonersDilemma regressionKaggleHousePrice"

_sbatch_llmg() {
    local TASK=$1 N=$2 RUN=$3 SCIENTIST=$4 TIME=$5
    local OUTDIR="$OUTBASE/llm_guided/${SCIENTIST}_scientist/${TASK}/llm_guided_n${N}_${RUN}"
    if [ "$MISSING_ONLY" = "1" ] && [ -f "$OUTDIR/result.json" ]; then
        echo "  SKIP: llm_guided/${SCIENTIST}/${TASK}/n${N}_${RUN}"
        return
    fi
    sbatch --job-name="llmg_${SCIENTIST:0:3}_${TASK:0:4}_n${N}" \
           --time="$TIME" \
           --export=ALL,TASK="$TASK",NODE_BUDGET="$N",RUN_ID="$RUN",SCIENTIST="$SCIENTIST" \
           "$SCRIPTS_DIR/template_llm_guided.sh"
    echo "  Submitted: llm_guided/${SCIENTIST}/${TASK}/n${N}_${RUN}"
}

_sbatch_aira() {
    local TASK=$1 N=$2 RUN=$3 MODEL=$4 TIME=$5 GPU_FLAGS=$6
    local MDIR="aira_o3"; [ "$MODEL" = "qwen" ] && MDIR="aira_vanilla"
    local OUTDIR="$OUTBASE/${MDIR}/${TASK}/aira_mcts_n${N}_${RUN}"
    if [ "$MISSING_ONLY" = "1" ] && [ -f "$OUTDIR/result.json" ]; then
        echo "  SKIP: ${MDIR}/${TASK}/n${N}_${RUN}"
        return
    fi
    sbatch --job-name="aira_${MODEL:0:3}_${TASK:0:4}_n${N}" \
           --time="$TIME" $GPU_FLAGS \
           --export=ALL,TASK="$TASK",NODE_BUDGET="$N",RUN_ID="$RUN",MODEL="$MODEL" \
           "$SCRIPTS_DIR/template_aira_mcts.sh"
    echo "  Submitted: ${MDIR}/${TASK}/n${N}_${RUN}"
}

echo "=== aira_vanilla (Qwen MCTS, 5 runs per task/budget) ==="
for TASK in $TASKS; do
    for N in 5 15; do
        TIME="3:00:00"; [ "$N" -eq 15 ] && TIME="5:00:00"
        for R in r1 r2 r3 r4 r5; do
            _sbatch_aira "$TASK" "$N" "$R" "qwen" "$TIME" \
                "--gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1 --mem=64G --cpus-per-task=8"
        done
    done
done

echo ""
echo "=== aira_o3 (o3 MCTS, 2 runs per task/budget) ==="
for TASK in $TASKS; do
    for N in 5 15; do
        TIME="3:00:00"; [ "$N" -eq 15 ] && TIME="5:00:00"
        for R in r1 r2; do
            _sbatch_aira "$TASK" "$N" "$R" "o3" "$TIME" "--cpus-per-task=4 --mem=32G"
        done
    done
done

echo ""
echo "=== llm_guided_o3 (o3 scientist + Qwen executor) ==="
for TASK in $TASKS; do
    for N in 5 15; do
        TIME="3:00:00"; [ "$N" -eq 15 ] && TIME="6:00:00"
        for R in r1 r2; do
            _sbatch_llmg "$TASK" "$N" "$R" "o3" "$TIME"
        done
    done
done

echo ""
echo "=== llm_guided_o3o3 (o3 scientist + o3 executor) ==="
for TASK in $TASKS; do
    for N in 5 15; do
        TIME="3:00:00"; [ "$N" -eq 15 ] && TIME="6:00:00"
        for R in r1 r2; do
            _sbatch_llmg "$TASK" "$N" "$R" "o3o3" "$TIME"
        done
    done
done

echo ""
echo "=== llm_guided_qwen (Qwen scientist + Qwen executor) ==="
for TASK in $TASKS; do
    for N in 5 15; do
        TIME="3:00:00"; [ "$N" -eq 15 ] && TIME="6:00:00"
        for R in r1 r2; do
            _sbatch_llmg "$TASK" "$N" "$R" "qwen" "$TIME"
        done
    done
done

echo ""
echo "Total jobs: $(( 5 * 2 * 5 + 5 * 2 * 2 + 5 * 2 * 2 * 3 )) = ~130"
echo "Use --missing-only to skip already-completed runs."
