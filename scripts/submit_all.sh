#!/bin/bash
# Submit all Feb28_Experiments using the two template scripts.
# Usage: bash submit_all.sh [--missing-only]
#
# --missing-only: only submit runs that don't have a result.json yet

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTBASE=/home/jarnav/MLScientist/air-agent/outputs/Feb28_Experiments
MISSING_ONLY=0
[[ "$1" == "--missing-only" ]] && MISSING_ONLY=1

_sbatch_llmg() {
    local TASK=$1 NODE_BUDGET=$2 RUN_ID=$3 SCIENTIST=$4 TIME=$5
    local OUTDIR="$OUTBASE/llm_guided/${SCIENTIST}_scientist/${TASK}/llm_guided_n${NODE_BUDGET}_${RUN_ID}"
    if [ "$MISSING_ONLY" = "1" ] && [ -f "$OUTDIR/result.json" ]; then
        echo "  SKIP (done): llm_guided/${SCIENTIST}/${TASK}/n${NODE_BUDGET}_${RUN_ID}"
        return
    fi
    local JNAME="llmg_${SCIENTIST:0:3}_${TASK:0:4}_n${NODE_BUDGET}"
    sbatch --job-name="$JNAME" --time="$TIME" \
        --export=ALL,TASK="$TASK",NODE_BUDGET="$NODE_BUDGET",RUN_ID="$RUN_ID",SCIENTIST="$SCIENTIST" \
        "$SCRIPTS_DIR/template_llm_guided.sh"
    echo "  Submitted: llm_guided/${SCIENTIST}/${TASK}/n${NODE_BUDGET}_${RUN_ID}"
}

_sbatch_aira() {
    local TASK=$1 NODE_BUDGET=$2 RUN_ID=$3 MODEL=$4 TIME=$5 GPU_FLAGS=$6
    local MDIR="aira_o3"; [ "$MODEL" = "qwen" ] && MDIR="aira_vanilla"
    local OUTDIR="$OUTBASE/${MDIR}/${TASK}/aira_mcts_n${NODE_BUDGET}_${RUN_ID}"
    if [ "$MISSING_ONLY" = "1" ] && [ -f "$OUTDIR/result.json" ]; then
        echo "  SKIP (done): ${MDIR}/${TASK}/n${NODE_BUDGET}_${RUN_ID}"
        return
    fi
    local JNAME="aira_${MODEL:0:3}_${TASK:0:4}_n${NODE_BUDGET}"
    sbatch --job-name="$JNAME" --time="$TIME" $GPU_FLAGS \
        --export=ALL,TASK="$TASK",NODE_BUDGET="$NODE_BUDGET",RUN_ID="$RUN_ID",MODEL="$MODEL" \
        "$SCRIPTS_DIR/template_aira_mcts.sh"
    echo "  Submitted: ${MDIR}/${TASK}/n${NODE_BUDGET}_${RUN_ID}"
}

TASKS_ML="titanic battleOfSexes blotto prisonersDilemma regressionKaggleHousePrice"
TASKS_RL="rlMountainCarContinuous rlMountainCarContinuousReinforce rlBreakoutMinAtar rlMetaMaze"
ALL_TASKS="$TASKS_ML $TASKS_RL"

echo "=== Submitting aira_vanilla (Qwen MCTS) ==="
for TASK in $ALL_TASKS; do
    for N in 5 15; do
        for R in r1 r2 r3 r4 r5; do
            T="3:00:00"; [ "$N" -eq 15 ] && T="5:00:00"
            _sbatch_aira "$TASK" "$N" "$R" "qwen" "$T" "--gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:2"
        done
    done
done

echo ""
echo "=== Submitting aira_o3 (o3 MCTS, no GPU) ==="
for TASK in $ALL_TASKS; do
    for N in 5 15; do
        for R in r1 r2; do
            T="3:00:00"; [ "$N" -eq 15 ] && T="5:00:00"
            _sbatch_aira "$TASK" "$N" "$R" "o3" "$T" ""
        done
    done
done

echo ""
echo "=== Submitting llm_guided_o3 (o3 scientist, Qwen executor) ==="
for TASK in $ALL_TASKS; do
    for N in 5 15; do
        T="3:00:00"; [ "$N" -eq 15 ] && T="6:00:00"
        _sbatch_llmg "$TASK" "$N" "r1" "o3" "$T"
    done
done

echo ""
echo "=== Submitting llm_guided_o3o3 (o3 scientist + o3 executor) ==="
for TASK in $ALL_TASKS; do
    for N in 5 15; do
        T="3:00:00"; [ "$N" -eq 15 ] && T="6:00:00"
        _sbatch_llmg "$TASK" "$N" "r1" "o3o3" "$T"
    done
done

echo ""
echo "=== Submitting llm_guided_qwen (Qwen scientist + Qwen executor) ==="
for TASK in $ALL_TASKS; do
    for N in 5 15; do
        T="3:00:00"; [ "$N" -eq 15 ] && T="6:00:00"
        _sbatch_llmg "$TASK" "$N" "r1" "qwen" "$T"
    done
done

echo ""
echo "Done submitting."
