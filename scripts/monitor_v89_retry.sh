#!/bin/bash
# Monitor the retry runs of v8 (4616793) and v9 (4616794).
TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "========== SNAPSHOT $TS =========="
squeue -j 4616793,4616794 --format='%.10i %.12j %.2t %.10M %.8L %R' 2>&1
echo ""
for J in 4616793:v8 4616794:v9; do
  JID=${J%:*}; TAG=${J#*:}
  LOGF=$(ls /scratch/jarnav/logs/rl_${TAG}_fmnist_${JID}.log 2>/dev/null)
  echo "=== ${TAG} (job $JID) ==="
  if [ -n "$LOGF" ]; then
    tail -n 4 "$LOGF" 2>/dev/null | sed 's/^/  /'
    # Check for __name__ errors specifically
    ERR=$(grep -c "has no attribute '__name__'" "$LOGF" 2>/dev/null || echo 0)
    echo "  __name__ errors: $ERR"
    # Step progress
    STEP=$(grep -oE 'Step [0-9]+ \| Time' "$LOGF" 2>/dev/null | tail -1)
    echo "  latest step: ${STEP:-(none yet)}"
  else
    echo "  no log yet"
  fi

  # Check output dir rollouts/
  OUT=/scratch/jarnav/rl_${TAG}_fmnist_retry
  if [ -d "$OUT/rollouts" ]; then
    NSTEPS=$(ls "$OUT/rollouts" 2>/dev/null | wc -l)
    echo "  rollout step dirs: $NSTEPS"
  fi
  # Check the retry orchestrator log for errors
  ORCH=$OUT/run_default/logs/orchestrator.log
  if [ -f "$ORCH" ]; then
    FAILS=$(grep -c "Rollout failed" "$ORCH" 2>/dev/null || echo 0)
    echo "  orchestrator rollout failures: $FAILS"
  fi
done

echo ""
for SCHEME in v8_global_best v9_percentile; do
  F=/scratch/jarnav/rollout_logs/rewards_${SCHEME}.jsonl
  if [ -f "$F" ]; then
    N=$(wc -l < "$F")
    echo "  rewards_${SCHEME}: $N"
    tail -1 "$F" | python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read())
    extra=''
    if 'best_ever_snapshot' in d: extra=f\" best_ever={d['best_ever_snapshot']}\"
    if 'p_threshold_snapshot' in d: extra=f\" p={d['p_threshold_snapshot']}\"
    print(f'    last: r={d[\"reward\"]:+.2f} s={d[\"best_score\"]}{extra}')
except: pass
" 2>&1
  fi
done
echo ""
