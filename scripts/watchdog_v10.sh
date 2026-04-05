#!/bin/bash
# Watchdog for v10 tree job. Runs every 10 minutes, checks for:
#  - Job exited (rescue: investigate, maybe re-submit)
#  - No tree rollout log entries in last 30 minutes (possible hang)
#  - __name__ attribute errors (shouldn't happen with the fix, but check)
#  - Rollout failed warnings (same)
#  - Tree depth stuck at 1 (parent parsing broken)
#
# Writes a status summary to outputs/v10_watchdog.log

JOBID=${1:-4617476}
LOGF="/home/jarnav/MLScientist/air-agent/outputs/v10_watchdog.log"
TS=$(date '+%Y-%m-%d %H:%M:%S')

{
  echo "===== $TS ====="

  # SLURM state
  STATE=$(squeue -j $JOBID --noheader --format='%T' 2>/dev/null | tr -d ' ')
  if [ -z "$STATE" ]; then
    # Not in queue — finished
    FINAL=$(sacct -j $JOBID --noheader --format='State,ExitCode,Elapsed' 2>/dev/null | head -1)
    echo "JOB NOT RUNNING — sacct: $FINAL"
  else
    echo "SLURM state: $STATE"
  fi

  # Log file tail
  LOG=$(ls /scratch/jarnav/logs/rl_v10_*_${JOBID}.log 2>/dev/null | head -1)
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    STEP=$(grep -oE 'Step [0-9]+ \| Time' "$LOG" 2>/dev/null | tail -1)
    echo "latest step: ${STEP:-(before step 1)}"
    NAMERR=$(grep -c "has no attribute '__name__'" "$LOG" 2>/dev/null)
    FAILERR=$(grep -c "Rollout failed" "$LOG" 2>/dev/null)
    echo "__name__ errors: ${NAMERR:-0}  |  Rollout failed: ${FAILERR:-0}"
    # Any Python traceback in last 200 lines
    TRACE=$(tail -200 "$LOG" 2>/dev/null | grep -c "Traceback")
    echo "tracebacks in last 200 lines: ${TRACE:-0}"
  fi

  # Tree rollout log stats
  TF="/scratch/jarnav/rollout_logs/fashionMnist_v6_binary_tree_rollouts.jsonl"
  if [ -f "$TF" ]; then
    N=$(wc -l < "$TF")
    LATEST=$(tail -1 "$TF" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d.get('timestamp','?'))" 2>/dev/null)
    echo "tree rollouts: $N (latest: $LATEST)"
    # Depth distribution
    python3 -c "
import json
from collections import Counter
dc = Counter()
parents = Counter()
for l in open('$TF'):
    try:
        d = json.loads(l)
        dc[d.get('new_node_depth', 1)] += 1
        parents[d.get('parent_id','root')] += 1
    except: pass
print(f'  depth dist: {dict(sorted(dc.items()))}')
root_ct = parents.get('root', 0)
non_root = sum(v for k,v in parents.items() if k != 'root')
print(f'  parent: root={root_ct}  non-root={non_root}')
if len(dc) == 1 and 1 in dc and dc[1] > 8:
    print(f'  WARNING: all nodes at depth 1 — parent parsing may be broken')
" 2>/dev/null
  fi

  echo ""
} >> "$LOGF" 2>&1
