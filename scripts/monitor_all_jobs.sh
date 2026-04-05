#!/bin/bash
# Combined monitor for v8/v9 linear retries (4616793/4616794)
# and v10 tree training (4617476).
TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "========== SNAPSHOT $TS =========="
squeue -j 4616793,4616794,4617476 --format='%.10i %.12j %.2t %.10M %.9L %R' 2>&1
echo ""

for J in 4616793:v8:linear 4616794:v9:linear 4617476:v10:tree; do
  JID=$(echo $J | cut -d: -f1)
  TAG=$(echo $J | cut -d: -f2)
  KIND=$(echo $J | cut -d: -f3)
  LOGF=$(ls /scratch/jarnav/logs/rl_${TAG}_*_${JID}.log 2>/dev/null | head -1)
  [ -z "$LOGF" ] && LOGF=$(ls /scratch/jarnav/logs/rl_${TAG}_*${JID}.log 2>/dev/null | head -1)
  echo "=== ${TAG} (${KIND}) job=${JID} ==="
  if [ -z "$LOGF" ] || [ ! -f "$LOGF" ]; then
    echo "  (no log file yet)"
    continue
  fi
  # Latest step
  STEP=$(grep -oE 'Step [0-9]+ \| Time: [0-9.]+s' "$LOGF" 2>/dev/null | tail -1)
  echo "  latest: ${STEP:-(before step 1)}"
  # Errors of interest
  NAMERR=$(grep -c "has no attribute '__name__'" "$LOGF" 2>/dev/null || echo 0)
  FAILERR=$(grep -c "Rollout failed" "$LOGF" 2>/dev/null || echo 0)
  echo "  __name__ errors: $NAMERR | rollout_failed warnings: $FAILERR"
  # Last 2 non-empty lines
  tail -40 "$LOGF" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -2 | sed 's/^/  TAIL: /'
done

echo ""
echo "=== Reward logs ==="
for F in /scratch/jarnav/rollout_logs/rewards_v7_fixed_tier.jsonl \
         /scratch/jarnav/rollout_logs/rewards_v8_global_best.jsonl \
         /scratch/jarnav/rollout_logs/rewards_v9_percentile.jsonl \
         /scratch/jarnav/rollout_logs/rewards_v6_binary.jsonl; do
  if [ -f "$F" ]; then
    N=$(wc -l < "$F")
    printf "  %-60s %6d\n" "$(basename $F)" "$N"
  fi
done

echo ""
echo "=== Tree rollout logs (v10) ==="
for F in /scratch/jarnav/rollout_logs/fashionMnist_v6_binary_tree_rollouts.jsonl \
         /scratch/jarnav/rollout_logs/fashionMnist_v7_fixed_tier_tree_rollouts.jsonl; do
  if [ -f "$F" ]; then
    N=$(wc -l < "$F")
    # Check tree depth distribution
    python3 -c "
import json
depths = []
parents = {'root': 0}
for l in open('$F'):
    try:
        d = json.loads(l)
        depths.append(d.get('new_node_depth', 0))
        p = d.get('parent_id', 'root')
        parents[p] = parents.get(p, 0) + 1
    except: pass
if depths:
    from collections import Counter
    dc = Counter(depths)
    print(f'  $(basename $F): {len(depths)} rollouts')
    print(f'    depth distribution: {dict(sorted(dc.items()))}')
    # Root vs non-root parent ratio
    root_ct = parents.get('root', 0)
    print(f'    parent selection: root={root_ct}  non-root={sum(v for k,v in parents.items() if k!=\"root\")}')
" 2>/dev/null
  fi
done
echo ""
