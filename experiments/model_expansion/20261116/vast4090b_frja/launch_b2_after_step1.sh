#!/usr/bin/env bash
# Waits for step 1 (InternVL3.5-8B REL new_only) to release /workspace/.queue_b2.lock, requires its verdict OK and passed gate,
# then starts queue_b2_frja.sh. Name deliberately avoids "queue_" so the full queue's OTHERS check does not match it.
cd /workspace || exit 1
echo "$(date -u +%FT%TZ) launcher waiting for step-1 lock" >> /workspace/launch_b2.log
exec 8> /workspace/.queue_b2.lock; flock 8; flock -u 8; exec 8>&-
sleep 60
v=$(cat QUEUE_B2_DONE 2>/dev/null || echo none); echo "$(date -u +%FT%TZ) step1: $v" >> QUEUE_B2_STEP1_VERDICT
[ "$v" = OK ] || { echo "step 1 not OK; full queue not started" >> QUEUE_B2_STEP1_VERDICT; exit 1; }
grep -q '"status": "passed"' mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_8b/full/gate_rel_complete.json \
  || { echo "step 1 gate not passed; full queue not started" >> QUEUE_B2_STEP1_VERDICT; exit 1; }
echo "$(date -u +%FT%TZ) starting queue_b2_frja.sh" >> /workspace/launch_b2.log
exec bash queue_b2_frja.sh >> /workspace/launch_b2.log 2>&1
