#!/usr/bin/env bash
# Waits until the running 6x6 queue (queue_b3_4b.sh / queue_6x6.sh, bash processes only) has ended, then starts queue_rstem6_b4b.sh
# (it refuses to start while another queue runs). r1 (2026-11-19, review fixes): anchored pgrep; NET_MIN_MBPS=5;
# up to 3 attempts 5 min apart when the queue stopped at its start checks (others / gpu / network), never after a model step ran.
cd /workspace
S=/workspace/QUEUE_RSTEM6B_STATUS
echo "$(date -u +%FT%TZ) waiter: waiting for queue_b3_4b.sh / queue_6x6.sh to end" >> "$S"
while pgrep -f '^(/bin/)?(ba)?sh .*(queue_b3_4b\.sh|queue_6x6\.sh)' >/dev/null 2>&1; do sleep 60; done
sleep 120
for a in 1 2 3; do
  echo "$(date -u +%FT%TZ) waiter: starting queue_rstem6_b4b.sh (attempt $a)" >> "$S"
  NET_MIN_MBPS=5 KEEP_MODEL_CACHE=1 bash queue_rstem6_b4b.sh
  v=$(cat /workspace/QUEUE_RSTEM6B_DONE 2>/dev/null)
  case "$v" in
    "FAILED steps: others"|"FAILED steps: gpu"|"FAILED steps: network") echo "$(date -u +%FT%TZ) waiter: start check failed ($v); retry in 5 min" >> "$S"; sleep 300 ;;
    *) echo "$(date -u +%FT%TZ) waiter: done ($v)" >> "$S"; exit 0 ;;
  esac
done
echo "$(date -u +%FT%TZ) waiter: gave up after 3 attempts" >> "$S"
