#!/usr/bin/env bash
# Flow tests for queue_sizes_uitars.sh in a fake /workspace (WORKSPACE=...) with stub hf/nvidia-smi/curl/timeout/df/flock and mock inference
# (LOCAL_MOCK=1: no GPU, no model). Nothing leaves this machine. The two repos are symlinks to the local scratch copies.
# usage: bash run_sz_queue_tests.sh <name>|all
#   names: ok partial_overnight noproc bound onefam gpu dup sigint net fail eta
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; QS="$(dirname "$HERE")"
export PATH="$HERE/shims:$PATH"
export LOCAL_MOCK=1 RETRY_SLEEP=0 POLL_S=1 GPU_POLL_S=1 FAKE_STATE="$S/szq/state"
QWEN_SRC=$S/qwen_sizes; IVL_SRC=$S/ivl_sizes
BAD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {  # $1 name; fresh fake /workspace
  export W=$S/szq/ws_$1; rm -rf "$W" "$FAKE_STATE"; mkdir -p "$W" "$FAKE_STATE"
  cp "$QS/queue_sizes_uitars.sh" "$QS/model_expansion_sizes_uitars_20261114.tar.gz" "$W/"
  ln -s $QWEN_SRC "$W/mpr-gui-crosslocale"; ln -s $IVL_SRC "$W/mpr-ivl"
  for r in $QWEN_SRC $IVL_SRC; do rm -rf $r/.cache/huggingface $r/followup_rel_4lang/results/sizes_*; mkdir -p $r/.cache/huggingface; done
  mkdir -p "$S/szq/b1_clean"
  echo "2026-11-14T00:00:00Z overnight started" > "$W/OVERNIGHT_STATUS"
}
fin() { echo "2026-11-14T09:00:00Z FINISHED qwen3=$1 internvl35=$2. Download" >> "$W/OVERNIGHT_STATUS"; }
fake8b() { mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/x" "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/x"; }
fake_overnight() {  # a process whose command line contains run_all_overnight.sh for $1 seconds
  mkdir -p "$S/szq/fakeproc"; printf '#!/bin/bash\nsleep %s\n' "$1" > "$S/szq/fakeproc/run_all_overnight.sh"; bash "$S/szq/fakeproc/run_all_overnight.sh"
}
q() { ( cd "$W" && WORKSPACE="$W" bash queue_sizes_uitars.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
rc() { cat "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }

t_ok() {
  echo "### ok: overnight ok/ok; batch-1 check runs; 8B caches deleted; 2 models; QUEUE_DONE=OK"
  mkws ok; fin ok ok; fake8b
  ONLY="qwen3_vl_2b internvl35_1b" q
  ck "exit 0" test "$(rc)" = 0
  ck "QUEUE_DONE = OK" done_is OK
  ck "batch-1 result json written" test -s "$W/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json"
  ck "batch-1 ran 1,008 inferences at batch size 1" has "$W/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json" '"inferences_in_run": 1008'
  ck "batch-1 compares O and R rows (672)" has "$W/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json" '"compared_rows": 672'
  ck "batch-1 status line in QUEUE_STATUS" has "$W/QUEUE_STATUS" 'batch-1 check done'
  ck "8B caches deleted" test ! -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  ck "network line with MB/s in QUEUE_STATUS" has "$W/QUEUE_STATUS" 'huggingface.co 40.0 MB/s'
  ck "both models COMPLETE" test -f "$W/queue_results/qwen3_vl_2b/COMPLETE" -a -f "$W/queue_results/internvl35_1b/COMPLETE"
  ck "smoke accuracy reported in status" has "$W/QUEUE_STATUS" 'smoke gates: rel: acc'
  ck "model caches deleted after success" test ! -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct"
}
t_partial_overnight() {
  echo "### partial_overnight: qwen3=ok internvl35=FAILED -> batch-1 check still runs, 8B caches KEPT"
  mkws partial; fin ok FAILED; fake8b
  ONLY="qwen3_vl_2b" q
  ck "QUEUE_DONE = OK" done_is OK
  ck "8B caches kept" test -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct" -a -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B"
  ck "status says kept" has "$W/QUEUE_STATUS" '8B caches kept'
  ck "batch-1 check ran" test -s "$W/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json"
  mkws partial2; fin FAILED ok; fake8b; ONLY="qwen3_vl_2b" q
  ck "qwen3=FAILED: batch-1 check skipped" has "$W/QUEUE_STATUS" 'batch-1 check skipped'
  ck "qwen3=FAILED: no batch-1 json" test ! -e "$W/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json"
}
t_noproc() {
  echo "### noproc: no run_all_overnight.sh process and no FINISHED -> STOP after the grace period, QUEUE_DONE=STOPPED"
  mkws noproc
  OVERNIGHT_GRACE_S=3 ONLY="qwen3_vl_2b" q
  ck "exit 1" test "$(rc)" = 1
  ck "QUEUE_DONE = STOPPED rc=1" done_is "STOPPED rc=1"
  ck "status explains" has "$W/QUEUE_STATUS" 'overnight job probably stopped itself'
  ck "nothing was started" test ! -d "$W/queue_results/qwen3_vl_2b"
}
t_bound() {
  echo "### bound: overnight process exists but never writes FINISHED -> STOP at the upper bound"
  mkws bound
  fake_overnight 40 & local p=$!; sleep 0.5
  OVERNIGHT_MAX_S=5 OVERNIGHT_GRACE_S=2 ONLY="qwen3_vl_2b" q   # grace 2 s: only the running process keeps it from the "gone" STOP
  kill $p 2>/dev/null; wait $p 2>/dev/null
  ck "QUEUE_DONE = STOPPED rc=1" done_is "STOPPED rc=1"
  ck "status says bound" has "$W/QUEUE_STATUS" 'no FINISHED line within'
}
t_onefam() {
  echo "### onefam: mpr-ivl missing -> only the InternVL models are skipped; Qwen model runs; repo checks happen after the wait"
  mkws onefam; rm -f "$W/mpr-ivl"; fin ok ok
  ONLY="qwen3_vl_2b internvl35_1b" q
  ck "QUEUE_DONE = OK" done_is OK
  ck "qwen3_vl_2b complete" test -f "$W/queue_results/qwen3_vl_2b/COMPLETE"
  ck "internvl35_1b skipped with reason" has "$W/QUEUE_STATUS" 'internvl35_1b SKIPPED: its repo'
  ck "warning about the missing repo" has "$W/QUEUE_STATUS" 'mpr-ivl is not set up'
  mkws onefam2; rm -f "$W/mpr-ivl" "$W/mpr-gui-crosslocale"; fin ok ok; ONLY="qwen3_vl_2b" q
  ck "both repos missing: STOP, QUEUE_DONE STOPPED" done_is "STOPPED rc=1"
  # the repo check must come AFTER the wait: start with no repos and no FINISHED, create FINISHED later
  mkws onefam3; rm -f "$W/mpr-ivl" "$W/mpr-gui-crosslocale"
  fake_overnight 30 & local p=$!; sleep 0.5
  ( sleep 3; ln -s $QWEN_SRC "$W/mpr-gui-crosslocale"; fin ok FAILED ) &
  ONLY="qwen3_vl_2b" q; kill $p 2>/dev/null; wait 2>/dev/null
  ck "repos that appear while waiting are accepted" test -f "$W/queue_results/qwen3_vl_2b/COMPLETE"
}
t_gpu() {
  echo "### gpu: GPU busy (memory.used 5000 MiB), also with SKIP_WAIT=1 -> STOP"
  mkws gpu; fin ok ok
  FAKE_GPU_USED=5000 GPU_WAIT_MIN=1 SKIP_WAIT=1 ONLY="qwen3_vl_2b" q
  ck "QUEUE_DONE = STOPPED rc=1" done_is "STOPPED rc=1"
  ck "status mentions MiB" has "$W/QUEUE_STATUS" 'GPU still busy (5000 MiB used)'
  mkws gpu2; fin ok ok; FAKE_GPU_USED=1499 SKIP_WAIT=1 ONLY="qwen3_vl_2b" q
  ck "1499 MiB counts as idle" done_is OK
}
t_dup() {
  echo "### dup: second copy must refuse to run (flock)"
  mkws dup
  ( cd "$W" && WORKSPACE="$W" OVERNIGHT_GRACE_S=100 bash queue_sizes_uitars.sh > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 3
  q
  ck "second run exit 1" test "$(rc)" = 1
  ck "second run says already running" has "$W/out.txt" 'already running'
  ck "first run still waiting (no QUEUE_DONE)" test ! -e "$W/QUEUE_DONE"
  kill -TERM $p 2>/dev/null; wait $p 2>/dev/null
}
t_sigint() {
  echo "### sigint: TERM/INT while waiting -> no QUEUE_DONE"
  mkws sigint
  ( cd "$W" && WORKSPACE="$W" OVERNIGHT_GRACE_S=100 bash queue_sizes_uitars.sh > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt" ) & local p=$!
  sleep 3; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; wait $p 2>/dev/null; sleep 1
  ck "no QUEUE_DONE after TERM" test ! -e "$W/QUEUE_DONE"
  ck "status says interrupted" has "$W/QUEUE_STATUS" 'interrupted by signal'
}
t_net() {
  echo "### net: 6 MB/s -> STOP; FORCE_SLOW_NET=1 continues; unreachable -> STOP"
  mkws net; fin ok ok; FAKE_NET_HF="6000000 10.0 206" ONLY="qwen3_vl_2b" q
  ck "slow net: QUEUE_DONE = STOPPED rc=1" done_is "STOPPED rc=1"
  ck "status has MB/s" has "$W/QUEUE_STATUS" 'huggingface.co 0.6 MB/s'
  ck "nothing downloaded" test ! -d "$W/queue_results/qwen3_vl_2b"
  mkws net2; fin ok ok; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 ONLY="qwen3_vl_2b" q
  ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws net3; fin ok ok; FAKE_NET_HF="0 1.0 000" ONLY="qwen3_vl_2b" q
  ck "unreachable: STOP" done_is "STOPPED rc=1"
}
t_fail() {
  echo "### fail: transient download errors are retried; a failed download keeps the partial cache; a timeout keeps the cache; low disk deletes it"
  mkws fail1; fin ok ok; FAIL_DL_FIRST=2 ONLY="qwen3_vl_2b" q
  ck "2 transient failures then success (retry)" test -f "$W/queue_results/qwen3_vl_2b/COMPLETE"
  mkws fail2; fin ok ok; FAIL_DL=Qwen/Qwen3-VL-2B-Instruct ONLY="qwen3_vl_2b" q
  ck "download failure reported" has "$W/QUEUE_STATUS" 'qwen3_vl_2b DOWNLOAD FAILED'
  ck "partial cache kept" test -e "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/blobs/part.incomplete"
  ck "status says kept" has "$W/QUEUE_STATUS" 'qwen3_vl_2b cache KEPT'
  ck "QUEUE_DONE still OK" done_is OK
  mkws fail3; fin ok ok; FAKE_TIMEOUT_MATCH="run_model.sh qwen3_vl_2b rel" FAKE_TIMEOUT_SECS=3 ONLY="qwen3_vl_2b" q
  ck "rel TIMEOUT reported" has "$W/QUEUE_STATUS" 'rel=TIMEOUT'
  ck "package is _PARTIAL" bash -c "ls $W/queue_results/qwen3_vl_2b/*_PARTIAL.tar.gz"
  ck "cache kept after timeout" test -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct"
  ck "not COMPLETE" test ! -f "$W/queue_results/qwen3_vl_2b/COMPLETE"
  echo "  -- rerun resumes without downloading again"
  ( cd "$W" && WORKSPACE="$W" ONLY="qwen3_vl_2b" bash queue_sizes_uitars.sh > "$W/out2.txt" 2>&1 )
  ck "rerun: model completes using the kept cache" test -f "$W/queue_results/qwen3_vl_2b/COMPLETE"
  ck "rerun: no second transfer (1 transfer in total)" test "$(cat $FAKE_STATE/transfers)" = 1
  mkws fail4; fin ok ok; FAKE_FREE=30 FAKE_TIMEOUT_MATCH="run_model.sh qwen3_vl_2b rel" FAKE_TIMEOUT_SECS=3 ONLY="qwen3_vl_2b uitars_15_7b" q
  ck "low disk (30 GB < 52 GB needed by UI-TARS): cache deleted" has "$W/QUEUE_STATUS" 'qwen3_vl_2b cache deleted (not finished; free 30 GB < 52 GB'
  ck "UI-TARS skipped for disk" has "$W/QUEUE_STATUS" 'uitars_15_7b SKIPPED: only 30 GB free'
  mkws fail5; fin ok ok; ONLY="internvl35_1b" FAKE_TIMEOUT_MATCH="run_model.sh internvl35_1b expa" FAKE_TIMEOUT_SECS=3 q
  ck "last model, 80 GB free >= 40 handoff: cache kept" has "$W/QUEUE_STATUS" 'internvl35_1b cache KEPT'
  mkws fail6; fin ok ok; ONLY="internvl35_1b" KEEP_FAILED_CACHE=0 FAKE_TIMEOUT_MATCH="run_model.sh internvl35_1b expa" FAKE_TIMEOUT_SECS=3 q
  ck "KEEP_FAILED_CACHE=0 deletes" has "$W/QUEUE_STATUS" 'internvl35_1b cache deleted'
  mkws fail7; fin ok ok; ONLY="internvl35_1b" FAKE_TIMEOUT_MATCH="run_model.sh internvl35_1b smoke" FAKE_TIMEOUT_SECS=2 q
  ck "smoke timeout: model not complete, full steps skipped" has "$W/QUEUE_STATUS" 'internvl35_1b smoke/gate FAILED'
}
t_eta() {
  echo "### eta: smoke ETA longer than the step timeout -> that step is not started (STOPPED_ETA), the other step still runs"
  mkws eta; fin ok ok; FORCE_REL_LIMIT_S=0.0001 ONLY="qwen3_vl_2b" q
  ck "status shows rel=STOPPED_ETA" has "$W/QUEUE_STATUS" 'rel=STOPPED_ETA'
  ck "expa still ok" has "$W/QUEUE_STATUS" 'expa=ok'
  ck "STOPPED line in model STATUS" has "$S/qwen_sizes/followup_rel_4lang/results/sizes_qwen3_vl_2b/STATUS" 'STOPPED: REL ETA'
  ck "REL full never started" test ! -s "$S/qwen_sizes/followup_rel_4lang/results/sizes_qwen3_vl_2b/rel_full/predictions.jsonl"
  mkws eta2; fin ok ok; FORCE_EXPA_LIMIT_S=0.0001 ONLY="qwen3_vl_2b" q
  ck "expa=STOPPED_ETA" has "$W/QUEUE_STATUS" 'expa=STOPPED_ETA'
}

case "${1:-}" in
  all) for t in ok partial_overnight noproc bound onefam gpu dup sigint net fail eta; do "t_$t"; done ;;
  ok|partial_overnight|noproc|bound|onefam|gpu|dup|sigint|net|fail|eta) "t_$1" ;;
  *) echo "usage: $0 <ok|partial_overnight|noproc|bound|onefam|gpu|dup|sigint|net|fail|eta|all>"; exit 2 ;;
esac
echo "failed checks: $BAD"; exit $((BAD > 0))
