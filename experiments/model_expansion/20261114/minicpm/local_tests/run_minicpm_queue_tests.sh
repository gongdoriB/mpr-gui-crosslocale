#!/usr/bin/env bash
# Flow tests for queue_minicpm.sh in a fake /workspace (W=...). The heavy functions (setup, model, smoke, full) are replaced by stubs
# (make_stub_queue.py); waiting, GPU wait, network check, retry, ETA guard, cache rule, trap and packaging are the real code.
# Stub commands: hf/nvidia-smi/curl/timeout/df/flock/sha256sum/python3 in shims/. Nothing leaves this machine.
# usage: bash run_minicpm_queue_tests.sh <name>|all     names: ok wait overnight net fail eta sigint dup
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 POLL_S=1 FAKE_STATE="$S/mcq/state"
BAD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {
  export W=$S/mcq/ws_$1; rm -rf "$W" "$FAKE_STATE"; mkdir -p "$W" "$FAKE_STATE"
  python3 "$HERE/make_stub_queue.py" "$M/upload/queue_minicpm.sh" "$W/queue_minicpm.sh"
  cp "$M/upload/minicpm_addon_20261114.tar.gz" "$W/"
  sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$(shasum -a 256 "$W/minicpm_addon_20261114.tar.gz" | cut -d' ' -f1)/" "$W/queue_minicpm.sh"
}
fake_overnight() {  # a process whose command line contains run_all_overnight.sh for $1 seconds
  mkdir -p "$S/mcq/fakeproc"; printf '#!/bin/bash\nsleep %s\n' "$1" > "$S/mcq/fakeproc/run_all_overnight.sh"; bash "$S/mcq/fakeproc/run_all_overnight.sh"
}
q() { ( cd "$W" && W="$W" bash queue_minicpm.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/MINICPM_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/MINICPM_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
ST() { echo "$W/QUEUE_STATUS"; }

t_ok() {
  echo "### ok: QUEUE_DONE with content 'STOPPED rc=1' (any content) -> proceeds; all steps; cache deleted on success"
  mkws ok; echo "STOPPED rc=1" > "$W/QUEUE_DONE"; q
  ck "MINICPM_DONE = OK" done_is OK
  ck "QUEUE_DONE content shown" has "$(ST)" 'QUEUE_DONE found: STOPPED rc=1'
  ck "network line with MB/s" has "$(ST)" 'huggingface.co 40.0 MB/s'
  ck "smoke accuracy reported" has "$(ST)" 'smoke gates: rel: acc 0.41'
  ck "ETA checks ran" has "$(ST)" 'step full_rel ETA check'
  ck "both full steps ran" test "$(wc -l < "$W/mpr-minicpm/steps_run.txt" | tr -d ' ')" = 2
  ck "model cache deleted after success" test ! -d "$W/mpr-minicpm/.cache/huggingface"
  ck "results tar written" bash -c "ls $W/queue_results/minicpm/minicpm_results_*.tar.gz"
}
t_wait() {
  echo "### wait: no QUEUE_DONE -> waits; proceeds when it appears; bounded wait aborts"
  mkws wait; ( sleep 4; echo OK > "$W/QUEUE_DONE" ) & q; wait
  ck "proceeded after QUEUE_DONE appeared" done_is OK
  mkws wait2; MAX_WAIT_S=3 q
  ck "bounded wait: MINICPM_DONE says FAILED" done_starts "FAILED steps: QUEUE_DONE"
  ck "status mentions the bound" has "$(ST)" 'did not appear within'
  ck "default max wait is 48 h" grep -q 'MAX_WAIT_H:-48' "$M/upload/queue_minicpm.sh"
}
t_overnight() {
  echo "### overnight: QUEUE_DONE exists (early STOP of the sizes queue) but run_all_overnight.sh still runs -> keep waiting"
  mkws overnight; echo "STOPPED rc=1" > "$W/QUEUE_DONE"
  fake_overnight 8 & local p=$!; sleep 0.5
  local t0=$SECONDS; q; local took=$((SECONDS - t0)); wait $p 2>/dev/null
  ck "waited for the overnight job" has "$(ST)" 'run_all_overnight.sh is still running'
  ck "ran after it ended" done_is OK
  ck "waited at least 6 s" test "$took" -ge 6
}
t_net() {
  echo "### net: slow -> STOP; FORCE_SLOW_NET=1 continues; HTTP 404 on the test URL -> warning only"
  mkws net; echo OK > "$W/QUEUE_DONE"; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow: MINICPM_DONE FAILED steps: network" done_starts "FAILED steps: network"
  ck "MB/s in status" has "$(ST)" 'huggingface.co 0.6 MB/s'
  ck "nothing set up" test ! -d "$W/mpr-minicpm"
  mkws net2; echo OK > "$W/QUEUE_DONE"; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws net3; echo OK > "$W/QUEUE_DONE"; FAKE_NET_HF="0 1.0 404" q
  ck "404: warning, continues" done_is OK
  mkws net4; echo OK > "$W/QUEUE_DONE"; FAKE_NET_HF="0 1.0 000" q
  ck "unreachable: STOP" done_starts "FAILED steps: network"
}
t_fail() {
  echo "### fail: retries; failure keeps the model cache when disk allows; low disk deletes it"
  mkws fail1; echo OK > "$W/QUEUE_DONE"; FAIL_DL_FIRST=2 q
  ck "2 transient download failures, then ok (retry x3)" done_is OK
  mkws fail2; echo OK > "$W/QUEUE_DONE"; FAIL_DL_FIRST=3 q
  ck "3 failures: model step FAILED" done_starts "FAILED steps: model"
  ck "partial cache kept (80 GB free)" test -d "$W/mpr-minicpm/.cache/huggingface"
  ck "status says kept" has "$(ST)" 'model cache KEPT'
  ck "pip cache deleted" test ! -d "$W/mpr-minicpm/.cache/pip"
  mkws fail3; echo OK > "$W/QUEUE_DONE"; FAKE_FAIL_STEP=full_rel q
  ck "full_rel failed, full_expa still ran" test "$(wc -l < "$W/mpr-minicpm/steps_run.txt" | tr -d ' ')" = 2
  ck "FAILED steps: full_rel" done_is "FAILED steps: full_rel"
  ck "cache kept after failure" test -d "$W/mpr-minicpm/.cache/huggingface"
  mkws fail4; echo OK > "$W/QUEUE_DONE"; MIN_FREE_GB=20 FAKE_FREE=30 FAKE_FAIL_STEP=full_rel q
  ck "low disk (30 < 52): cache deleted" has "$(ST)" "model cache deleted (not finished; free 30 GB < 52 GB"
  ck "no model cache left" test ! -d "$W/mpr-minicpm/.cache/huggingface"
  mkws fail5; echo OK > "$W/QUEUE_DONE"; MIN_FREE_GB=20 FAKE_FREE=30 KEEP_MODEL_CACHE=1 FAKE_FAIL_STEP=full_rel q
  ck "KEEP_MODEL_CACHE=1 keeps it" test -d "$W/mpr-minicpm/.cache/huggingface"
  mkws fail6; echo OK > "$W/QUEUE_DONE"; FAKE_FAIL_STEP=smoke_rel q
  ck "smoke failure: full steps not run" test ! -e "$W/mpr-minicpm/steps_run.txt"
  ck "smoke failure reported" has "$(ST)" 'smoke failed'
  mkws fail7; echo OK > "$W/QUEUE_DONE"; FAKE_FAIL_STEP=full_rel_sleep FAKE_TIMEOUT_MATCH="full_rel" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported" has "$(ST)" 'step full_rel TIMEOUT'
  ck "cache kept after timeout" test -d "$W/mpr-minicpm/.cache/huggingface"
  ck "static: model() uses retry 3 hf download" grep -q 'retry 3 hf download' "$M/upload/queue_minicpm.sh"
  ck "static: git clone / pip / fetch-images use retry" bash -c "grep -q 'retry 5 clone_repo' '$M/upload/queue_minicpm.sh' && grep -q 'retry 3 python -m pip install -r' '$M/upload/queue_minicpm.sh' && grep -q 'retry 6 python followup_rel_4lang/data_bundle.py fetch-images' '$M/upload/queue_minicpm.sh'"
}
t_eta() {
  echo "### eta: smoke ETA longer than the step limit -> step not started, other step runs"
  mkws eta; echo OK > "$W/QUEUE_DONE"; FAKE_ETA_REL_S=100000 q
  ck "STOP step full_rel in status" has "$(ST)" 'STOP step full_rel'
  ck "FAILED steps: full_rel_eta" done_is "FAILED steps: full_rel_eta"
  ck "only full_expa ran" bash -c "[ \"\$(cat $W/mpr-minicpm/steps_run.txt)\" = 'STUB full_expa' ]"
  mkws eta2; echo OK > "$W/QUEUE_DONE"; FAKE_ETA_REL_S=100000 IGNORE_ETA=1 q
  ck "IGNORE_ETA=1 starts it" done_is OK
  mkws eta3; echo OK > "$W/QUEUE_DONE"; FAKE_ETA_EXPA_S=40000 q
  ck "expa ETA 11.1 h > 3 h: full_expa_eta" done_is "FAILED steps: full_expa_eta"
}
t_sigint() {
  echo "### sigint: TERM while waiting -> no MINICPM_DONE"
  mkws sigint
  ( cd "$W" && W="$W" bash queue_minicpm.sh > "$W/out.txt" 2>&1 ) & local p=$!
  sleep 3; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; wait $p 2>/dev/null; sleep 1
  ck "no MINICPM_DONE" test ! -e "$W/MINICPM_DONE"
  ck "status says interrupted" has "$(ST)" 'interrupted by signal'
}
t_dup() {
  echo "### dup: second copy refuses (flock)"
  mkws dup
  ( cd "$W" && W="$W" bash queue_minicpm.sh > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 3; q
  ck "second run says already running" has "$W/out.txt" 'already running'
  kill -TERM $p 2>/dev/null; wait $p 2>/dev/null
}
case "${1:-}" in
  all) for t in ok wait overnight net fail eta sigint dup; do "t_$t"; done ;;
  ok|wait|overnight|net|fail|eta|sigint|dup) "t_$1" ;;
  *) echo "usage: $0 <ok|wait|overnight|net|fail|eta|sigint|dup|all>"; exit 2 ;;
esac
echo "failed checks: $BAD"; exit $((BAD > 0))
