#!/usr/bin/env bash
# Flow tests for queue_internvl25_expA.sh in a fake /workspace (W=...). Nothing leaves this machine, no GPU, no pip, no model download.
#  "real" tests: real setup(), model() (offline, against a fake HF cache holding the REAL small files of the pinned revision and sparse safetensors
#                of the true sizes), real run_expA_ivl25.sh with --mock-model (LOCAL_MOCK=1), real gates.
#  "stub" tests: setup/model/smoke/full replaced by stubs (make_stub_queue.py) to exercise waiting, retry, timeout, ETA exit code, cache rule.
# Stand-ins in shims/: hf, nvidia-smi, curl, timeout, df, flock, sha256sum, python3.
# usage: bash run_ivl25_queue_tests.sh <name>|all     names: real wait stop6 others net fail setup sigint dup plan
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
TEMPLATE_REPO=$S/ivl25_repo          # mpr-ivl stand-in: 666440 + 5 patches, data, venv stub
FAKE_HF=$S/ivl25_work/hf_fake
export PIP_INDEX_URL=http://127.0.0.1:9/simple PIP_EXTRA_INDEX_URL= PIP_RETRIES=0 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1   # tests must never reach a package index
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 POLL_S=1 GPU_POLL_S=1 FAKE_STATE="$S/ivq/state" HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh'  # stale queue_minicpm.sh test processes of earlier sessions run on this Mac
BAD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {  # $1 name $2 = real|stub
  export W=$S/ivq/ws_$1; rm -rf "$W" "$FAKE_STATE"; mkdir -p "$W" "$FAKE_STATE"
  if [ "$2" = stub ]; then python3 "$HERE/make_stub_queue.py" "$M/queue_internvl25_expA.sh" "$W/queue_internvl25_expA.sh"
  else cp "$M/queue_internvl25_expA.sh" "$W/"; fi
  cp "$M/internvl25_expA_addon_20261114.tar.gz" "$W/"
  if [ "$2" = real ]; then
    cp -Rl "$TEMPLATE_REPO" "$W/mpr-ivl"; rm -rf "$W/mpr-ivl/followup_rel_4lang/expA_ivl25" "$W/mpr-ivl/followup_rel_4lang/results"
    printf '#!/bin/bash\nPYTHONPATH="%s/mpr-ivl/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$W" "$S" > "$W/mpr-ivl/.venv/bin/python.new"
    rm "$W/mpr-ivl/.venv/bin/python" "$W/mpr-ivl/.venv/bin/activate" "$W/mpr-ivl/.venv/bin/python3" 2>/dev/null
    mv "$W/mpr-ivl/.venv/bin/python.new" "$W/mpr-ivl/.venv/bin/python"; chmod +x "$W/mpr-ivl/.venv/bin/python"; ln -s python "$W/mpr-ivl/.venv/bin/python3"
    printf 'export PATH="%s/mpr-ivl/.venv/bin:$PATH"\n' "$W" > "$W/mpr-ivl/.venv/bin/activate"
    mkdir -p "$W/mpr-ivl/.cache"; cp -R "$FAKE_HF" "$W/mpr-ivl/.cache/huggingface"
  else
    mkdir -p "$W/mpr-ivl/.git" "$W/mpr-ivl/.venv/bin"; touch "$W/mpr-ivl/.venv/bin/activate"
  fi
  sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$(shasum -a 256 "$W/internvl25_expA_addon_20261114.tar.gz" | cut -d' ' -f1)/" "$W/queue_internvl25_expA.sh"
}
fake_proc() {  # a process whose command line contains $1 for $2 seconds
  mkdir -p "$S/ivq/fakeproc"; printf '#!/bin/bash\nsleep %s\n' "$2" > "$S/ivq/fakeproc/$1"; bash "$S/ivq/fakeproc/$1"
}
q() { ( cd "$W" && W="$W" bash queue_internvl25_expA.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE7_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE7_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
ST() { echo "$W/QUEUE7_STATUS"; }

t_real() {
  echo "### real: QUEUE6_DONE present; real setup/model(offline fake cache)/smoke/full with --mock-model"
  mkws real real; echo OK > "$W/QUEUE6_DONE"; q
  ck "QUEUE7_DONE = OK" done_is OK
  ck "network line with MB/s" has "$(ST)" 'huggingface.co 40.0 MB/s'
  ck "setup ok" has "$(ST)" 'step setup ok'
  ck "model ok (real checks: bytes, InternLM2 arch, A-D single tokens 290/309/289/302)" has "$(ST)" 'step model ok'
  ck "model log shows label ids" has "$W/queue_results/internvl25_expA/logs/model.log" "'A': \[290\]"
  ck "model log shows InternLM2ForCausalLM" has "$W/queue_results/internvl25_expA/logs/model.log" 'InternLM2ForCausalLM'
  ck "smoke gate line in status" has "$(ST)" 'smoke gate: expa: acc'
  ck "smoke passed + full done" has "$W/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/STATUS" 'Exp A full done'
  ck "full gate passed with 5,688 rows" bash -c "grep -q '\"status\": \"passed\"' $W/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/expa_full/gate_expa_complete.json && grep -q '\"rows\": 5688' $W/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/expa_full/gate_expa_complete.json"
  ck "addon unpacked in repo" test -s "$W/mpr-ivl/followup_rel_4lang/expA_ivl25/run_expA_ivl25.sh"
  ck "results tar + sha256" bash -c "ls $W/queue_results/internvl25_expA/internvl25_expA_results_*.tar.gz && ls $W/queue_results/internvl25_expA/*.sha256"
  ck "model cache of this model deleted after success" test ! -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  ck "rest of the HF cache dir untouched" test -d "$W/mpr-ivl/.cache/huggingface/hub"
  mkws real2 real; echo OK > "$W/QUEUE6_DONE"; KEEP_MODEL_CACHE=1 q; q
  ck "rerun resumes: still OK (full already done)" done_is OK
  ck "rerun says full already done" has "$W/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/STATUS" 'already done'
}
t_wait() {
  echo "### wait: no QUEUE6_DONE -> waits; proceeds when it appears; bounded wait aborts; QUEUE_DONE (old name) does NOT start it"
  mkws wait stub; ( sleep 4; echo OK > "$W/QUEUE6_DONE" ) & q; wait
  ck "proceeded after QUEUE6_DONE appeared" done_is OK
  mkws wait2 stub; MAX_WAIT_S=3 q
  ck "bounded wait: FAILED steps: wait" done_starts "FAILED steps: wait"
  ck "status mentions the bound" has "$(ST)" 'did not appear within'
  mkws wait3 stub; echo OK > "$W/QUEUE_DONE"; MAX_WAIT_S=3 q
  ck "QUEUE_DONE of the sizes queue is not enough" done_starts "FAILED steps: wait"
  mkws wait4 stub; echo OK > "$W/MY_GATE"; WAIT_FOR="$W/MY_GATE" q
  ck "WAIT_FOR override works" done_is OK
  ck "default max wait is 168 h" grep -q 'MAX_WAIT_H:-168' "$M/queue_internvl25_expA.sh"
}
t_others() {
  echo "### others: QUEUE6_DONE exists but queue_6x6.sh / run_all_overnight.sh still runs -> keep waiting"
  for p in queue_6x6.sh run_all_overnight.sh; do
    mkws others stub; echo OK > "$W/QUEUE6_DONE"
    fake_proc $p 8 & local pid=$!; sleep 0.5
    local t0=$SECONDS; q; local took=$((SECONDS - t0)); wait $pid 2>/dev/null
    ck "$p: waited for it" has "$(ST)" 'still running'
    ck "$p: ran after it ended" done_is OK
    ck "$p: waited at least 6 s" test "$took" -ge 6
  done
  mkws others2 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_GPU_USED=20000 GPU_POLL_S=0 q
  ck "static: default pattern covers overnight, 6x6, minicpm, sizes queue" grep -q 'run_all_overnight.*queue_6x6.*queue_minicpm.*queue_sizes_uitars' "$M/queue_internvl25_expA.sh"
  ck "GPU busy -> FAILED steps: GPU" done_starts "FAILED steps: gpu"
}
t_net() {
  echo "### net: slow -> STOP; FORCE_SLOW_NET=1 continues; HTTP 404 -> warning only; unreachable -> STOP"
  mkws net stub; echo OK > "$W/QUEUE6_DONE"; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow: FAILED steps: network" done_starts "FAILED steps: network"
  ck "MB/s in status" has "$(ST)" 'huggingface.co 0.6 MB/s'
  ck "nothing downloaded or set up" test ! -d "$W/mpr-ivl/.cache"
  mkws net2 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws net3 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_NET_HF="0 1.0 404" q
  ck "404: warning, continues" done_is OK
  mkws net4 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_NET_HF="0 1.0 000" q
  ck "unreachable: STOP" done_starts "FAILED steps: network"
}
t_fail() {
  echo "### fail: retries, failed smoke/full, timeout, ETA exit 3, cache rule"
  mkws fail1 stub; echo OK > "$W/QUEUE6_DONE"; FAIL_DL_FIRST=2 q
  ck "2 transient download failures, then ok (retry x3)" done_is OK
  mkws fail2 stub; echo OK > "$W/QUEUE6_DONE"; FAIL_DL_FIRST=3 q
  ck "3 failures: model step FAILED" done_starts "FAILED steps: model"
  ck "partial cache kept" test -d "$W/mpr-ivl/.cache/huggingface"
  ck "smoke/full not run after model failure" test ! -e "$W/mpr-ivl/steps_run.txt"
  mkws fail3 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_FAIL_STEP=full_expa q
  ck "FAILED steps: full_expa" done_is "FAILED steps: full_expa"
  ck "cache kept after failure" test -d "$W/mpr-ivl/.cache/huggingface"
  ck "status says kept" has "$(ST)" 'model cache KEPT'
  mkws fail4 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_FAIL_STEP=smoke_expa q
  ck "smoke failure: full not run" test ! -e "$W/mpr-ivl/steps_run.txt"
  ck "smoke failure reported" has "$(ST)" 'smoke failed'
  mkws fail5 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_FAIL_STEP=full_expa_sleep FAKE_TIMEOUT_MATCH="full_expa" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported" has "$(ST)" 'step full_expa TIMEOUT'
  mkws fail6 stub; echo OK > "$W/QUEUE6_DONE"; FAKE_FAIL_STEP=full_expa_eta q
  ck "ETA stop (exit 3) -> FAILED steps: full_expa_eta" done_is "FAILED steps: full_expa_eta"
  ck "ETA stop message" has "$(ST)" 'STOP step full_expa'
  mkws fail7 stub; echo OK > "$W/QUEUE6_DONE"; KEEP_MODEL_CACHE=1 q
  ck "KEEP_MODEL_CACHE=1 keeps cache on success" test -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  mkws fail8 stub; echo OK > "$W/QUEUE6_DONE"; q
  ck "success: this model's cache deleted" test ! -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  ck "static: model() uses retry 3 hf download" grep -q 'retry 3 hf download' "$M/queue_internvl25_expA.sh"
}
t_setup() {
  echo "### setup: real setup() must refuse a wrong repo / config / venv (negative cases)"
  mkws s1 real; echo OK > "$W/QUEUE6_DONE"; sed -i '' 's/INTERNVL_FAMILIES/INTERNVL_FAMILIEZ/' "$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/run_cross_locale.py" 2>/dev/null
  rm -f "$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/run_cross_locale.py.bak"
  q; ck "repo without InternVL patch marker: setup FAILED" done_starts "FAILED steps: setup"
  mkws s2 real; echo OK > "$W/QUEUE6_DONE"; cp "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml" "$W/y"; rm "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml"; sed 's/max_num: 7/max_num: 8/' "$W/y" > "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml"
  q; ck "repo config differs from the addon config: setup FAILED" done_starts "FAILED steps: setup"
  mkws s3 real; echo OK > "$W/QUEUE6_DONE"; EXPECT_TORCH=9.9.9 q
  ck "wrong torch version: venv check fails (would pip install; stubbed out here -> setup FAILED)" done_starts "FAILED steps: setup"
  mkws s4 real; echo OK > "$W/QUEUE6_DONE"; rm -rf "$W/mpr-ivl"; q
  ck "no mpr-ivl: STOP before any step" done_starts "FAILED steps: repo"
  mkws s5 real; echo OK > "$W/QUEUE6_DONE"; echo "0000  x" > /dev/null; sed -i '' 's/^ADDON_SHA=.*/ADDON_SHA=0000/' "$W/queue_internvl25_expA.sh"; q
  ck "addon sha mismatch: STOP" done_starts "FAILED steps: addon"
  mkws s6 real; echo OK > "$W/QUEUE6_DONE"; rm -rf "$W/mpr-ivl/.cache/huggingface/hub"; FAKE_FREE=10 q
  ck "10 GB free and no cache: STOP (need 30)" done_starts "FAILED steps: disk"
  mkws s7 real; echo OK > "$W/QUEUE6_DONE"; truncate -s 100 "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463/model-00004-of-00004.safetensors"; q
  ck "wrong safetensors size: model step FAILED" done_starts "FAILED steps: model"
}
t_sigint() {
  echo "### sigint: TERM while waiting -> no QUEUE7_DONE"
  mkws sigint stub
  ( cd "$W" && W="$W" bash queue_internvl25_expA.sh > "$W/out.txt" 2>&1 ) & local p=$!
  sleep 3; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; wait $p 2>/dev/null; sleep 1
  ck "no QUEUE7_DONE" test ! -e "$W/QUEUE7_DONE"
  ck "status says interrupted" has "$(ST)" 'interrupted by signal'
}
t_dup() {
  echo "### dup: second copy refuses (flock)"
  mkws dup stub
  ( cd "$W" && W="$W" bash queue_internvl25_expA.sh > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 3; q
  ck "second run says already running" has "$W/out.txt" 'already running'
  kill -TERM $p 2>/dev/null; wait $p 2>/dev/null
}
t_plan() {
  echo "### plan: --plan prints and changes nothing"
  mkws plan stub; q --plan
  ck "plan mentions QUEUE6_DONE" has "$W/out.txt" 'wait for /.*QUEUE6_DONE'
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE7_STATUS ] && [ ! -e $W/QUEUE7_DONE ] && [ ! -d $W/queue_results ]"
}
t_stop6() {
  echo "### stop6: queue_6x6.sh STOPped early (no QUEUE6_DONE, not running) -> continue after 2 checks; a waiting line does not count"
  mkws stop6 stub; echo "2026-11-15T00:00:00Z STOP: network too slow" > "$W/QUEUE6_STATUS"; MAX_WAIT_S=30 q
  ck "proceeded after early STOP" done_is OK
  ck "status says continuing without QUEUE6_DONE" has "$(ST)" 'continuing without'
  mkws stop6b stub; echo "2026-11-15T00:00:00Z 6x6: still waiting for /workspace/MINICPM_DONE" > "$W/QUEUE6_STATUS"; MAX_WAIT_S=4 q
  ck "waiting line does not release" done_starts "FAILED steps: wait"
  mkws stop6c stub; echo "2026-11-15T00:00:00Z STOP: x" > "$W/QUEUE6_STATUS"; fake_proc queue_6x6.sh 6 & local pid=$!; sleep 0.5; MAX_WAIT_S=4 q; wait $pid 2>/dev/null
  ck "running queue_6x6.sh blocks release" done_starts "FAILED steps: wait"
}
case "${1:-}" in
  all) for t in plan real wait stop6 others net fail setup sigint dup; do "t_$t"; done ;;
  real|wait|stop6|others|net|fail|setup|sigint|dup|plan) "t_$1" ;;
  *) echo "usage: $0 <real|wait|others|net|fail|setup|sigint|dup|plan|all>"; exit 2 ;;
esac
echo "failed checks: $BAD"; exit $((BAD > 0))
