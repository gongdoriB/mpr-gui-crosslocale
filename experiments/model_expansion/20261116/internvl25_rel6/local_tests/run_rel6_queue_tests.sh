#!/usr/bin/env bash
# Flow tests for queue_internvl25_rel6.sh in a fake /workspace (W=...). Nothing leaves this machine, no GPU, no pip, no model download, no network
# (https_proxy points to a closed port as a second safety net).
#  "real" tests: real setup() (addon + fr/ja image tar unpack, verification, venv checks), real model() (offline, against a fake HF cache holding the REAL small
#                files of the pinned revision and sparse safetensors of the true sizes), real run_rel6_ivl25.sh with --mock-model (LOCAL_MOCK=1), real gates.
#  "stub" tests: setup/model/smoke/full replaced by stubs (make_stub_queue.py) to exercise waiting, retry, timeout, ETA exit code, cache rule.
# Stand-ins in shims/: hf, nvidia-smi, curl, timeout, df, flock, sha256sum, python3.
# usage: bash run_rel6_queue_tests.sh <name>|all     names: plan real subset wait stop7 others net fail setup imgtar sigint dup
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
TEMPLATE_REPO=$S/ivl25_repo          # mpr-ivl stand-in: 666440 + 5 patches, 4-language data + hard-linked images, venv stub
FAKE_HF=$S/ivl25_work/hf_fake
FRJA=${CUAI_ROOT}/CUAI_하계\ 컨퍼런스/followup_rel_6lang/assets   # source of the fr/ja images (read only; hard links)
export PIP_INDEX_URL=http://127.0.0.1:9/simple PIP_EXTRA_INDEX_URL= PIP_RETRIES=0 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 POLL_S=1 GPU_POLL_S=1 FAKE_STATE="$S/ivq6/state" HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 WAIT_AFTER=2000-01-01T00:00:00Z OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh'  # stale queue_minicpm.sh test processes of earlier sessions may run on this Mac
BAD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
OUTR=followup_rel_6lang/results/rel6_internvl25_8b
mkws() {  # $1 name $2 = real|stub  [$3 = tar  -> also link the fr/ja image tar]
  export W=$S/ivq6/ws_$1; rm -rf "$W" "$FAKE_STATE"; mkdir -p "$W" "$FAKE_STATE"
  if [ "$2" = stub ]; then python3 "$HERE/make_stub_queue.py" "$M/queue_internvl25_rel6.sh" "$W/queue_internvl25_rel6.sh"
  else cp "$M/queue_internvl25_rel6.sh" "$W/"; fi
  ln "$M/internvl25_rel6_addon_20261116.tar.gz" "$W/"
  if [ "${3:-}" = tar ]; then ln "$M/internvl25_rel6_frja_images_20261116.tar" "$W/"; fi
  if [ "$2" = real ]; then
    cp -Rl "$TEMPLATE_REPO" "$W/mpr-ivl"; rm -rf "$W/mpr-ivl/followup_rel_4lang/expA_ivl25" "$W/mpr-ivl/followup_rel_4lang/results"
    printf '#!/bin/bash\nPYTHONPATH="%s/mpr-ivl/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$W" "$S" > "$W/mpr-ivl/.venv/bin/python.new"
    rm "$W/mpr-ivl/.venv/bin/python" "$W/mpr-ivl/.venv/bin/activate" "$W/mpr-ivl/.venv/bin/python3" 2>/dev/null
    mv "$W/mpr-ivl/.venv/bin/python.new" "$W/mpr-ivl/.venv/bin/python"; chmod +x "$W/mpr-ivl/.venv/bin/python"; ln -s python "$W/mpr-ivl/.venv/bin/python3"
    printf 'export PATH="%s/mpr-ivl/.venv/bin:$PATH"\n' "$W" > "$W/mpr-ivl/.venv/bin/activate"
    mkdir -p "$W/mpr-ivl/.cache"; cp -R "$FAKE_HF" "$W/mpr-ivl/.cache/huggingface"
    if [ "${3:-}" != tar ]; then   # no image tar: the images are "already there" (hard links), as after an earlier fetch
      mkdir -p "$W/mpr-ivl/followup_rel_6lang"; cp -Rl "$FRJA" "$W/mpr-ivl/followup_rel_6lang/assets"
    fi
    ( cd "$W/mpr-ivl" && find followup_rel_4lang -type f | sort > "$W/files4_before.txt"; shasum -a 256 01_code/src/mpr_crosslocale/inference/run_rel_followup.py 01_code/src/mpr_crosslocale/data/rel_followup.py > "$W/orig_sha_before.txt" )
  else
    mkdir -p "$W/mpr-ivl/.git" "$W/mpr-ivl/.venv/bin"; touch "$W/mpr-ivl/.venv/bin/activate"
  fi
}
fake_proc() {  # a process whose command line contains $1 for $2 seconds
  mkdir -p "$S/ivq6/fakeproc"; printf '#!/bin/bash\nsleep %s\n' "$2" > "$S/ivq6/fakeproc/$1"; bash "$S/ivq6/fakeproc/$1"
}
q() { ( cd "$W" && W="$W" bash queue_internvl25_rel6.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE8_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE8_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
ST() { echo "$W/QUEUE8_STATUS"; }
GATE_FULL="$OUTR/full/gate_rel_complete.json"

t_real() {
  echo "### real: MINICPM_DONE present; real setup (addon + image tar)/model(offline fake cache)/smoke/full with --mock-model"
  mkws real real tar; echo OK > "$W/MINICPM_DONE"; q
  ck "QUEUE8_DONE = OK" done_is OK
  ck "network line with MB/s" has "$(ST)" 'huggingface.co 40.0 MB/s'
  ck "setup ok" has "$(ST)" 'step setup ok'
  ck "model ok (real checks: bytes, InternLM2 arch, A-D single tokens 290/309/289/302)" has "$(ST)" 'step model ok'
  ck "model log shows label ids" has "$W/queue_results/internvl25_rel6/logs/model.log" "'A': \[290\]"
  ck "image tar unpacked (729 fr/ja images)" bash -c "[ \$(find $W/mpr-ivl/followup_rel_6lang/assets -type f | wc -l) -eq 729 ]"
  ck "6-language modules import" has "$W/queue_results/internvl25_rel6/logs/setup.log" '6-language modules import ok'
  ck "setup log: 6-language verify passed (conditions sha256 locked)" has "$W/mpr-ivl/$OUTR/logs/bundle_verify.log" '63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed'
  ck "preflight: model config and system prompt equal the 2026-10-05 contract" has "$W/queue_results/internvl25_rel6/logs/smoke_rel6.log" 'equal the 2026-10-05 REL contract'
  ck "preflight: 4-language rows equal the 4-language file on the server (checked in the run script)" has "$W/mpr-ivl/$OUTR/logs/legacy_subset_check.log" '"rows": 23424\|subset check skipped'
  ck "preflight: legacy_subset_check ran on the real 4-language file (not skipped)" bash -c "! grep -q skipped $W/mpr-ivl/$OUTR/logs/legacy_subset_check.log"
  ck "smoke gate line in status" has "$(ST)" 'smoke gate: rel6: acc'
  ck "smoke plan: --max-items 2 -> 288 rows / 192 inferences" bash -c "grep -q '\"condition_rows\": 288' $W/mpr-ivl/$OUTR/smoke/plan.json && grep -q '\"actual_inferences\": 192' $W/mpr-ivl/$OUTR/smoke/plan.json"
  ck "full done" has "$W/mpr-ivl/$OUTR/STATUS" 'REL6 full done'
  ck "full gate passed with 44,489 rows" bash -c "grep -q '\"status\": \"passed\"' $W/mpr-ivl/$GATE_FULL && grep -q '\"rows\": 44489' $W/mpr-ivl/$GATE_FULL"
  ck "full summary: 52,572 condition rows, all success" bash -c "grep -q '\"condition_rows\": 52572' $W/mpr-ivl/$OUTR/full/summary.json && grep -q '\"success\": 52572' $W/mpr-ivl/$OUTR/full/summary.json"
  ck "gate reports per-language rows (fr and ja present)" bash -c "grep -q '\"fr\"' $W/mpr-ivl/$GATE_FULL && grep -q '\"ja\"' $W/mpr-ivl/$GATE_FULL"
  ck "addon files unpacked in repo" test -s "$W/mpr-ivl/followup_rel_6lang/ivl25_rel6/run_rel6_ivl25.sh"
  ck "results tar + sha256" bash -c "ls $W/queue_results/internvl25_rel6/internvl25_rel6_results_*.tar.gz && ls $W/queue_results/internvl25_rel6/*.sha256"
  ck "model cache of this model deleted after success" test ! -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  ck "rest of the HF cache dir untouched" test -d "$W/mpr-ivl/.cache/huggingface/hub"
  ck "4-language folder: same file list as before (no result written there)" bash -c "cd $W/mpr-ivl && find followup_rel_4lang -type f | sort | diff -q - $W/files4_before.txt"
  ck "original runner/data module byte-identical after the run" bash -c "cd $W/mpr-ivl && shasum -a 256 -c $W/orig_sha_before.txt"
  mkws real2 real tar; echo OK > "$W/MINICPM_DONE"; KEEP_MODEL_CACHE=1 q; q
  ck "rerun resumes: still OK (full already done)" done_is OK
  ck "rerun says full already done" has "$W/mpr-ivl/$OUTR/STATUS" 'already done'
}
t_subset() {
  echo "### subset: SUBSET=new_only -> 29,148 rows / 26,057 inferences in its own folder"
  mkws subset real; echo OK > "$W/MINICPM_DONE"; SUBSET=new_only q
  ck "QUEUE8_DONE = OK" done_is OK
  ck "new_only output folder rel6new_internvl25_8b with 26,057 inferences" bash -c "grep -q '\"actual_inferences\": 26057' $W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl25_8b/full/plan.json && grep -q '\"condition_rows\": 29148' $W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl25_8b/full/plan.json"
  ck "the all-rows folder was not created" test ! -d "$W/mpr-ivl/$OUTR"
}
t_wait() {
  echo "### wait: no fresh MINICPM_DONE -> waits; proceeds when it appears; stale marker ignored; bounded wait aborts; other markers do not start it"
  mkws wait stub; ( sleep 4; echo OK > "$W/MINICPM_DONE" ) & q; wait
  ck "proceeded after MINICPM_DONE appeared" done_is OK
  mkws wait2 stub; MAX_WAIT_S=3 q
  ck "bounded wait: FAILED steps: wait" done_starts "FAILED steps: wait"
  ck "status mentions the bound" has "$(ST)" 'did not appear within'
  mkws wait3 stub; echo OK > "$W/QUEUE6_DONE"; echo OK > "$W/QUEUE7_DONE"; echo OK > "$W/QUEUE_DONE"; MAX_WAIT_S=3 q
  ck "QUEUE_DONE / QUEUE6_DONE / QUEUE7_DONE are not the marker" done_starts "FAILED steps: wait"
  mkws wait5 stub; echo OK > "$W/MY_GATE"; WAIT_FOR="$W/MY_GATE" q
  ck "WAIT_FOR override works" done_is OK
  mkws wait6 stub; echo OK > "$W/QUEUE7_DONE"; WAIT_FOR="$W/QUEUE7_DONE" q
  ck "WAIT_FOR=QUEUE7_DONE (expA queue marker) works as an override" done_is OK
  mkws wait7 stub; echo OK > "$W/MINICPM_DONE"; touch -t 202011150310 "$W/MINICPM_DONE"; WAIT_AFTER=2026-11-16T02:11:00Z MAX_WAIT_S=3 q
  ck "stale marker (mtime 2020) older than WAIT_AFTER counts as absent" done_starts "FAILED steps: wait"
  ck "status says an older marker is ignored" has "$(ST)" 'older marker exists and is ignored'
  mkws wait8 stub; echo OK > "$W/MINICPM_DONE"; WAIT_AFTER=2026-11-16T02:11:00Z q
  ck "fresh marker (mtime now) newer than WAIT_AFTER passes" done_is OK
  ck "found line logs the marker mtime and WAIT_AFTER" has "$(ST)" 'is later than 2026-11-16T02:11:00Z'
  ck "start line logs WAIT_AFTER and the marker state" has "$(ST)" 'newer than 2026-11-16T02:11:00Z; marker now'
  mkws wait9 stub; echo OK > "$W/MINICPM_DONE"; WAIT_AFTER=garbage q
  ck "invalid WAIT_AFTER: STOP" done_starts "FAILED steps: wait"
  mkws wait10 stub; echo OK > "$W/MINICPM_DONE"; q
  ck "default WAIT_AFTER is 2026-11-16T02:11:00Z" grep -q 'WAIT_AFTER:-2026-11-16T02:11:00Z' "$M/queue_internvl25_rel6.sh"
  ck "default max wait is 168 h" grep -q 'MAX_WAIT_H:-168' "$M/queue_internvl25_rel6.sh"
}
t_minicpm() {
  echo "### minicpm: the running MiniCPM r2 queue (queue_minicpm_r2.sh) must hold this queue back"
  export OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh|queue_minicpm_r2\.sh'   # narrowed: stale fake expA processes of 2026-11-14 run on this Mac
  export WAIT_AFTER=2026-11-16T02:11:00Z
  mkws mc_a stub; echo OK > "$W/MINICPM_DONE"; touch -t 202611150310 "$W/MINICPM_DONE"
  fake_proc queue_minicpm_r2.sh 12 & local pid=$!; sleep 0.5; MAX_WAIT_S=4 q; wait $pid 2>/dev/null
  ck "(a) stale MINICPM_DONE (2026-11-15) + running queue_minicpm_r2.sh: keeps waiting, does not start" done_starts "FAILED steps: wait"
  ck "(a) no smoke/full/model step ran" test ! -e "$W/mpr-ivl/steps_run.txt"
  mkws mc_b stub; echo OK > "$W/MINICPM_DONE"
  fake_proc queue_minicpm_r2.sh 8 & pid=$!; sleep 0.5
  local t0=$SECONDS; q; local took=$((SECONDS - t0)); wait $pid 2>/dev/null
  ck "(b) fresh marker but queue_minicpm_r2.sh still running: said it waits" has "$(ST)" 'still running'
  ck "(b) waited for the process (>= 6 s) and then ran" bash -c "[ $took -ge 6 ]"
  ck "(b) finished OK after it ended" done_is OK
  mkws mc_c stub; echo OK > "$W/MINICPM_DONE"; q
  ck "(c) fresh marker, no process, GPU idle: proceeds" done_is OK
  ck "(c) did not announce waiting for another queue" bash -c "! grep -q 'still running' $(ST)"
  mkws mc_d stub; echo OK > "$W/MINICPM_DONE"; FAKE_GPU_USED=20000 GPU_POLL_S=0 q
  ck "(d) fresh marker, no process, GPU busy: STOP (gpu)" done_starts "FAILED steps: gpu"
  # default pattern: sample command lines
  local rx; rx=$(sed -n 's/^OTHERS="${OTHERS_RE:-\(.*\)}".*/\1/p' "$M/queue_internvl25_rel6.sh")
  ck "default OTHERS pattern read from the script: $rx" test -n "$rx"
  for c in "bash /workspace/queue_minicpm_r2.sh" "bash queue_minicpm.sh" "bash /workspace/queue_6x6.sh" "bash run_all_overnight.sh" "sh queue_sizes_uitars.sh" "bash queue_internvl25_expA.sh" "bash queue_whatever_v3.sh"; do
    ck "default pattern matches: $c" bash -c "echo '$c' | grep -q -E '^(/bin/)?(ba)?sh .*($rx)'"
  done
  ck "default pattern ignores a tmux server line" bash -c "! echo 'tmux new-session -d bash queue_6x6.sh' | grep -q -E '^(/bin/)?(ba)?sh .*($rx)'"
  # this script itself, also renamed with a suffix, is never counted
  export OTHERS_RE='queue_internvl25_rel[A-Za-z0-9_]*\.sh|queue_fakeother_[a-z0-9_]+\.sh'
  mkws self stub; echo OK > "$W/MINICPM_DONE"; cp "$W/queue_internvl25_rel6.sh" "$W/queue_internvl25_rel6_v2.sh"
  ( cd "$W" && W="$W" bash queue_internvl25_rel6_v2.sh ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"
  ck "renamed copy queue_internvl25_rel6_v2.sh does not wait for itself (pattern matches its own name)" done_is OK
  mkws self2 stub; echo OK > "$W/MINICPM_DONE"; fake_proc queue_internvl25_rel6_other.sh 8 & pid=$!; sleep 0.5; q; wait $pid 2>/dev/null
  ck "a second copy named queue_internvl25_rel6_other.sh is also excluded by the suffix rule" done_is OK
  mkws self3 stub; echo OK > "$W/MINICPM_DONE"; fake_proc queue_fakeother_x.sh 8 & pid=$!; sleep 0.5; q; wait $pid 2>/dev/null
  ck "control: a different queue name matching the pattern does block" has "$(ST)" 'still running'
  export OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh'
  unset WAIT_AFTER; export WAIT_AFTER=2000-01-01T00:00:00Z
}
t_others() {
  echo "### others: marker present but another queue still runs -> keep waiting"
  for p in queue_6x6.sh run_all_overnight.sh; do
    mkws others stub; echo OK > "$W/MINICPM_DONE"
    fake_proc $p 8 & local pid=$!; sleep 0.5
    local t0=$SECONDS; q; local took=$((SECONDS - t0)); wait $pid 2>/dev/null
    ck "$p: waited for it" has "$(ST)" 'still running'
    ck "$p: ran after it ended" done_is OK
    ck "$p: waited at least 6 s" test "$took" -ge 6
  done
  mkws others2 stub; echo OK > "$W/MINICPM_DONE"; FAKE_GPU_USED=20000 GPU_POLL_S=0 q
  ck "GPU busy -> FAILED steps: GPU" done_starts "FAILED steps: gpu"
  mkws others3 stub; echo OK > "$W/MINICPM_DONE"; ( exec -a "tmux new-session -d bash queue_6x6.sh" sleep 6 ) & local tp=$!; sleep 0.5
  q; kill $tp 2>/dev/null; wait $tp 2>/dev/null
  ck "a tmux-style process that merely contains a queue script name does not block (r2)" done_is OK
}
t_net() {
  echo "### net: slow -> STOP; FORCE_SLOW_NET=1 continues; HTTP 404 -> warning only; unreachable -> STOP"
  mkws net stub; echo OK > "$W/MINICPM_DONE"; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow: FAILED steps: network" done_starts "FAILED steps: network"
  ck "MB/s in status" has "$(ST)" 'huggingface.co 0.6 MB/s'
  ck "nothing downloaded or set up" test ! -d "$W/mpr-ivl/.cache"
  mkws net2 stub; echo OK > "$W/MINICPM_DONE"; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws net3 stub; echo OK > "$W/MINICPM_DONE"; FAKE_NET_HF="0 1.0 404" q
  ck "404: warning, continues" done_is OK
  mkws net4 stub; echo OK > "$W/MINICPM_DONE"; FAKE_NET_HF="0 1.0 000" q
  ck "unreachable: STOP" done_starts "FAILED steps: network"
}
t_fail() {
  echo "### fail: retries, failed smoke/full, timeout, ETA exit 3, cache rule"
  mkws fail1 stub; echo OK > "$W/MINICPM_DONE"; FAIL_DL_FIRST=2 q
  ck "2 transient download failures, then ok (retry x3)" done_is OK
  mkws fail2 stub; echo OK > "$W/MINICPM_DONE"; FAIL_DL_FIRST=3 q
  ck "3 failures: model step FAILED" done_starts "FAILED steps: model"
  ck "partial cache kept" test -d "$W/mpr-ivl/.cache/huggingface"
  ck "smoke/full not run after model failure" test ! -e "$W/mpr-ivl/steps_run.txt"
  mkws fail3 stub; echo OK > "$W/MINICPM_DONE"; FAKE_FAIL_STEP=full_rel6 q
  ck "FAILED steps: full_rel6" done_is "FAILED steps: full_rel6"
  ck "cache kept after failure" test -d "$W/mpr-ivl/.cache/huggingface"
  ck "status says kept" has "$(ST)" 'model cache KEPT'
  mkws fail4 stub; echo OK > "$W/MINICPM_DONE"; FAKE_FAIL_STEP=smoke_rel6 q
  ck "smoke failure: full not run" test ! -e "$W/mpr-ivl/steps_run.txt"
  ck "smoke failure reported" has "$(ST)" 'smoke failed'
  mkws fail5 stub; echo OK > "$W/MINICPM_DONE"; FAKE_FAIL_STEP=full_rel6_sleep FAKE_TIMEOUT_MATCH="full_rel6" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported" has "$(ST)" 'step full_rel6 TIMEOUT'
  mkws fail6 stub; echo OK > "$W/MINICPM_DONE"; FAKE_FAIL_STEP=full_rel6_eta q
  ck "ETA stop (exit 3) -> FAILED steps: full_rel6_eta" done_is "FAILED steps: full_rel6_eta"
  ck "ETA stop message" has "$(ST)" 'STOP step full_rel6'
  mkws fail7 stub; echo OK > "$W/MINICPM_DONE"; KEEP_MODEL_CACHE=1 q
  ck "KEEP_MODEL_CACHE=1 keeps cache on success" test -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  mkws fail8 stub; echo OK > "$W/MINICPM_DONE"; q
  ck "success: this model's cache deleted" test ! -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B"
  ck "static: model() uses retry 3 hf download" grep -q 'retry 3 hf download' "$M/queue_internvl25_rel6.sh"
}
t_setup() {
  echo "### setup: real setup() must refuse a wrong repo / config / venv (negative cases)"
  mkws s1 real; echo OK > "$W/MINICPM_DONE"; sed -i '' 's/INTERNVL_FAMILIES/INTERNVL_FAMILIEZ/' "$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/run_cross_locale.py" 2>/dev/null
  q; ck "repo without InternVL patch marker: setup FAILED" done_starts "FAILED steps: setup"
  mkws s2 real; echo OK > "$W/MINICPM_DONE"; cp "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml" "$W/y"; rm "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml"; sed 's/max_num: 7/max_num: 8/' "$W/y" > "$W/mpr-ivl/01_code/configs/models/internvl2_5_8b.yaml"
  q; ck "repo config differs from the addon config: setup FAILED" done_starts "FAILED steps: setup"
  mkws s3 real; echo OK > "$W/MINICPM_DONE"; EXPECT_TORCH=9.9.9 q
  ck "wrong torch version: venv check fails (would pip install; stubbed out here -> setup FAILED)" done_starts "FAILED steps: setup"
  mkws s4 real; echo OK > "$W/MINICPM_DONE"; rm -rf "$W/mpr-ivl"; q
  ck "no mpr-ivl: STOP before any step" done_starts "FAILED steps: repo"
  mkws s5 real; echo OK > "$W/MINICPM_DONE"; sed -i '' 's/^ADDON_SHA=.*/ADDON_SHA=0000/' "$W/queue_internvl25_rel6.sh"; q
  ck "addon sha mismatch: STOP" done_starts "FAILED steps: addon"
  mkws s6 real; echo OK > "$W/MINICPM_DONE"; rm -rf "$W/mpr-ivl/.cache/huggingface/hub"; FAKE_FREE=10 q
  ck "10 GB free and no cache: STOP (need 30)" done_starts "FAILED steps: disk"
  mkws s7 real; echo OK > "$W/MINICPM_DONE"; truncate -s 100 "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463/model-00004-of-00004.safetensors"; q
  ck "wrong safetensors size: model step FAILED" done_starts "FAILED steps: model"
  mkws s8 real; echo OK > "$W/MINICPM_DONE"; rm -rf "$W/mpr-ivl/followup_rel_4lang/assets/images/1/ru"; q
  ck "a 4-language image missing: setup does not silently continue (verify/fetch fails offline)" done_starts "FAILED steps: setup"
  mkws s9 real; echo OK > "$W/MINICPM_DONE"; q   # mkws without tar -> hard-linked images; then corrupt nothing; control: ok
  ck "control: images pre-placed (no tar) -> OK" done_is OK
  ck "control: status says images will be downloaded when no tar is present" has "$(ST)" 'no internvl25_rel6_frja_images'
}
t_imgtar() {
  echo "### imgtar: bad image tar / addon with a path outside its own files"
  mkws it1 real tar; echo OK > "$W/MINICPM_DONE"; sed -i '' 's/^IMAGES_SHA=.*/IMAGES_SHA=0000/' "$W/queue_internvl25_rel6.sh"; q
  ck "image tar sha mismatch: STOP" done_starts "FAILED steps: addon"
  mkws it2 real; echo OK > "$W/MINICPM_DONE"
  # malicious addon: same content plus one overwrite of an existing 4-language module
  rm -rf "$S/ivq6/evil"; mkdir -p "$S/ivq6/evil"; tar -xzf "$M/internvl25_rel6_addon_20261116.tar.gz" -C "$S/ivq6/evil"
  mkdir -p "$S/ivq6/evil/01_code/src/mpr_crosslocale/inference"; echo "# overwritten" > "$S/ivq6/evil/01_code/src/mpr_crosslocale/inference/run_rel_followup.py"
  rm -f "$W/internvl25_rel6_addon_20261116.tar.gz"; tar -czf "$W/internvl25_rel6_addon_20261116.tar.gz" -C "$S/ivq6/evil" 01_code followup_rel_6lang
  sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$(shasum -a 256 "$W/internvl25_rel6_addon_20261116.tar.gz" | cut -d' ' -f1)/" "$W/queue_internvl25_rel6.sh"; q
  ck "addon with a file outside its own paths: setup FAILED" done_starts "FAILED steps: setup"
  ck "the existing 4-language runner was NOT overwritten" bash -c "cd $W/mpr-ivl && shasum -a 256 -c $W/orig_sha_before.txt"
  mkws it3 real tar; echo OK > "$W/MINICPM_DONE"
  rm -rf "$S/ivq6/evil2"; mkdir -p "$S/ivq6/evil2/followup_rel_6lang/assets/images/1/en"; echo x > "$S/ivq6/evil2/followup_rel_6lang/assets/images/1/en/x.png"
  rm -f "$W/internvl25_rel6_frja_images_20261116.tar"; tar -cf "$W/internvl25_rel6_frja_images_20261116.tar" -C "$S/ivq6/evil2" followup_rel_6lang
  sed -i '' "s/^IMAGES_SHA=.*/IMAGES_SHA=$(shasum -a 256 "$W/internvl25_rel6_frja_images_20261116.tar" | cut -d' ' -f1)/" "$W/queue_internvl25_rel6.sh"; q
  ck "image tar with a non fr/ja path: setup FAILED" done_starts "FAILED steps: setup"
}
t_sigint() {
  echo "### sigint: TERM while waiting -> no QUEUE8_DONE"
  mkws sigint stub
  ( cd "$W" && W="$W" bash queue_internvl25_rel6.sh > "$W/out.txt" 2>&1 ) & local p=$!
  sleep 3; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; wait $p 2>/dev/null; sleep 1
  ck "no QUEUE8_DONE" test ! -e "$W/QUEUE8_DONE"
  ck "status says interrupted" has "$(ST)" 'interrupted by signal'
}
t_dup() {
  echo "### dup: second copy refuses (flock)"
  mkws dup stub
  ( cd "$W" && W="$W" bash queue_internvl25_rel6.sh > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 3; q
  ck "second run says already running" has "$W/out.txt" 'already running'
  kill -TERM $p 2>/dev/null; wait $p 2>/dev/null
}
t_plan() {
  echo "### plan: --plan prints and changes nothing"
  mkws plan stub; q --plan
  ck "plan mentions MINICPM_DONE and WAIT_AFTER" has "$W/out.txt" 'MINICPM_DONE with mtime later than'
  ck "plan mentions the image tar or the HF fetch" has "$W/out.txt" 'frja_images'
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE8_STATUS ] && [ ! -e $W/QUEUE8_DONE ] && [ ! -d $W/queue_results ]"
}
t_stop7() {
  echo "### stop7: the expA queue STOPped early (no MINICPM_DONE, not running) -> continue after 2 checks; a waiting line does not count"
  echo "  (WAIT_PROC is set to a fake process name: stale fake expA queue processes of 2026-11-14 still run on this Mac)"
  export WAIT_PROC='queue_fakeexpA\.sh' WAIT_STATUS_ENABLE=1
  mkws stop7 stub; echo "2026-11-15T00:00:00Z STOP: network too slow" > "$W/QUEUE7_STATUS"; WAIT_STATUS="$W/QUEUE7_STATUS" MAX_WAIT_S=30 q
  ck "proceeded after early STOP" done_is OK
  ck "status says continuing without MINICPM_DONE" has "$(ST)" 'continuing without'
  mkws stop7b stub; echo "2026-11-15T00:00:00Z ivl25_expA: still waiting for QUEUE6_DONE" > "$W/QUEUE7_STATUS"; WAIT_STATUS="$W/QUEUE7_STATUS" MAX_WAIT_S=4 q
  ck "waiting line does not release" done_starts "FAILED steps: wait"
  mkws stop7c stub; echo "2026-11-15T00:00:00Z STOP: x" > "$W/QUEUE7_STATUS"; fake_proc queue_fakeexpA.sh 6 & local pid=$!; sleep 0.5; WAIT_STATUS="$W/QUEUE7_STATUS" MAX_WAIT_S=4 q; wait $pid 2>/dev/null
  ck "a running expA-type queue blocks release" done_starts "FAILED steps: wait"
  mkws stop7d stub; echo "2026-11-15T00:00:00Z STOP: x" > "$W/QUEUE7_STATUS"; MAX_WAIT_S=4 q
  ck "no WAIT_STATUS given: the early-stop fallback is off" done_starts "FAILED steps: wait"
  unset WAIT_PROC
}
case "${1:-}" in
  all) for t in plan real subset wait minicpm stop7 others net fail setup imgtar sigint dup; do "t_$t"; done ;;
  real|subset|wait|minicpm|stop7|others|net|fail|setup|imgtar|sigint|dup|plan) "t_$1" ;;
  *) echo "usage: $0 <plan|real|subset|wait|minicpm|stop7|others|net|fail|setup|imgtar|sigint|dup|all>"; exit 2 ;;
esac
echo "failed checks: $BAD"; exit $((BAD > 0))
