#!/usr/bin/env bash
# Local flow tests for queue_6x6.sh in a fake /workspace with stub hf/nvidia-smi/curl/timeout/df (Q6_TEST=1: mock model, stub venv).
# usage: bash run_6x6_tests.sh <test-name>    (fresh | existing | net | wait | fail | rerun | cache | retry | eta | waitany | stopnet)
# The last five have PASS/FAIL checks (added 2026-11-14): cache rule, download retries, ETA stop, WAIT_FOR with any content, network STOP.
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; QS="$(dirname "$HERE")"
export PATH="$HERE/shims:$PATH"
export Q6_TEST=1 Q6_TEST_PY="$HERE/test_python.sh" EXPECT_TORCH=2.9.1 REPO_URL=$S/qwen_sizes BRANCH=followup-rel-4lang-inference
export LOCAL_DATA_SRC=$S/sx/sx_qwen/01_code/data/raw/mpr_gui_bench DL_RETRY_SLEEP=1 POLL_S=1 IGNORE_OTHER_QUEUES=1
export FAKE_STATE=$S/sx/fake_state
BAD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {
  mkdir -p "$FAKE_STATE"; rm -f "$FAKE_STATE"/*
  export W=$S/sx/ws_$1; rm -rf "$W"; mkdir -p "$W"; cp "$QS/queue_6x6.sh" "$QS/sixbysix_addon_20261114.tar.gz" "$W/"; }
q() { ( cd "$W" && bash queue_6x6.sh "$@" ); }
existing_repos() {  # imitate an instance after the 8B runs / earlier queues
  git clone -q --branch followup-rel-4lang-inference $S/qwen_sizes "$W/mpr-gui-crosslocale" && git -C "$W/mpr-gui-crosslocale" checkout -q 666440011f538cd9336794c9965429de640843c3
  mkdir -p "$W/mpr-gui-crosslocale/01_code/data/raw" && cp -R "$LOCAL_DATA_SRC" "$W/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench"
  mkdir -p "$W/mpr-gui-crosslocale/01_code/data/manifests" && cp "$S/sx/sx_qwen/01_code/data/manifests/mpr_gui_manifest.jsonl" "$W/mpr-gui-crosslocale/01_code/data/manifests/"
  git clone -q $S/ivl_sizes "$W/mpr-ivl" && git -C "$W/mpr-ivl" checkout -q 2406d7e
  git clone -q $S/mc/repo "$W/mpr-minicpm" && git -C "$W/mpr-minicpm" checkout -q f4f0d97
}
case "$1" in
  fresh)     mkws fresh;    q --plan | head -20; MODELS="qwen3_vl_4b internvl35_4b minicpm_v45" q ;;
  existing)  mkws existing; existing_repos; q ;;
  net)       mkws net; FAKE_NET_HF="6000000 15.0 206" MODELS="qwen3_vl_2b" q; echo "exit=$? done_file=$(ls $W/QUEUE6_DONE 2>&1 | tail -1)";
             FAKE_NET_HF="6000000 15.0 206" FORCE_SLOW_NET=1 FAKE_FULL_LIMIT=20 MODELS="internvl35_1b" q; FAKE_NET_PT="1000000 15.0 206" q --net ;;
  fail)      mkws fail; existing_repos
             echo "### run 1: 4b download fails, full step limited to 7 s -> TIMEOUT, _PARTIAL"; FAIL_DL=OpenGVLab/InternVL3_5-4B FAKE_FULL_LIMIT=7 MODELS="qwen3_vl_4b internvl35_4b qwen3_vl_2b" q
             echo "### run 2: low disk skips uitars; qwen3_vl_2b resumes after the timeout (still limited)"; FAKE_FREE=30 FAKE_FULL_LIMIT=7 MODELS="uitars_15_7b qwen3_vl_2b" q
             echo "### run 3: no limit; unfinished models resume and complete, internvl35_4b downloads again"; MODELS="qwen3_vl_4b internvl35_4b uitars_15_7b qwen3_vl_2b" q ;;
  wait)      mkws wait;     existing_repos; ( sleep 4; touch "$W/MINICPM_DONE" ) & WAIT_FOR="$W/MINICPM_DONE" FAKE_FULL_LIMIT=20 MODELS="qwen3_vl_2b" q; wait ;;
  cache)     mkws cache; existing_repos
             echo "### cache rule: partial download kept, timeout keeps cache, low disk deletes it, success deletes it"
             FAIL_DL=OpenGVLab/InternVL3_5-4B FAKE_FULL_LIMIT=7 MODELS="qwen3_vl_4b internvl35_4b" q
             ST=$W/QUEUE6_STATUS
             ck "full-run TIMEOUT reported" has $ST "full=TIMEOUT"
             ck "Qwen3-4B cache kept after timeout (80 GB free)" test -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct
             ck "status says KEPT" has $ST "qwen3_vl_4b model cache KEPT"
             ck "InternVL download failed" has $ST "internvl35_4b DOWNLOAD FAILED"
             ck "partial InternVL download kept" test -e $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B/blobs/part.incomplete
             ck "model not COMPLETE" test ! -f $W/queue_results/6x6_qwen3_vl_4b/COMPLETE
             echo "  -- run 2 (no limit): resumes from the kept cache without downloading, completes, then deletes the cache"
             MODELS="qwen3_vl_4b" q
             ck "qwen3_vl_4b COMPLETE using the kept cache" test -f $W/queue_results/6x6_qwen3_vl_4b/COMPLETE
             ck "no second transfer of Qwen3-4B (kept cache reused)" test "$(cat $FAKE_STATE/transfers)" = 1
             ck "cache deleted after success" test ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct
             echo "  -- low disk: 30 GB free < 52 GB needed by the next model (UI-TARS) -> cache of the unfinished model deleted"
             FAKE_FREE=30 FAKE_FULL_LIMIT=7 MODELS="qwen3_vl_2b uitars_15_7b" q
             ck "cache deleted for disk" has $ST "qwen3_vl_2b model cache deleted (not finished; free 30 GB < 52 GB"
             ck "UI-TARS skipped for disk" has $ST "uitars_15_7b SKIPPED: only 30 GB free"
             echo "failed checks: $BAD" ;;
  retry)     mkws retry; existing_repos
             echo "### retry: two transient hf failures, then success"
             FAIL_DL_FIRST=2 FAKE_FULL_LIMIT=20 MODELS="qwen3_vl_2b" q
             ck "download succeeded after 2 failures" has $W/QUEUE6_STATUS "qwen3_vl_2b downloaded"
             ck "retry messages in the download log" has $W/queue_logs6/qwen3_vl_2b_download.log "retry 1/3 failed"
             ck "static: git clone retried" grep -qF 'retry 5 clone_repo' "$QS/queue_6x6.sh"
             ck "static: git fetch retried" grep -qF 'retry 3 git fetch' "$QS/queue_6x6.sh"
             ck "static: pip install retried" grep -qF 'retry 3 python -m pip install -r' "$QS/queue_6x6.sh"
             ck "static: hf download retried" grep -qF 'retry 3 timeout "$T_DOWNLOAD" hf download' "$QS/queue_6x6.sh"
             echo "failed checks: $BAD" ;;
  eta)       mkws eta; existing_repos
             echo "### eta: smoke gate says 99,999,999 s for the full run -> full run not started, STOPPED_ETA, model partial"
             OUT=$W/mpr-gui-crosslocale/followup_rel_4lang/results/6x6_qwen3_vl_2b; mkdir -p $OUT
             echo '{"status": "passed", "eta_full_seconds": 99999999}' > $OUT/gate_6x6_smoke.json
             MODELS="qwen3_vl_2b" q
             ck "STOPPED line in QUEUE6_STATUS" has $W/QUEUE6_STATUS "full 6x6 STOPPED, not started: ETA"
             ck "full=STOPPED_ETA in the model line" has $W/QUEUE6_STATUS "full=STOPPED_ETA"
             ck "no full run output" test ! -s $OUT/full_6x6.jsonl
             ck "package is _PARTIAL" bash -c "ls $W/queue_results/6x6_qwen3_vl_2b/*_PARTIAL.tar.gz"
             rm -rf $OUT; mkdir -p $OUT; echo '{"status": "passed", "eta_full_seconds": 99999999}' > $OUT/gate_6x6_smoke.json
             IGNORE_ETA=1 FAKE_FULL_LIMIT=15 MODELS="qwen3_vl_2b" q
             ck "IGNORE_ETA=1 starts the full run" has $W/QUEUE6_STATUS "qwen3_vl_2b full 6x6 started"
             echo "failed checks: $BAD" ;;
  waitany)   mkws waitany; existing_repos
             echo "### WAIT_FOR accepts any content (and keeps waiting while run_all_overnight.sh runs)"
             ( sleep 3; echo "FAILED steps: full_rel" > "$W/MINICPM_DONE" ) & WAIT_FOR="$W/MINICPM_DONE" FAKE_FULL_LIMIT=15 MODELS="qwen3_vl_2b" q; wait
             ck "started after a MINICPM_DONE with FAILED content" has $W/QUEUE6_STATUS "found: FAILED steps: full_rel"
             ck "model ran" has $W/QUEUE6_STATUS "qwen3_vl_2b start"
             mkws waitany2; existing_repos; echo "STOPPED rc=1" > "$W/QUEUE_DONE"
             mkdir -p $S/sx/fakeproc; printf '#!/bin/bash\nsleep 8\n' > $S/sx/fakeproc/run_all_overnight.sh; bash $S/sx/fakeproc/run_all_overnight.sh & sleep 0.5
             t0=$SECONDS; WAIT_FOR="$W/QUEUE_DONE" FAKE_FULL_LIMIT=15 MODELS="qwen3_vl_2b" q; took=$((SECONDS - t0)); wait
             ck "waited while run_all_overnight.sh ran" has $W/QUEUE6_STATUS "run_all_overnight.sh is still running"
             ck "took at least 6 s" test $took -ge 6
             echo "failed checks: $BAD" ;;
  stopnet)   mkws stopnet; existing_repos
             echo "### network: slow -> STOP with MB/s in the status; 404 test URL -> warning only"
             FAKE_NET_HF="6000000 10.0 206" MODELS="qwen3_vl_2b" q
             ck "STOP line" has $W/QUEUE6_STATUS "STOP: network too slow"
             ck "MB/s logged" has $W/QUEUE6_STATUS "huggingface.co 0.6 MB/s"
             ck "no QUEUE6_DONE (STOP exits before the end)" test ! -e $W/QUEUE6_DONE
             FAKE_NET_HF="0 1.0 404" FAKE_FULL_LIMIT=15 MODELS="qwen3_vl_2b" q
             ck "404: warning, continues" has $W/QUEUE6_STATUS "answered HTTP 404"
             echo "failed checks: $BAD" ;;
  *) echo "unknown test"; exit 2 ;;
esac
