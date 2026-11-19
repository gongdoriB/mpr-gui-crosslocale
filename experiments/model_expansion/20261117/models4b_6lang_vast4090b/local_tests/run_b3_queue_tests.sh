#!/usr/bin/env bash
# Flow tests for queue_b3_4b.sh in a fake /workspace. No GPU, no network (proxy -> closed port), no pip, no model download, no server.
# real  = real setup() (addon copy-if-absent next to the files the B2 addon already put there, SHA256SUMS, env_check against the earlier 4B run contracts, data/image verify), real model() (offline fake HF caches of the true
#         byte size, stand-in tokenizers), real run scripts with --mock-model, real gates, real packaging, and the REAL queue_6x6.sh (Q6_TEST=1: mock model) with the real 6x6 data + manifest of the local copy (79,020 inputs).
# stub  = setup/model/smoke/full/retry/six_* replaced by stubs (make_stub_b3.py): wait, guards, failure rule, order, cache rule, packaging.
# usage: run_b3_queue_tests.sh plan|stub|wait|guards|counts|real|resume|setup|six|all
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
ROOT="${PROJECT_ROOT}"
WS=$S/b3/qtest
B2ADDON="$ROOT/model_expansion_20261116/vast4090b_frja/b2_frja_addon_20261116.tar.gz"
FRJA="$ROOT/followup_rel_6lang/assets"
RAWSRC=$S/sx/sx_qwen/01_code/data/raw/mpr_gui_bench; MANSRC=$S/sx/sx_qwen/01_code/data/manifests
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 PIP_INDEX_URL=http://127.0.0.1:9/simple
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 EXPECT_SKIP=qwen-vl-utils,torchvision,protobuf,sentencepiece GPU_RECHECK_TRIES=2 GPU_RECHECK_S=1 POLL_S=1
export OTHERS_RE='run_all_overnight\.sh|queue_b2_[a-z_]*\.sh|queue_other_test\.sh'
export Q6_TEST=1 Q6_TEST_PY="$HERE/testpy.sh" LOCAL_DATA_SRC="$RAWSRC" DL_RETRY_SLEEP=1 DATA_RETRY_SLEEP=1
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mk_repo() {  # $1 = target, $2 = source repo (hard-link copy; .git really copied so tests never touch the source), $3 = family
  cp -Rl "$2" "$1"; rm -rf "${1:?}/.git" "${1:?}/followup_rel_4lang/results"; cp -R "$2/.git" "$1/.git"
  rm -rf "${1:?}/.venv" "${1:?}/.cache"; mkdir -p "$1/.venv/bin" "$1/.cache/huggingface"
  printf '#!/bin/bash\nTESTPY_REPO="%s" exec "%s/testpy.sh" "$@"\n' "$1" "$HERE" > "$1/.venv/bin/python"; chmod +x "$1/.venv/bin/python"; ln -s python "$1/.venv/bin/python3"
  printf 'export PATH="%s/.venv/bin:$PATH"\n' "$1" > "$1/.venv/bin/activate"
}
fake_hf() { [ -d "$S/b3/hf_fakeQ" ] && [ -d "$S/b3/hf_fakeI" ] || python3 "$HERE/make_fake_hf_b3.py" "$S/b3/hf_fakeQ" "$S/b3/hf_fakeI" >/dev/null; }
mkws() {  # $1 name  $2 = real|stub|real-nodata
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state
  rm -rf "${WS:?}/ws_$1"; mkdir -p "$W" "$FAKE_STATE"; echo OK > "$W/QUEUE_B2_DONE"
  ln "$M/upload/b3_4b_addon_20261117.tar.gz" "$M/upload/sixbysix_addon_20261114.tar.gz" "$M/upload/queue_6x6.sh" "$W/" 2>/dev/null || cp "$M/upload/b3_4b_addon_20261117.tar.gz" "$M/upload/sixbysix_addon_20261114.tar.gz" "$M/upload/queue_6x6.sh" "$W/"
  if [ "$2" = stub ]; then
    python3 "$HERE/make_stub_b3.py" "$M/queue_b3_4b.sh" "$W/queue_b3_4b.sh"
    for r in mpr-ivl mpr-gui-crosslocale; do mkdir -p "$W/$r/.git" "$W/$r/.venv/bin" "$W/$r/01_code"; touch "$W/$r/.venv/bin/activate"; done
  else
    cp "$M/queue_b3_4b.sh" "$W/"; fake_hf
    mk_repo "$W/mpr-gui-crosslocale" "$S/qwen_sizes"; mk_repo "$W/mpr-ivl" "$S/ivl_sizes"
    cp -Rl "$S/b3/hf_fakeQ/hub" "$W/mpr-gui-crosslocale/.cache/huggingface/hub"; cp -Rl "$S/b3/hf_fakeI/hub" "$W/mpr-ivl/.cache/huggingface/hub"
    # server state after the B2 queue: its addon unpacked in both repos, fr/ja images in mpr-ivl and in mpr-gui-crosslocale
    for r in mpr-ivl mpr-gui-crosslocale; do ( cd "$W/$r" && tar -xzf "$B2ADDON" && cp -Rl "$FRJA" followup_rel_6lang/assets ); done
    if [ "$2" = real ]; then   # 6x6 raw data + manifest of the local copy sit in the Qwen repo (as after the earlier 6x6 queue); the InternVL repo gets the link from queue_6x6.sh
      mkdir -p "$W/mpr-gui-crosslocale/01_code/data/raw" "$W/mpr-gui-crosslocale/01_code/data/manifests"
      cp -Rl "$RAWSRC" "$W/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench"; cp "$MANSRC/mpr_gui_manifest.jsonl" "$W/mpr-gui-crosslocale/01_code/data/manifests/"
    fi
    for r in mpr-ivl mpr-gui-crosslocale; do ( cd "$W/$r" && find followup_rel_4lang -type f | sort > "$W/files4_$r.txt"; shasum -a 256 01_code/src/mpr_crosslocale/inference/run_rel_followup.py 01_code/src/mpr_crosslocale/data/rel_followup.py > "$W/orig_sha_$r.txt"
         find followup_rel_6lang -type f | sort > "$W/files6_before_$r.txt"; xargs shasum -a 256 < "$W/files6_before_$r.txt" > "$W/b2addon_sha_$r.txt" ); done
  fi
}
q() { ( cd "$W" && W="$W" bash queue_b3_4b.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_B3_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_B3_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
RES() { echo "$W/queue_results/b3_4b"; }
ST() { echo "$W/QUEUE_B3_STATUS"; }
FULLSTEPS="setup qwen3_vl_4b,model qwen3_vl_4b,smoke_rel qwen3_vl_4b,smoke_expa qwen3_vl_4b,full_rel qwen3_vl_4b,full_expa qwen3_vl_4b,retry_expa qwen3_vl_4b,setup internvl35_4b,model internvl35_4b,smoke_rel internvl35_4b,smoke_expa internvl35_4b,full_rel internvl35_4b,full_expa internvl35_4b,retry_expa internvl35_4b,six_setup qwen3_vl_4b,six_run qwen3_vl_4b,six_setup internvl35_4b,six_run internvl35_4b,"

t_plan() { echo "### plan"; mkws plan stub; q --plan
  ck "plan: order, REL 26,057, Exp A 8,532 / 444, 79,020, strict true, wait rule" bash -c "grep -q 'qwen3_vl_4b internvl35_4b' $W/out.txt && grep -q '26,057' $W/out.txt && grep -q '8,532' $W/out.txt && grep -q '444' $W/out.txt && grep -q '79,020' $W/out.txt && grep -q 'strict true' $W/out.txt && grep -q 'QUEUE_B2_DONE exists' $W/out.txt"
  ck "plan labels estimates and states disk budget and failure rule" bash -c "grep -q 'estimates from the medians' $W/out.txt && grep -q 'Disk (34-50 GB' $W/out.txt && grep -q 'Failure rule' $W/out.txt"
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE_B3_STATUS ] && [ ! -e $W/QUEUE_B3_DONE ] && [ ! -d $W/queue_results ]"
  mkws plan2 stub; ONLY="internvl35_4b qwen3_vl_4b" q --plan; ck "ONLY keeps the fixed order" has "$W/out.txt" 'Models in this order: qwen3_vl_4b internvl35_4b'
  ck "bash -n on every script" bash -c "cd \"$M\" && bash -n queue_b3_4b.sh && bash -n upload/queue_b3_4b.sh && for f in addon/followup_rel_6lang/b3_4b/*.sh build_addon.sh local_tests/run_b3_queue_tests.sh; do bash -n \$f || exit 1; done"
  ck "python files compile" bash -c "cd \"$M/addon/followup_rel_6lang\" && python3 -m py_compile b3_4b/*.py b3_4b/expA6/*.py run_inference_6lang.py data_bundle_6lang.py && find . -name __pycache__ -prune -exec rm -rf {} +"
  ck "upload copy of the queue equals the script and the addon sha in it matches the tar" bash -c "cmp \"$M/queue_b3_4b.sh\" \"$M/upload/queue_b3_4b.sh\" && grep -q \"^ADDON_SHA=\$(shasum -a 256 \"$M/upload/b3_4b_addon_20261117.tar.gz\" | cut -d' ' -f1)\" \"$M/queue_b3_4b.sh\""; }

t_stub() { echo "### stub flow, failure rule"
  mkws a stub; q; ck "all succeed: DONE OK, 17 steps in the planned order" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ] && [ \"\$(tr '\n' ',' < $W/steps_run.txt)\" = '$FULLSTEPS' ]"
  ck "per model: 01_rel, 02_expA, 03_6x6 packages with sha256 + per-file lists; COMPLETE; final package" bash -c "for k in qwen3_vl_4b internvl35_4b; do ls $(RES)/\$k/b3_\${k}_01_rel_*.tar.gz $(RES)/\$k/b3_\${k}_02_expA_*.tar.gz $(RES)/\$k/*6x6*.tar.gz >/dev/null && (cd $(RES)/\$k && shasum -a 256 -c b3_\${k}_01_rel_*.tar.gz.sha256 b3_\${k}_02_expA_*.tar.gz.sha256) && ls $(RES)/\$k/b3_\${k}_03_6x6_*.files.sha256 && [ -f $(RES)/\$k/COMPLETE ] || exit 1; done; ls $(RES)/b3_4b_00_final_*.tar.gz && [ -s $(RES)/PACKAGES_SHA256.txt ]"
  ck "final status lists both models with a+b ok and 6x6 ok" has "$(ST)" 'models: qwen3_vl_4b=ab_ok 6x6_ok internvl35_4b=ab_ok 6x6_ok'
  ck "status lines name 6x6 as the unchanged queue_6x6.sh step" has "$(ST)" 'step 6x6_internvl35_4b ok'
  q; ck "rerun: parts a+b of COMPLETE models skipped, 6x6 complete skipped, DONE OK" bash -c "grep -c 'parts a+b already complete' $W/out.txt | grep -q 2 && grep -c '6x6 already complete' $W/out.txt | grep -q 2 && [ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ]"
  mkws k stub; KEEP_MODEL_CACHE=1 SKIP_6X6=1 q; ck "SKIP_6X6=1: no six steps, caches kept with KEEP_MODEL_CACHE=1" bash -c "! grep -q six_ $W/steps_run.txt && [ -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct ]"
  mkws k2 stub; SKIP_6X6=1 q; ck "SKIP_6X6=1 without KEEP: caches deleted after parts a+b" bash -c "[ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct ] && [ ! -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B ] && done=\$(cat $W/QUEUE_B3_DONE) && [ \"\$done\" = OK ]"
  mkws k3 stub; ONLY_6X6=1 q; ck "ONLY_6X6=1: only six steps" bash -c "[ \"\$(tr '\n' ',' < $W/steps_run.txt)\" = 'six_setup qwen3_vl_4b,six_run qwen3_vl_4b,six_setup internvl35_4b,six_run internvl35_4b,' ]"
  mkws c1 stub; q; ck "model 1 cache kept for its 6x6 until its 6x6 (stub does not delete): both caches present at the end with 80 GB free" test -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct"
  mkws c2 stub; FAKE_FREE=25 q
  ck "25 GB free < 20+10: the cache of model 1 is removed before model 2's download; status says so" bash -c "grep -q 'removed the kept cache of qwen3_vl_4b to make room' $W/QUEUE_B3_STATUS && [ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct ] && done=\$(cat $W/QUEUE_B3_DONE) && [ \"\$done\" = OK ]"
  mkws c3 stub; FAKE_FREE=15 q
  ck "15 GB free (< 20): STOP disk before the first download, environmental" bash -c "grep -q 'STOP: only 15 GB free (need 20 GB for qwen3_vl_4b)' $W/QUEUE_B3_STATUS && ! grep -q 'model ' $W/steps_run.txt && grep -q 'disk_qwen3_vl_4b' $W/QUEUE_B3_DONE"
  # model-specific failures: the queue continues
  mkws f1 stub; FAKE_FAIL="full_rel:qwen3_vl_4b" q
  ck "Qwen REL full fails: Exp A still runs for it, PARTIAL REL package, no COMPLETE, InternVL runs, its 6x6 runs" bash -c "grep -q 'full_expa qwen3_vl_4b' $W/steps_run.txt && ls $(RES)/qwen3_vl_4b/*01_rel*_PARTIAL.tar.gz && [ ! -f $(RES)/qwen3_vl_4b/COMPLETE ] && grep -q 'full_expa internvl35_4b' $W/steps_run.txt && grep -q 'six_run qwen3_vl_4b' $W/steps_run.txt && [ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: full_rel_qwen3_vl_4b' ]"
  ck "status says 'continuing'" has "$(ST)" 'model-specific problem; environment re-check ok; continuing'
  ck "final line shows ab_FAILED for qwen with rel=failed" has "$(ST)" 'qwen3_vl_4b=ab_FAILED(rel=failed,expA=done)'
  mkws f2 stub; FAKE_FAIL="smoke_rel:internvl35_4b" q
  ck "InternVL REL smoke fails: REL full skipped, Exp A runs, 6x6 of both still runs" bash -c "! grep -q 'full_rel internvl35_4b' $W/steps_run.txt && grep -q 'full_expa internvl35_4b' $W/steps_run.txt && grep -q 'six_run internvl35_4b' $W/steps_run.txt && [ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: smoke_rel_internvl35_4b' ]"
  mkws f3 stub; FAKE_FAIL="model:qwen3_vl_4b" q
  ck "Qwen model step fails (download/check): that model stops (no smoke), InternVL runs; Qwen 6x6 still attempted" bash -c "! grep -q 'smoke_rel qwen3_vl_4b' $W/steps_run.txt && grep -q 'full_expa internvl35_4b' $W/steps_run.txt && grep -q 'six_run qwen3_vl_4b' $W/steps_run.txt && grep -q 'model_qwen3_vl_4b' $W/QUEUE_B3_DONE"
  mkws f4 stub; FAKE_ETA="full_rel:internvl35_4b" q
  ck "ETA stop (exit 3) on InternVL REL: model-specific, listed with _eta, queue continues" bash -c "grep -q 'full_rel_internvl35_4b_eta' $W/QUEUE_B3_DONE && grep -q 'smoke ETA longer' $W/QUEUE_B3_STATUS && grep -q 'six_run internvl35_4b' $W/steps_run.txt"
  mkws f5 stub; FAKE_SLEEP="full_rel:qwen3_vl_4b" FAKE_TIMEOUT_MATCH="full_rel" FAKE_TIMEOUT_SECS=2 q
  ck "timeout of a full step: reported, queue continues" bash -c "grep -q 'step full_rel_qwen3_vl_4b TIMEOUT' $W/QUEUE_B3_STATUS && grep -q 'retry_expa internvl35_4b' $W/steps_run.txt"
  mkws f6 stub; FAKE_FAIL="six_run:qwen3_vl_4b" q
  ck "Qwen 6x6 incomplete: model-specific, InternVL 6x6 still runs, PARTIAL 6x6 sha list, verdict names it" bash -c "grep -q 'six_run internvl35_4b' $W/steps_run.txt && ls $(RES)/qwen3_vl_4b/b3_qwen3_vl_4b_03_6x6_*_PARTIAL.files.sha256 2>/dev/null; grep -q '6x6_qwen3_vl_4b' $W/QUEUE_B3_DONE && grep -q 'qwen3_vl_4b=ab_ok 6x6_NOT_COMPLETE' $W/QUEUE_B3_STATUS"
  mkws f7 stub; FAKE_FAIL="six_setup:internvl35_4b" q
  ck "six_setup fails (repo state / data differs): no six_run for that model, DONE lists six_setup_..., qwen 6x6 done before" bash -c "grep -q 'six_run qwen3_vl_4b' $W/steps_run.txt && ! grep -q 'six_run internvl35_4b' $W/steps_run.txt && grep -q 'six_setup_internvl35_4b' $W/QUEUE_B3_DONE"
  mkws f8 stub; mkdir -p "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full" "$W/mpr-ivl/followup_rel_6lang/results/expA6_internvl35_4b/full" "$W/mpr-ivl/followup_rel_6lang/results/expA6_internvl35_4b/retry"
  echo '{"status": "passed"}' > "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full/gate_rel_complete.json"; echo '{"status": "passed"}' > "$W/mpr-ivl/followup_rel_6lang/results/expA6_internvl35_4b/full/gate_complete.json"; echo '{"status": "passed"}' > "$W/mpr-ivl/followup_rel_6lang/results/expA6_internvl35_4b/retry/gate_complete.json"; ONLY="internvl35_4b" q
  ck "a+b gates already passed on disk: no setup/model/smoke/full steps, packaging only, COMPLETE, 6x6 runs" bash -c "! grep -q -E '^(setup|model|smoke|full|retry)' $W/steps_run.txt && grep -q 'gates already passed on disk; packaging only' $W/QUEUE_B3_STATUS && [ -f $(RES)/internvl35_4b/COMPLETE ] && grep -q 'six_run internvl35_4b' $W/steps_run.txt"
  mkws f9 stub; mkdir -p "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full"; echo '{"status": "passed"}' > "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full/gate_rel_complete.json"; ONLY="internvl35_4b" SKIP_6X6=1 q
  ck "REL gate passed earlier: REL steps skipped with a status line, Exp A runs, REL package still written" bash -c "! grep -q -E '(smoke|full)_rel' $W/steps_run.txt && grep -q 'REL fr/ja already complete' $W/QUEUE_B3_STATUS && grep -q 'full_expa internvl35_4b' $W/steps_run.txt && ls $(RES)/internvl35_4b/*01_rel*.tar.gz"
  # environmental failures: the queue stops
  mkws e1 stub; FAKE_FAIL="setup:internvl35_4b" q
  ck "setup of InternVL fails (identity/venv): queue STOPS, no 6x6 for anyone, qwen had finished a+b" bash -c "grep -q 'FAILED steps: setup_internvl35_4b' $W/QUEUE_B3_DONE && ! grep -q 'six_' $W/steps_run.txt && grep -q 'STOP QUEUE (environmental failure' $W/QUEUE_B3_STATUS && [ -f $(RES)/qwen3_vl_4b/COMPLETE ]"
  mkws e2 stub; FAKE_CUDA="full_rel:qwen3_vl_4b" q
  ck "CUDA fatal error text in a step log: queue STOPS (InternVL not started), final package still written" bash -c "grep -q 'STOP QUEUE (environmental failure: CUDA fatal' $W/QUEUE_B3_STATUS && ! grep -q 'setup internvl35_4b' $W/steps_run.txt && ls $(RES)/b3_4b_00_final_*.tar.gz"
  mkws e3 stub; FAKE_FAIL="full_expa:internvl35_4b" FAKE_BUSY_AFTER=1 q
  ck "failed step leaves the GPU busy: environment re-check STOPS before any 6x6" bash -c "grep -q 'environment re-check after the failed internvl35_4b' $W/QUEUE_B3_STATUS && ! grep -q 'six_' $W/steps_run.txt && grep -q 'env_after_internvl35_4b' $W/QUEUE_B3_DONE"
  mkws e4 stub; FAKE_CUDA="six_run:qwen3_vl_4b" q
  ck "CUDA text in the 6x6 step: queue STOPS before the InternVL 6x6" bash -c "grep -q 'STOP QUEUE (environmental failure: CUDA fatal' $W/QUEUE_B3_STATUS && ! grep -q 'six_run internvl35_4b' $W/steps_run.txt"
  mkws e5 stub; rm -rf "${W:?}/mpr-gui-crosslocale/.venv"; q
  ck "Qwen repo without venv: STOP repo for qwen3_vl_4b, queue ends (environmental), nothing else started" bash -c "grep -q 'repo_qwen3_vl_4b' $W/QUEUE_B3_DONE && [ ! -e $W/steps_run.txt ]"
  mkws d1 stub; ( cd "$W" && W="$W" FAKE_SLEEP="smoke_expa:qwen3_vl_4b" bash queue_b3_4b.sh > out1.txt 2>&1 ) & local p=$!; sleep 4; q
  ck "duplicate start refused (already running), DONE untouched" bash -c "grep -q 'already running' $W/out.txt && [ \$(cat $W/rc.txt) = 1 ] && [ ! -e $W/QUEUE_B3_DONE ]"
  kill -TERM $p 2>/dev/null; pkill -TERM -P $p 2>/dev/null; wait $p 2>/dev/null
  mkws d2 stub; ( cd "$W" && W="$W" FAKE_SLEEP="smoke_expa:qwen3_vl_4b" bash queue_b3_4b.sh > out1.txt 2>&1 ) & p=$!; sleep 4; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; for _ in $(seq 1 45); do has "$W/QUEUE_B3_STATUS" 'interrupted by signal' && break; sleep 1; done; wait $p 2>/dev/null
  ck "SIGTERM: status says interrupted, no QUEUE_B3_DONE" bash -c "grep -q 'interrupted by signal' $W/QUEUE_B3_STATUS && [ ! -e $W/QUEUE_B3_DONE ]"; }

t_wait() { echo "### wait for QUEUE_B2_DONE + idle GPU"
  mkws w1 stub; rm -f "$W/QUEUE_B2_DONE"; ( sleep 4; echo OK > "$W/QUEUE_B2_DONE" ) & local t0=$SECONDS; q; local took=$((SECONDS - t0)); wait
  ck "QUEUE_B2_DONE absent: waits (>= 4 s), then runs everything, status says it waited" bash -c "[ $took -ge 4 ] && grep -q 'does not exist yet' $W/QUEUE_B3_STATUS && grep -q 'wait over after' $W/QUEUE_B3_STATUS && [ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ]"
  ck "nothing ran before the file appeared (no step line before 'wait over')" bash -c "a=\$(grep -n 'wait over' $W/QUEUE_B3_STATUS | head -1 | cut -d: -f1); b=\$(grep -n 'step setup_' $W/QUEUE_B3_STATUS | head -1 | cut -d: -f1); [ \$a -lt \$b ]"
  mkws w2 stub; touch "$FAKE_STATE/busy"; ( sleep 4; rm -f "$FAKE_STATE/busy" ) & t0=$SECONDS; q; took=$((SECONDS - t0)); wait
  ck "DONE exists but GPU busy (20,000 MiB): waits until it is free" bash -c "[ $took -ge 4 ] && grep -q 'GPU in use' $W/QUEUE_B3_STATUS && [ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ]"
  mkws w3 stub; mkdir -p "$S/b3/fp"; printf '#!/bin/bash\nsleep 6\n' > "$S/b3/fp/queue_b2_fake.sh"; bash "$S/b3/fp/queue_b2_fake.sh" & local p=$!; sleep 1; t0=$SECONDS; q; took=$((SECONDS - t0)); wait $p 2>/dev/null
  ck "DONE exists but a queue_b2 process still runs: waits for it" bash -c "[ $took -ge 4 ] && grep -q 'other queue script running' $W/QUEUE_B3_STATUS && [ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ]"
  mkws w4 stub; rm -f "$W/QUEUE_B2_DONE"; WAIT_MAX_H=0 q
  ck "DONE never appears: STOP wait after the limit, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: wait' ] && [ ! -e $W/steps_run.txt ]"
  mkws w5 stub; echo "FAILED steps: full_expa_uitars_15_7b" > "$W/QUEUE_B2_DONE"; q
  ck "B2 verdict FAILED: still starts (any content), content is logged" bash -c "grep -q \"QUEUE_B2_DONE = 'FAILED steps: full_expa_uitars_15_7b\" $W/QUEUE_B3_STATUS && [ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ]"
  mkws w6 stub; rm -f "$W/QUEUE_B2_DONE"; NO_WAIT=1 q; ck "NO_WAIT=1 starts without QUEUE_B2_DONE" done_is OK
  mkws w7 stub; rm -f "$W/QUEUE_B2_DONE"; ( sleep 5; echo OK > "$W/QUEUE_B2_DONE" ) & q --plan >/dev/null; wait; ck "--plan never waits and writes nothing" bash -c "[ ! -e $W/QUEUE_B3_STATUS ]"; }

t_guards() { echo "### guards"
  mkws g1 stub; NO_WAIT=1 FAKE_GPU_APPS="4242, python, 9000 MiB" WAIT_MAX_H=0 q; ck "NO_WAIT=1 + GPU process: refused, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: gpu' ] && [ ! -e $W/steps_run.txt ]"
  mkws g2 stub; NO_WAIT=1 FAKE_GPU_USED=20000 q; ck "NO_WAIT=1 + GPU memory used: refused" done_starts "FAILED steps: gpu"
  mkws g3 stub; FAKE_GPU_NAME="NVIDIA A100" q; ck "not a 4090: refused" done_starts "FAILED steps: gpu"
  mkws g4 stub; mkdir -p "$S/b3/fp"; printf '#!/bin/bash\nsleep 25\n' > "$S/b3/fp/queue_other_test.sh"; bash "$S/b3/fp/queue_other_test.sh" & local p=$!; sleep 1; NO_WAIT=1 q; kill $p 2>/dev/null; wait $p 2>/dev/null
  ck "NO_WAIT=1 + another queue script running: refused (others), nothing started" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: others' ] && [ ! -e $W/steps_run.txt ]"
  mkws g5 stub; FAKE_NET_HF="6000000 10.0 206" q; ck "slow network: STOP network, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: network' ] && [ ! -e $W/steps_run.txt ]"
  mkws g6 stub; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q; ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws g7 stub; FAKE_NET_HF="0 1.0 404" q; ck "HTTP 404 on the test URL: warning only" done_is OK
  mkws g8 stub; rm -f "$W/b3_4b_addon_20261117.tar.gz"; echo junk > "$W/b3_4b_addon_20261117.tar.gz"; cp "$M/queue_b3_4b.sh" "$W/queue_b3_4b.sh"; q; ck "addon sha mismatch: STOP addon, nothing started, before any waiting" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = 'FAILED steps: addon' ] && [ ! -e $W/steps_run.txt ]"
  mkws g9 stub; rm -f "$W/b3_4b_addon_20261117.tar.gz"; q; ck "addon missing: STOP addon" done_is "FAILED steps: addon"
  mkws g10 stub; rm -f "$W/queue_6x6.sh"; echo junk > "$W/queue_6x6.sh"; q; ck "queue_6x6.sh with a different sha256: STOP addon (the 6x6 queue must be the unchanged file)" done_is "FAILED steps: addon"
  mkws g11 stub; echo junk > "$W/internvl25_rel6_frja_images_20261116.tar"; q; ck "image tar with a wrong sha256: STOP addon" done_is "FAILED steps: addon"
  ck "static: no pip install, no git checkout/reset/clean, no rm -rf outside the queue's own cache/stage paths" bash -c "! sed 's/ #.*//' \"$M/queue_b3_4b.sh\" | grep -v -E '^ *#|echo ' | grep -q -E 'pip install|git (checkout|reset|clean|am)' && ! grep 'rm -rf' \"$M/queue_b3_4b.sh\" | grep -v -E 'CACHE_DIR|stage|RES|\\\$fs|\\\$fs|WS' | grep -q ."
  ck "static: nothing is written under followup_rel_4lang/results except the 6x6 result folder named by queue_6x6.sh" bash -c "cd \"$M\" && ! grep -h 'followup_rel_4lang/results' queue_b3_4b.sh addon/followup_rel_6lang/b3_4b/*.sh | grep -v -E '^#|# never|never followup|OUT_6X6|6x6_' | grep -q ."; }

t_counts() { echo "### counts from the real local data (dry-run, no model)"; mkws cnt real
  local I="$W/mpr-ivl" Q="$W/mpr-gui-crosslocale"
  for kr in "qwen3_vl_4b $Q" "internvl35_4b $I"; do set -- $kr; local k=$1 r=$2
    ( cd "$r" && source .venv/bin/activate && for f in common_b3.sh; do :; done
      tar -xzf "$W/b3_4b_addon_20261117.tar.gz" 2>/dev/null; X=followup_rel_6lang/b3_4b
      case $k in qwen3_vl_4b) CFG=$X/models/qwen3_vl_4b_4090_channels_last.yaml;; *) CFG=$X/models/internvl3_5_4b_4090.yaml;; esac
      python followup_rel_6lang/run_inference_6lang.py --model-config $CFG --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang --batch-size 1 --strict-batch-size --languages-subset new_only --dry-run --output-dir "$W/cnt_rel_$k" > "$W/cnt_rel_$k.log" 2>&1
      for v in main retry; do python $X/expA6/run_rel_T_6lang_multi.py --model $k --scope new --batch-size 1 --model-config $CFG --split main --variant $v --output-dir "$W/cnt_expa_${k}_$v" --dry-run > "$W/cnt_expa_${k}_$v.log" 2>&1; done )
    ck "$k REL new_only dry-run: 26,057 inferences, 29,148 condition rows, 366 items, languages en zh th ru fr ja, batch 1" bash -c "python3 -c \"
import json; p=json.load(open('$W/cnt_rel_$k/dry_run_plan.json'))
assert p['actual_inferences']==26057 and p['condition_rows']==29148 and p['items']==366 and p['batch_size']==1 and p['languages_subset']=='new_only' and p['languages']==['en','zh','th','ru','fr','ja'], p
print(p['actual_inferences'], p['condition_rows'])\""
    ck "$k Exp A main dry-run: 2,844 pairs x O/T/R = 8,532 (2,844 each), batch 1" bash -c "python3 -c \"
import json; p=json.load(open('$W/cnt_expa_${k}_main/plan.json'))
assert p['inferences']==8532 and p['pairs']==2844 and p['by_condition']=={'O':2844,'T':2844,'R':2844} and p['batch_size']==1 and p['scope']=='new', p
print(p['inferences'])\""
    ck "$k Exp A T_retry dry-run: 148 pairs x O/T_retry/R = 444 (148 each)" bash -c "python3 -c \"
import json; p=json.load(open('$W/cnt_expa_${k}_retry/plan.json'))
assert p['inferences']==444 and p['pairs']==148 and p['by_condition']=={'O':148,'T_retry':148,'R':148}, p
print(p['inferences'])\""
  done
  # 6x6: the dry-run plan function of the unchanged queue_6x6.sh (79,020 unique ids and the ID-list sha of the 8B runs)
  for kr in "qwen3_vl_4b $Q qwen3_vl_4b_4090_channels_last" "internvl35_4b $Q internvl3_5_4b_4090"; do set -- $kr
    ( cd "$2" && mkdir -p followup_rel_4lang/results/6x6_$1 && PATH="$2/.venv/bin:$PATH" W="$W" KEY=$1 OUT="$2/followup_rel_4lang/results/6x6_$1" CFGPATH="01_code/configs/models/$3.yaml" bash "$W/queue_6x6.sh" --fn dry_plan ) > "$W/cnt_6x6_$1.log" 2>&1; echo "rc=$?" >> "$W/cnt_6x6_$1.log"
    ck "$1 6x6 dry-run: 79,020 unique input ids, 2,195 semantic items, ID-list sha256 = the 8B 6x6 runs (22ac60aa...)" bash -c "grep -q 'input_ids 79020 unique 79020 semantic_items 2195' $W/cnt_6x6_$1.log && grep -q 'plan ids sha256 22ac60aa' $W/cnt_6x6_$1.log && grep -q 'rc=0' $W/cnt_6x6_$1.log"; done
  grep -h "input_ids\|plan ids" "$W/cnt_6x6_qwen3_vl_4b.log" "$W/cnt_6x6_internvl35_4b.log"; }

chk_real_model() {  # $1 key $2 repo
  local k=$1 r=$W/$2 R6=$W/$2/followup_rel_6lang/results
  ck "$k: packages 01_rel, 02_expA (+ 03_6x6 files), sha256 checks, per-file lists, COMPLETE" bash -c "cd $(RES)/$k && ls b3_${k}_01_rel_*.tar.gz b3_${k}_02_expA_*.tar.gz | grep -v PARTIAL | wc -l | grep -q 2 && shasum -a 256 -c b3_${k}_01_rel_*.tar.gz.sha256 b3_${k}_02_expA_*.tar.gz.sha256 && tar -tzf b3_${k}_01_rel_*.tar.gz | grep -q results_files.sha256 && [ -f COMPLETE ] && ls *6x6*.tar.gz b3_${k}_03_6x6_*.files.sha256"
  ck "$k: model step ok with real checks (bytes, architecture, A-D ids)" has "$(RES)/logs/model_$k.log" "'A': \["
  ck "$k: env check equals the earlier 4-language contracts" has "$(RES)/logs/setup_$k.log" 'equal the earlier 4-language runs'
  ck "$k: REL new_only 26,057 gate passed (rows 26,057), batch 1, strict true, 29,148 rows" bash -c "python3 -c \"
import json
g=json.load(open('$R6/rel6new_$k/full/gate_rel_complete.json')); assert g['status']=='passed' and g['rows']==26057, g
c=json.load(open('$R6/rel6new_$k/full/run_contract.json')); assert c['strict_batch_size']==True and c['batch_size']==1 and c['languages_subset']=='new_only', c
p=json.load(open('$R6/rel6new_$k/full/plan.json')); assert p['condition_rows']==29148 and p['actual_inferences']==26057, p
\""
  ck "$k: Exp A 8,532 and T_retry 444 gates passed; batch 1, strict true; anchor check vs own REL run ran (8,532-row O/R pairs = 5,688)" bash -c "python3 -c \"
import json
x=json.load(open('$R6/expA6_$k/full/gate_complete.json')); assert x['status']=='passed' and x['rows']==8532 and x['batch_sizes']=={'1':8532}, x
r=json.load(open('$R6/expA6_$k/retry/gate_complete.json')); assert r['status']=='passed' and r['rows']==444
c=json.load(open('$R6/expA6_$k/full/run_contract.json')); assert c['strict_batch_size']==True and c['batch_size']==1 and c['scope']=='new' and c['model']=='$k'
a=x['anchor_vs_rel_new_only_run']; assert a and a['anchor_rows']==5688, a
\""
  ck "$k: 6x6 gate passed, 79,020 rows, COMPLETE by queue_6x6.sh, tar linked into the b3 package folder" bash -c "python3 -c \"
import json
g=json.load(open('$r/followup_rel_4lang/results/6x6_$k/gate_6x6_full.json')); assert g['status']=='passed', g
print(g.get('rows') or g.get('success_rows'))
\" && [ -f $W/queue_results/6x6_$k/COMPLETE ] && [ \$(grep -c . $r/followup_rel_4lang/results/6x6_$k/full_6x6.jsonl) -ge 79020 ] && [ \$(wc -l < $(RES)/$k/b3_${k}_03_6x6_*.files.sha256) -ge 5 ]"
  ck "$k: model cache deleted after success (by queue_6x6.sh)" bash -c "[ ! -d $r/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct ] || [ '$k' != qwen3_vl_4b ]; [ ! -d $r/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B ] || [ '$k' != internvl35_4b ]"
  ck "$k: 4-language folder file list unchanged and originals byte-identical" bash -c "cd $r && find followup_rel_4lang -type f | sort | grep -v '^followup_rel_4lang/results/6x6_' | diff -q - <(grep -v '^followup_rel_4lang/results/' $W/files4_$2.txt; true) >/dev/null || find followup_rel_4lang -type f | grep -v '^followup_rel_4lang/results/' | sort | diff -q - $W/files4_$2.txt && shasum -a 256 -c $W/orig_sha_$2.txt"
  ck "$k: files of the B2 addon already in the repo were NOT touched (sha256 unchanged)" bash -c "cd $r && shasum -a 256 -c $W/b2addon_sha_$2.txt"
}
t_real() { echo "### real flow, two models (mock model, real queue_6x6.sh with Q6_TEST=1)"; mkws real real; q
  ck "QUEUE_B3_DONE = OK" done_is OK
  ck "network, GPU and wait lines in the status" bash -c "grep -q 'huggingface.co 40.0 MB/s' $(ST) && grep -q 'GPU idle' $(ST) && grep -q 'wait over after' $(ST)"
  for kr in "qwen3_vl_4b mpr-gui-crosslocale" "internvl35_4b mpr-ivl"; do chk_real_model $kr; done
  ck "result folders: exactly the new ones (rel6new_, expA6_) per model, nothing in the 4-language tree except 6x6_<key>" bash -c "[ \"\$(ls $W/mpr-ivl/followup_rel_6lang/results | tr '\n' ' ')\" = 'expA6_internvl35_4b rel6new_internvl35_4b ' ] && [ \"\$(ls $W/mpr-gui-crosslocale/followup_rel_6lang/results | tr '\n' ' ')\" = 'expA6_qwen3_vl_4b rel6new_qwen3_vl_4b ' ] && [ \"\$(ls $W/mpr-ivl/followup_rel_4lang/results | tr '\n' ' ')\" = '6x6_internvl35_4b ' ]"
  ck "InternVL repo got the 6x6 raw data as a link to the Qwen repo data (verified: HF c1edb808, manifest 4dd919fc)" bash -c "[ -L $W/mpr-ivl/01_code/data/raw/mpr_gui_bench ] && grep -q 'raw data verified' $W/queue_results/b3_4b/logs/six_setup_internvl35_4b.log && grep -q 'OK: .* raw data verified' $W/queue_results/b3_4b/logs/six_setup_qwen3_vl_4b.log"
  ck "6x6 setup log: Qwen data reused (verified in place), no download" bash -c "grep -q 'raw data present and verified' $W/queue_results/b3_4b/logs/six_setup_qwen3_vl_4b.log && ! grep -q -i 'downloading MPR-GUI-Bench' $W/queue_results/b3_4b/logs/six_setup_qwen3_vl_4b.log"
  ck "final package contains the sha list of every package" bash -c "tar -xzOf $(RES)/b3_4b_00_final_*.tar.gz ./PACKAGES_SHA256.txt | grep -c tar.gz | grep -q -E '^[6-9]|^[1-9][0-9]'"
  local n; n=$(wc -l < "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full/predictions.jsonl")
  rm -f "$(RES)/internvl35_4b/COMPLETE" "$W/queue_results/6x6_internvl35_4b/COMPLETE"; cp -Rl "$S/b3/hf_fakeI/hub/"* "$W/mpr-ivl/.cache/huggingface/hub/"; ONLY="internvl35_4b" q
  ck "rerun of one model without COMPLETE: gates passed on disk -> packaging only, no new predictions, 6x6 resumes, DONE OK" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ] && grep -q 'gates already passed on disk; packaging only' $W/QUEUE_B3_STATUS && [ $n -eq \$(wc -l < $W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_4b/full/predictions.jsonl) ]"
}
t_resume() { echo "### resume after a cut"; mkws res real; ONLY="qwen3_vl_4b" SKIP_6X6=1 q; local U=$W/mpr-gui-crosslocale/followup_rel_6lang/results
  head -n 9000 "$U/rel6new_qwen3_vl_4b/full/predictions.jsonl" > "$W/head.jsonl"; cp "$W/head.jsonl" "$U/rel6new_qwen3_vl_4b/full/predictions.jsonl"
  rm -f "${U:?}/rel6new_qwen3_vl_4b/full/gate_rel_complete.json" "${U:?}/rel6new_qwen3_vl_4b/full/results.jsonl" "${U:?}/rel6new_qwen3_vl_4b/full/summary.json" "$(RES)/qwen3_vl_4b/COMPLETE"
  cp -Rl "$S/b3/hf_fakeQ/hub/"* "$W/mpr-gui-crosslocale/.cache/huggingface/hub/"; ONLY="qwen3_vl_4b" SKIP_6X6=1 q
  ck "completed: 26,057 unique rows, the first 9,000 lines unchanged, DONE OK" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ] && [ \$(wc -l < $U/rel6new_qwen3_vl_4b/full/predictions.jsonl) -eq 26057 ] && head -n 9000 $U/rel6new_qwen3_vl_4b/full/predictions.jsonl | cmp - $W/head.jsonl"; }
t_setup() { echo "### real setup / model / 6x6 negatives (ONLY=internvl35_4b unless noted, SKIP_6X6=1 unless noted)"
  mkws s1 real; echo "# changed" >> "$W/mpr-ivl/followup_rel_6lang/run_inference_6lang.py"; ONLY=internvl35_4b SKIP_6X6=1 q
  ck "an existing server file differs from the addon version: setup refuses to overwrite, queue STOPS before any download" bash -c "grep -q 'refusing to overwrite' $(RES)/logs/setup_internvl35_4b.log && grep -q 'FAILED steps: setup_internvl35_4b' $W/QUEUE_B3_DONE && ! grep -q 'step model_internvl35_4b started' $W/QUEUE_B3_STATUS"
  mkws s2 real; f="$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/system_prompts.py"; sed 's/def build_system_prompt/def build_system_promptX/' "$f" > "$W/sp.py"; rm -f "$f"; cp "$W/sp.py" "$f"; ONLY=internvl35_4b SKIP_6X6=1 q
  ck "repo inference code differs from the earlier run: setup FAILED before any download" bash -c "grep -q 'FAILED steps: setup_internvl35_4b' $W/QUEUE_B3_DONE && ! grep -q 'model_internvl35_4b started' $W/QUEUE_B3_STATUS"
  mkws s3 real; EXPECT_TORCH=9.9.9 ONLY=internvl35_4b SKIP_6X6=1 q; ck "wrong torch version: setup FAILED, nothing installed" done_starts "FAILED steps: setup_internvl35_4b"
  mkws s4 real; EXPECT_SKIP=torchvision,sentencepiece,protobuf ONLY=qwen3_vl_4b SKIP_6X6=1 q; ck "qwen-vl-utils missing from the venv (not skipped): setup FAILED for qwen3_vl_4b" done_starts "FAILED steps: setup_qwen3_vl_4b"
  mkws s5 real; f=$(ls $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/model-00002*); rm -f "$f"; truncate -s 100 "$f"; ONLY=qwen3_vl_4b SKIP_6X6=1 q
  ck "Qwen3-VL-4B safetensors not exactly 8,875,719,344 bytes: model step FAILED" done_starts "FAILED steps: model_qwen3_vl_4b"
  mkws s6 real; f=$(ls $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B/snapshots/*/model-00002*); rm -f "$f"; truncate -s 1000 "$f"; ONLY=internvl35_4b SKIP_6X6=1 q
  ck "InternVL3.5-4B safetensors outside 8.0-10.5 GB: model step FAILED" done_starts "FAILED steps: model_internvl35_4b"
  mkws s7 real; c=$(ls $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B/snapshots/*/vocab.json); rm -f "$c"; echo '{"x": 1}' > "$c"; ONLY=internvl35_4b SKIP_6X6=1 q
  ck "tokenizer without the expected A-D ids / unusable: model step FAILED" done_starts "FAILED steps: model_internvl35_4b"
  mkws s8 real; rm -rf "${W:?}/mpr-ivl/followup_rel_4lang/assets/images/1/ru"; ONLY=internvl35_4b SKIP_6X6=1 q; ck "missing 4-language image (offline): setup FAILED" done_starts "FAILED steps: setup_internvl35_4b"
  mkws s9 real; f=$(ls $W/mpr-ivl/followup_rel_6lang/assets/images/*/fr/* | head -1); rm -f "$f"; echo bad > "$f"; ONLY=internvl35_4b SKIP_6X6=1 q; ck "corrupted fr image (sha256): setup FAILED" done_starts "FAILED steps: setup_internvl35_4b"
  mkws s10 real; rm -rf "${W:?}/mpr-ivl/.venv"; q; ck "InternVL repo without venv: STOP repo for internvl35_4b after qwen finished a+b; no 6x6 (environmental)" bash -c "grep -q 'repo_internvl35_4b' $W/QUEUE_B3_DONE && [ -f $(RES)/qwen3_vl_4b/COMPLETE ] && ! grep -q 'step six_setup' $W/QUEUE_B3_STATUS"
  mkws s11 real
  ( cd "$W/mpr-ivl" && tar -xzf "$W/b3_4b_addon_20261117.tar.gz" && echo "# extra" >> followup_rel_6lang/b3_4b/models/internvl3_5_4b_4090.yaml ); ONLY=internvl35_4b SKIP_6X6=1 q
  ck "b3 addon file on the server modified after an earlier unpack (config yaml): refused (differs / SHA256SUMS)" done_starts "FAILED steps: setup_internvl35_4b"; }
t_six() { echo "### 6x6 guards (ONLY_6X6=1, real queue_6x6.sh, Q6_TEST=1)"
  mkws x1 real; ONLY_6X6=1 q
  ck "6x6 only: both models complete with the 79,020 gate, DONE OK, a+b folders untouched (no results dirs of 6lang)" bash -c "[ \"\$(cat $W/QUEUE_B3_DONE)\" = OK ] && [ -f $W/queue_results/6x6_qwen3_vl_4b/COMPLETE ] && [ -f $W/queue_results/6x6_internvl35_4b/COMPLETE ] && [ ! -d $W/mpr-ivl/followup_rel_6lang/results ] && [ ! -d $W/mpr-gui-crosslocale/followup_rel_6lang/results ]"
  ck "the real 6x6 smoke gate and the dry-run plan were run for each model" bash -c "grep -q 'dry-run plan: input_ids 79020' $W/queue_logs6/qwen3_vl_4b_smoke.log || grep -q 'GATE' $W/mpr-gui-crosslocale/followup_rel_4lang/results/6x6_qwen3_vl_4b/logs/smoke_gate.log; ls $W/mpr-ivl/followup_rel_4lang/results/6x6_internvl35_4b/gate_6x6_smoke.json $W/mpr-ivl/followup_rel_4lang/results/6x6_internvl35_4b/gate_6x6_full.json"
  mkws x2 real; ( cd "$W/mpr-gui-crosslocale" && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m extra ); ONLY_6X6=1 q
  ck "Qwen repo HEAD is not 6664400: six_setup refuses (no git checkout -f), queue_6x6.sh not started for it; InternVL 6x6 still runs" bash -c "grep -q 'queue_6x6.sh would run git checkout -f' $(RES)/logs/six_setup_qwen3_vl_4b.log && [ ! -d $W/mpr-gui-crosslocale/followup_rel_4lang/results/6x6_qwen3_vl_4b ] && [ -f $W/queue_results/6x6_internvl35_4b/COMPLETE ] && grep -q 'six_setup_qwen3_vl_4b' $W/QUEUE_B3_DONE"
  mkws x3 real; ( cd "$W/mpr-ivl" && git reset -q --hard HEAD~1 ); ONLY_6X6=1 q
  ck "InternVL repo has only 4 of the 5 patches: six_setup refuses (nothing reset by us), Qwen 6x6 completes" bash -c "grep -q 'expected the 5 InternVL patches' $(RES)/logs/six_setup_internvl35_4b.log && [ -f $W/queue_results/6x6_qwen3_vl_4b/COMPLETE ]"
  mkws x4 real; f="$W/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench/SOURCE.json"; sed 's/c1edb808d424a2fa7bc4a2e601d39b431b04acd5/0000000000000000000000000000000000000000/' "$f" > "$W/src.json"; rm -f "$f"; cp "$W/src.json" "$f"; ONLY_6X6=1 q
  ck "existing 6x6 raw data with another HF revision: six_setup refuses (nothing downloaded, nothing deleted)" bash -c "grep -q 'differs from the pinned revision' $(RES)/logs/six_setup_qwen3_vl_4b.log && grep -q 'Hugging Face revision' $(RES)/logs/six_setup_qwen3_vl_4b.log && [ -f $f ]"
  mkws x5 real; ( cd "$W/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench/images" && f=$(ls */*/* | head -1) && cp "$f" "$W/img.bak" && rm -f "$f" && : > "$f" ); ONLY_6X6=1 q
  ck "a manifest image is empty/missing: refused at the full check (verify after fn_setup / before)" bash -c "grep -q -E 'manifest images missing or empty' $(RES)/logs/six_setup_qwen3_vl_4b.log"
  mkws x6 real; rm -rf "${W:?}/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench" "${W:?}/mpr-gui-crosslocale/01_code/data/manifests/mpr_gui_manifest.jsonl"; ONLY=qwen3_vl_4b ONLY_6X6=1 q
  ck "no 6x6 data anywhere: setup says it downloads (2.6 GB); queue_6x6.sh (test download = copy) completes, manifest rebuilt with the right sha" bash -c "grep -q 'queue_6x6.sh downloads it' $(RES)/logs/six_setup_qwen3_vl_4b.log && [ -f $W/queue_results/6x6_qwen3_vl_4b/COMPLETE ]"
  mkws x7 real; rm -rf "${W:?}/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench"; FAKE_FREE=6 ONLY=qwen3_vl_4b ONLY_6X6=1 q
  ck "6x6 data absent and less than 8 GB free: six_setup stops before queue_6x6.sh" bash -c "grep -q 'less than 8 GB free for the 2.6 GB download' $(RES)/logs/six_setup_qwen3_vl_4b.log && [ ! -d $W/queue_results/6x6_qwen3_vl_4b ]"
}
case "${1:-all}" in all) for t in plan stub wait guards counts real resume setup six; do "t_$t"; done ;; *) "t_$1" ;; esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
