#!/usr/bin/env bash
# Flow tests for queue_b2_frja.sh in a fake /workspace. No GPU, no network (proxy -> closed port), no pip, no model download.
# real = real setup() (addon copy-if-absent, SHA256SUMS, env_check, data/image verify), real model() (offline fake HF caches, stand-in tokenizers), real run scripts with --mock-model, real gates, packaging.
# stub = setup/model/smoke/full/retry replaced by stubs (make_stub_b2.py): guards, failure rule, order, cache rule, packaging.
# usage: run_b2_queue_tests.sh plan|stub|guards|real|setup|all
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
ROOT="${PROJECT_ROOT}"
WS=$S/b2/qtest
ADDON25="$ROOT/model_expansion_20261116/internvl25_rel6/internvl25_rel6_addon_20261116.tar.gz"
FRJA="$ROOT/followup_rel_6lang/assets"
SZ="$ROOT/model_expansion_20261114/sizes_uitars/addon"
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 PIP_INDEX_URL=http://127.0.0.1:9/simple
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 EXPECT_SKIP=qwen-vl-utils,torchvision,protobuf,sentencepiece GPU_RECHECK_TRIES=2 GPU_RECHECK_S=1
export OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh|queue_other_test\.sh'
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
pystub() { printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$1" "$S"; }
mkrepo() {  # $1 = path, $2 = template, $3 = name
  cp -Rl "$2" "$1"; rm -rf "${1:?}/followup_rel_4lang/expA_ivl25" "${1:?}/followup_rel_4lang/results"
  rm -f "$1/.venv/bin/python" "$1/.venv/bin/activate" "$1/.venv/bin/python3"
  pystub "$1" > "$1/.venv/bin/python"; chmod +x "$1/.venv/bin/python"; ln -s python "$1/.venv/bin/python3"
  printf 'export PATH="%s/.venv/bin:$PATH"\n' "$1" > "$1/.venv/bin/activate"; mkdir -p "$1/.cache/huggingface"
}
mkws() {  # $1 name  $2 = real|stub
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state
  rm -rf "${WS:?}/ws_$1"; mkdir -p "$W" "$FAKE_STATE"
  ln "$M/b2_frja_addon_20261116.tar.gz" "$W/"
  if [ "$2" = stub ]; then
    python3 "$HERE/make_stub_b2.py" "$M/queue_b2_frja.sh" "$W/queue_b2_frja.sh"
    for r in mpr-ivl mpr-gui-crosslocale; do mkdir -p "$W/$r/.git" "$W/$r/.venv/bin" "$W/$r/01_code"; touch "$W/$r/.venv/bin/activate"; done
  else
    cp "$M/queue_b2_frja.sh" "$W/"
    mkrepo "$W/mpr-ivl" "$S/ivl25_repo" ivl; mkrepo "$W/mpr-gui-crosslocale" "$S/q3frja/template" q
    cp -Rl "$S/ivl25_work/hf_fake/hub" "$W/mpr-ivl/.cache/huggingface/hub"; cp -Rl "$S/b2/hf_fake35/hub/"* "$W/mpr-ivl/.cache/huggingface/hub/"
    cp -Rl "$S/b2/hf_fakeU/hub" "$W/mpr-gui-crosslocale/.cache/huggingface/hub"
    mkdir -p "$W/mpr-gui-crosslocale/followup_rel_4lang/model_expansion"; rm -f "$W/mpr-gui-crosslocale/01_code/configs/models/uitars_1_5_7b_4090.yaml"; cp "$SZ/01_code/configs/models/uitars_1_5_7b_4090.yaml" "$W/mpr-gui-crosslocale/01_code/configs/models/"
    ( cd "$W/mpr-ivl" && tar -xzf "$ADDON25" && cp -Rl "$FRJA" followup_rel_6lang/assets )   # server state after the InternVL2.5 REL6 queue (its addon + images)
    for r in mpr-ivl mpr-gui-crosslocale; do ( cd "$W/$r" && find followup_rel_4lang -type f | sort > "$W/files4_$r.txt"; shasum -a 256 01_code/src/mpr_crosslocale/inference/run_rel_followup.py 01_code/src/mpr_crosslocale/data/rel_followup.py > "$W/orig_sha_$r.txt" ); done
    ( cd "$W/mpr-ivl" && shasum -a 256 followup_rel_6lang/ivl25_rel6/* 01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py followup_rel_6lang/run_inference_6lang.py > "$W/ivl25_addon_sha.txt" )
  fi
}
q() { ( cd "$W" && W="$W" bash queue_b2_frja.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_B2_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_B2_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
steps() { tr '\n' ',' < "$W/steps_run.txt" 2>/dev/null; }
RES() { echo "$W/queue_results/b2_frja"; }
ST() { echo "$W/QUEUE_B2_STATUS"; }
FULLSTEPS="setup internvl25_8b,model internvl25_8b,smoke_expa internvl25_8b,full_expa internvl25_8b,retry_expa internvl25_8b,setup internvl35_8b,model internvl35_8b,smoke_rel internvl35_8b,smoke_expa internvl35_8b,full_rel internvl35_8b,full_expa internvl35_8b,retry_expa internvl35_8b,setup uitars_15_7b,model uitars_15_7b,smoke_rel uitars_15_7b,smoke_expa uitars_15_7b,full_rel uitars_15_7b,full_expa uitars_15_7b,retry_expa uitars_15_7b,"

t_plan() { echo "### plan"; mkws plan stub; q --plan
  ck "plan: order, 14,220 / 786 for InternVL2.5 (all6), 26,057 REL, 8,532 / 444 Exp A, strict flags" bash -c "grep -q 'internvl25_8b internvl35_8b uitars_15_7b' $W/out.txt && grep -q '14,220' $W/out.txt && grep -q 786 $W/out.txt && grep -q '26,057' $W/out.txt && grep -q '8,532' $W/out.txt && grep -q 'strict false' $W/out.txt && grep -q -- '--strict-batch-size' $W/out.txt"
  ck "plan labels estimates, states disk budget and the failure rule" bash -c "grep -q 'Estimated times (estimates' $W/out.txt && grep -q 'Disk (about 67 GB' $W/out.txt && grep -q 'Failure rule' $W/out.txt"
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE_B2_STATUS ] && [ ! -e $W/QUEUE_B2_DONE ] && [ ! -d $W/queue_results ]"
  mkws plan2 stub; ONLY="uitars_15_7b internvl25_8b" q --plan; ck "ONLY keeps the fixed order" has "$W/out.txt" 'Models in this order: internvl25_8b uitars_15_7b'
  mkws plan3 stub; EXPA_BATCH_internvl25_8b=3 q; ck "EXPA_BATCH_internvl25_8b=3 refused (earlier run batch 1)" bash -c "[ \$(cat $W/rc.txt) = 2 ]"
  ck "bash -n on all scripts" bash -c "cd \"$M\" && bash -n queue_b2_frja.sh && bash -n step1_ivl35_rel/queue_b2_step1_ivl35_rel.sh && for f in addon/followup_rel_6lang/b2_frja/*.sh build_addon.sh; do bash -n \$f || exit 1; done"; }
t_stub() { echo "### stub flow, failure rule"
  mkws a stub; q; ck "all succeed: DONE OK, 19 steps in the planned order" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ] && [ \"\$(tr '\n' ',' < $W/steps_run.txt)\" = '$FULLSTEPS' ]"
  ck "per-model packages with sha256 and COMPLETE markers; caches deleted" bash -c "for k in internvl25_8b internvl35_8b uitars_15_7b; do ls $(RES)/\$k/b2_\${k}_results_*.tar.gz >/dev/null && (cd $(RES)/\$k && shasum -a 256 -c *.sha256) && [ -f $(RES)/\$k/COMPLETE ] || exit 1; done; [ ! -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B ] && [ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B ]"
  ck "final status lists the three models as ok" has "$(ST)" 'models: internvl25_8b=ok internvl35_8b=ok uitars_15_7b=ok'
  q; ck "rerun: COMPLETE models skipped, nothing re-run" bash -c "grep -c 'already complete' $W/out.txt | grep -q 3 && [ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ]"
  mkws k stub; KEEP_MODEL_CACHE=1 q; ck "KEEP_MODEL_CACHE=1 keeps all caches" test -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B"
  # model-specific failures: the queue continues
  mkws f1 stub; FAKE_FAIL="full_expa:internvl25_8b" q
  ck "InternVL2.5 Exp A full fails: retry skipped for it, the next two models still run" bash -c "! grep -q 'retry_expa internvl25_8b' $W/steps_run.txt && grep -q 'full_expa internvl35_8b' $W/steps_run.txt && grep -q 'retry_expa uitars_15_7b' $W/steps_run.txt"
  ck "verdict names the failed step; PARTIAL package; cache of the failed model kept; others deleted" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: full_expa_internvl25_8b' ] && ls $(RES)/internvl25_8b/*_PARTIAL.tar.gz && [ ! -f $(RES)/internvl25_8b/COMPLETE ] && [ -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B ] && [ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B ]"
  ck "status says 'continuing with the next model'" has "$(ST)" 'model-specific problem; environment re-check ok; continuing'
  mkws f2 stub; FAKE_FAIL="smoke_rel:internvl35_8b" q
  ck "InternVL3.5 REL smoke fails: its Exp A still runs, REL full skipped, UI-TARS runs" bash -c "! grep -q 'full_rel internvl35_8b' $W/steps_run.txt && grep -q 'full_expa internvl35_8b' $W/steps_run.txt && grep -q 'full_rel uitars_15_7b' $W/steps_run.txt && [ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: smoke_rel_internvl35_8b' ]"
  mkws f3 stub; FAKE_FAIL="model:internvl35_8b" q
  ck "InternVL3.5 model step fails (download/check): that model stops, UI-TARS still runs" bash -c "grep -q 'setup uitars_15_7b' $W/steps_run.txt && ! grep -q 'smoke_expa internvl35_8b' $W/steps_run.txt && [ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: model_internvl35_8b' ]"
  mkws f4 stub; FAKE_ETA="full_rel:internvl35_8b" q
  ck "ETA stop (exit 3) on InternVL3.5 REL: model-specific, queue continues" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: full_rel_internvl35_8b_eta' ] && grep -q 'full_expa uitars_15_7b' $W/steps_run.txt && grep -q 'smoke ETA longer' $W/QUEUE_B2_STATUS"
  mkws f5 stub; FAKE_SLEEP="full_rel:internvl35_8b" FAKE_TIMEOUT_MATCH="full_rel" FAKE_TIMEOUT_SECS=2 q
  ck "timeout of a full step: reported, model-specific, queue continues" bash -c "has() { grep -q -- \"\$2\" \"\$1\"; }; grep -q 'step full_rel_internvl35_8b TIMEOUT' $W/QUEUE_B2_STATUS && grep -q 'retry_expa uitars_15_7b' $W/steps_run.txt"
  mkws f6 stub; FAKE_FAIL="smoke_expa:uitars_15_7b" q
  ck "UI-TARS Exp A smoke fails: REL runs, Exp A full skipped" bash -c "grep -q 'full_rel uitars_15_7b' $W/steps_run.txt && ! grep -q 'full_expa uitars_15_7b' $W/steps_run.txt"
  mkws f7 stub; ONLY="internvl35_8b" SKIP_REL_internvl35_8b=1 q; ck "SKIP_REL_internvl35_8b=1: REL steps not run" bash -c "! grep -q 'rel internvl35_8b' $W/steps_run.txt && [ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ]"
  mkws f8 stub; mkdir -p "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_8b/full"; echo '{"status": "passed"}' > "$W/mpr-ivl/followup_rel_6lang/results/rel6new_internvl35_8b/full/gate_rel_complete.json"; ONLY="internvl35_8b" q
  ck "step 1 already passed its REL gate: REL steps skipped with a status line, Exp A runs" bash -c "! grep -q 'rel internvl35_8b' $W/steps_run.txt && grep -q 'REL fr/ja already complete' $W/QUEUE_B2_STATUS && grep -q 'full_expa internvl35_8b' $W/steps_run.txt"
  # environmental failures: the queue stops
  mkws e1 stub; FAKE_FAIL="setup:internvl35_8b" q
  ck "setup of InternVL3.5 fails (identity/venv): queue STOPS, UI-TARS not started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: setup_internvl35_8b' ] && ! grep -q 'uitars_15_7b' $W/steps_run.txt && grep -q 'STOP QUEUE (environmental failure' $W/QUEUE_B2_STATUS && grep -q 'uitars_15_7b=not_started' $W/QUEUE_B2_STATUS"
  mkws e2 stub; FAKE_CUDA="full_rel:internvl35_8b" q
  ck "CUDA fatal error text in a step log: queue STOPS, UI-TARS not started" bash -c "grep -q 'STOP QUEUE (environmental failure: CUDA fatal' $W/QUEUE_B2_STATUS && ! grep -q 'setup uitars_15_7b' $W/steps_run.txt"
  mkws e3 stub; FAKE_FAIL="full_expa:internvl35_8b" FAKE_BUSY_AFTER=1 q
  ck "failed step leaves the GPU busy: environment re-check STOPS the queue" bash -c "grep -q 'environment re-check after the failed internvl35_8b' $W/QUEUE_B2_STATUS && ! grep -q 'setup uitars_15_7b' $W/steps_run.txt && grep -q 'env_after_internvl35_8b' $W/QUEUE_B2_DONE"
  mkws e4 stub; FAKE_FREE=40 q
  ck "40 GB free: both InternVL models run, UI-TARS (needs 48) STOPS on disk before downloading" bash -c "grep -q 'full_expa internvl35_8b' $W/steps_run.txt && grep -q 'STOP: only 40 GB free (need 48 GB for uitars_15_7b)' $W/QUEUE_B2_STATUS && ! grep -q 'model uitars_15_7b' $W/steps_run.txt && grep -q 'disk_uitars_15_7b' $W/QUEUE_B2_DONE"
  mkws e5 stub; rm -rf "${W:?}/mpr-gui-crosslocale/.venv"; q
  ck "UI-TARS repo without venv: STOP repo for that model (InternVL models done before)" bash -c "grep -q 'repo_uitars_15_7b' $W/QUEUE_B2_DONE && grep -q 'retry_expa internvl35_8b' $W/steps_run.txt"
  mkws d1 stub; ( cd "$W" && W="$W" FAKE_SLEEP="smoke_expa:internvl25_8b" bash queue_b2_frja.sh > out1.txt 2>&1 ) & local p=$!; sleep 4; q
  ck "duplicate start refused (already running), DONE untouched" bash -c "grep -q 'already running' $W/out.txt && [ \$(cat $W/rc.txt) = 1 ] && [ ! -e $W/QUEUE_B2_DONE ]"
  kill -TERM $p 2>/dev/null; pkill -TERM -P $p 2>/dev/null; wait $p 2>/dev/null
  mkws d2 stub; ( cd "$W" && W="$W" FAKE_SLEEP="smoke_expa:internvl25_8b" bash queue_b2_frja.sh > out1.txt 2>&1 ) & p=$!; sleep 4; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; for _ in $(seq 1 45); do has "$W/QUEUE_B2_STATUS" 'interrupted by signal' && break; sleep 1; done; wait $p 2>/dev/null
  ck "SIGTERM: status says interrupted, no QUEUE_B2_DONE" bash -c "grep -q 'interrupted by signal' $W/QUEUE_B2_STATUS && [ ! -e $W/QUEUE_B2_DONE ]"; }
t_guards() { echo "### guards"
  mkws g1 stub; FAKE_GPU_APPS="4242, python, 9000 MiB" q; ck "GPU process: FAILED steps: gpu, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: gpu' ] && [ ! -e $W/steps_run.txt ]"
  mkws g2 stub; FAKE_GPU_USED=20000 q; ck "GPU memory used: refused" done_starts "FAILED steps: gpu"
  mkws g3 stub; FAKE_GPU_NAME="NVIDIA A100" q; ck "not a 4090: refused" done_starts "FAILED steps: gpu"
  mkws g4 stub; mkdir -p "$S/b2/fp"; printf '#!/bin/bash\nsleep 40\n' > "$S/b2/fp/queue_other_test.sh"; bash "$S/b2/fp/queue_other_test.sh" & local p=$!; sleep 1; q; kill $p 2>/dev/null; wait $p 2>/dev/null
  ck "another queue script running: refused (others), nothing started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: others' ] && [ ! -e $W/steps_run.txt ]"
  mkws g5 stub; FAKE_NET_HF="6000000 10.0 206" q; ck "slow network: STOP network, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: network' ] && [ ! -e $W/steps_run.txt ]"
  mkws g6 stub; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q; ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws g7 stub; FAKE_NET_HF="0 1.0 404" q; ck "HTTP 404 on the test URL: warning only" done_is OK
  mkws g8 stub; rm -f "$W/b2_frja_addon_20261116.tar.gz"; echo junk > "$W/b2_frja_addon_20261116.tar.gz"; q; ck "addon sha mismatch: STOP addon, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: addon' ] && [ ! -e $W/steps_run.txt ]"
  mkws g9 stub; rm -f "$W/b2_frja_addon_20261116.tar.gz"; q; ck "addon missing: STOP addon" done_is "FAILED steps: addon"
  mkws g10 stub; echo junk > "$W/internvl25_rel6_frja_images_20261116.tar"; q; ck "image tar with a wrong sha256: STOP addon" done_is "FAILED steps: addon"
  ck "static: no pip install, no rm -rf outside the queue's own cache/stage paths" bash -c "! grep -q 'pip install' \"$M/queue_b2_frja.sh\" && ! grep 'rm -rf' \"$M/queue_b2_frja.sh\" | grep -v -E 'CACHE_DIR|stage|RES' | grep -q .";
  ck "static: nothing is written under followup_rel_4lang/results" bash -c "cd \"$M\" && ! grep -h 'followup_rel_4lang/results' queue_b2_frja.sh addon/followup_rel_6lang/b2_frja/*.sh | grep -v -E '^#|# never|never followup' | grep -q ."; }
chk_real_model() {  # $1 key $2 repo
  local k=$1 r=$W/$2
  ck "$k: package, sha256, per-file list, COMPLETE" bash -c "cd $(RES)/$k && ls b2_${k}_results_*.tar.gz | grep -v PARTIAL && shasum -a 256 -c b2_${k}_results_*.tar.gz.sha256 && tar -tzf b2_${k}_results_*.tar.gz | grep -q results_files.sha256 && [ -f COMPLETE ]"
  ck "$k: model step ok with real checks (bytes, architecture, A-D ids)" has "$(RES)/logs/model_$k.log" "'A': \["
  ck "$k: env check equals the earlier 4-language contracts" has "$(RES)/logs/setup_$k.log" 'equal the earlier 4-language runs'
  case $k in internvl25_8b) pat=InternVL2_5;; internvl35_8b) pat=InternVL3_5;; *) pat=UI-TARS;; esac
  ck "$k: model cache deleted after success" bash -c "! ls $r/.cache/huggingface/hub | grep -q -i -- $pat"
  ck "$k: 4-language folder file list unchanged and originals byte-identical" bash -c "cd $r && find followup_rel_4lang -type f | sort | diff -q - $W/files4_$2.txt && shasum -a 256 -c $W/orig_sha_$2.txt && [ ! -e followup_rel_4lang/results ]"
}
t_real() { echo "### real flow, three models (mock model)"; mkws real real; q
  ck "QUEUE_B2_DONE = OK" done_is OK
  ck "network and GPU lines in the status" bash -c "grep -q 'huggingface.co 40.0 MB/s' $(ST) && grep -q 'GPU idle' $(ST)"
  for kr in "internvl25_8b mpr-ivl" "internvl35_8b mpr-ivl" "uitars_15_7b mpr-gui-crosslocale"; do chk_real_model $kr; done
  local I=$W/mpr-ivl/followup_rel_6lang/results U=$W/mpr-gui-crosslocale/followup_rel_6lang/results
  ck "InternVL2.5 Exp A all6: 14,220 rows gate passed; retry 786 passed; batch sizes {1}" bash -c "python3 -c \"
import json
g=json.load(open('$I/expA6_internvl25_8b/full/gate_complete.json')); r=json.load(open('$I/expA6_internvl25_8b/retry/gate_complete.json'))
assert g['status']=='passed' and g['rows']==14220 and g['batch_sizes']=={'1':14220}, g['batch_sizes']
assert r['status']=='passed' and r['rows']==786
c=json.load(open('$I/expA6_internvl25_8b/full/run_contract.json')); assert c['scope']=='all6' and c['batch_size']==1 and c['strict_batch_size']==True
\""
  ck "InternVL2.5: no REL step in this queue (no rel6new_internvl25_8b folder)" test ! -e "$I/rel6new_internvl25_8b"
  ck "InternVL3.5: REL new_only 26,057 / 29,148 gate passed, strict false; Exp A 8,532 and retry 444 passed" bash -c "python3 -c \"
import json
g=json.load(open('$I/rel6new_internvl35_8b/full/gate_rel_complete.json')); assert g['status']=='passed' and g['rows']==26057
c=json.load(open('$I/rel6new_internvl35_8b/full/run_contract.json')); assert c['strict_batch_size']==False and c['batch_size']==1 and c['languages_subset']=='new_only'
x=json.load(open('$I/expA6_internvl35_8b/full/gate_complete.json')); assert x['status']=='passed' and x['rows']==8532
assert json.load(open('$I/expA6_internvl35_8b/retry/gate_complete.json'))['rows']==444
assert json.load(open('$I/expA6_internvl35_8b/full/run_contract.json'))['strict_batch_size']==False
\""
  ck "UI-TARS: REL 26,057 (strict true), Exp A 8,532 + retry 444 passed; fr/ja images were copied from mpr-ivl and verified" bash -c "python3 -c \"
import json
g=json.load(open('$U/rel6new_uitars_15_7b/full/gate_rel_complete.json')); assert g['status']=='passed' and g['rows']==26057
c=json.load(open('$U/rel6new_uitars_15_7b/full/run_contract.json')); assert c['strict_batch_size']==True and c['batch_size']==1
x=json.load(open('$U/expA6_uitars_15_7b/full/gate_complete.json')); assert x['status']=='passed' and x['rows']==8532
assert json.load(open('$U/expA6_uitars_15_7b/retry/gate_complete.json'))['rows']==444
\" && [ \$(find $W/mpr-gui-crosslocale/followup_rel_6lang/assets -type f ! -name .DS_Store | wc -l) -ge 729 ]"
  ck "Exp A anchor check ran for InternVL3.5 (O/R answers vs its own REL run) and for UI-TARS" bash -c "python3 -c \"
import json
for f in ['$I/expA6_internvl35_8b/full/gate_complete.json','$U/expA6_uitars_15_7b/full/gate_complete.json']:
    a=json.load(open(f))['anchor_vs_rel_new_only_run']; assert a and a['anchor_rows']==5688, a
\""
  ck "addon files: the InternVL2.5 REL6 files already on the server were NOT touched (sha256 unchanged)" bash -c "cd $W/mpr-ivl && shasum -a 256 -c $W/ivl25_addon_sha.txt"
  ck "result folders: exactly the 5 new ones, one set per model, none in the 4-language tree" bash -c "[ \"\$(ls $I | tr '\n' ' ')\" = 'expA6_internvl25_8b expA6_internvl35_8b rel6new_internvl35_8b ' ] && [ \"\$(ls $U | tr '\n' ' ')\" = 'expA6_uitars_15_7b rel6new_uitars_15_7b ' ]"
  local n; n=$(wc -l < "$I/rel6new_internvl35_8b/full/predictions.jsonl")
  rm -f "$(RES)/internvl35_8b/COMPLETE"; cp -Rl "$S/b2/hf_fake35/hub/"* "$W/mpr-ivl/.cache/huggingface/hub/"; ONLY="internvl35_8b" q
  ck "rerun of one model without COMPLETE: resumes ('already done'), no new predictions, DONE OK" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ] && grep -q 'REL fr/ja already complete' $W/QUEUE_B2_STATUS && grep -q 'Exp A full already done' $I/expA6_internvl35_8b/STATUS && [ $n -eq \$(wc -l < $I/rel6new_internvl35_8b/full/predictions.jsonl) ]"
}
t_resume() { echo "### resume after a cut"; mkws res real; ONLY="uitars_15_7b" q; local U=$W/mpr-gui-crosslocale/followup_rel_6lang/results
  head -n 9000 "$U/rel6new_uitars_15_7b/full/predictions.jsonl" > "$W/head.jsonl"; cp "$W/head.jsonl" "$U/rel6new_uitars_15_7b/full/predictions.jsonl"
  rm -f "${U:?}/rel6new_uitars_15_7b/full/gate_rel_complete.json" "${U:?}/rel6new_uitars_15_7b/full/results.jsonl" "${U:?}/rel6new_uitars_15_7b/full/summary.json" "$(RES)/uitars_15_7b/COMPLETE"
  cp -Rl "$S/b2/hf_fakeU/hub/"* "$W/mpr-gui-crosslocale/.cache/huggingface/hub/"; ONLY="uitars_15_7b" q
  ck "completed: 26,057 unique rows, the first 9,000 lines unchanged, DONE OK" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ] && [ \$(wc -l < $U/rel6new_uitars_15_7b/full/predictions.jsonl) -eq 26057 ] && head -n 9000 $U/rel6new_uitars_15_7b/full/predictions.jsonl | cmp - $W/head.jsonl"; }
t_setup() { echo "### real setup / model negatives (ONLY=internvl35_8b unless noted)"
  mkws s1 real; echo "# changed" >> "$W/mpr-ivl/followup_rel_6lang/run_inference_6lang.py"; ONLY=internvl35_8b q
  ck "an existing server file differs from the addon version: setup refuses to overwrite, queue STOPS" bash -c "grep -q 'refusing to overwrite' $(RES)/logs/setup_internvl35_8b.log && grep -q 'FAILED steps: setup_internvl35_8b' $W/QUEUE_B2_DONE && ! grep -q 'step model_internvl35_8b started' $W/QUEUE_B2_STATUS"
  mkws s2 real; f="$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/system_prompts.py"; sed 's/def build_system_prompt/def build_system_promptX/' "$f" > "$W/sp.py"; rm -f "$f"; cp "$W/sp.py" "$f"; ONLY=internvl35_8b q
  ck "repo inference code differs from the earlier run: setup FAILED before any download" bash -c "grep -q 'FAILED steps: setup_internvl35_8b' $W/QUEUE_B2_DONE && ! grep -q 'model_internvl35_8b started' $W/QUEUE_B2_STATUS"
  mkws s3 real; EXPECT_TORCH=9.9.9 ONLY=internvl35_8b q; ck "wrong torch version: setup FAILED, nothing installed" done_starts "FAILED steps: setup_internvl35_8b"
  mkws s4 real; EXPECT_SKIP=torchvision,sentencepiece,protobuf ONLY=uitars_15_7b q; ck "qwen-vl-utils missing from the venv (not skipped): setup FAILED for UI-TARS" done_starts "FAILED steps: setup_uitars_15_7b"
  mkws s5 real; f=$(ls $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/*/model-00004*); rm -f "$f"; truncate -s 100 "$f"; ONLY=uitars_15_7b q
  ck "UI-TARS safetensors total outside 32.5-34.0 GB: model step FAILED" done_starts "FAILED steps: model_uitars_15_7b"
  mkws s6 real; f=$(ls $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/*/model-00004*); rm -f "$f"; truncate -s 1000 "$f"; ONLY=internvl25_8b q
  ck "InternVL2.5 byte sum differs from the exact 16,150,800,432: model step FAILED" done_starts "FAILED steps: model_internvl25_8b"
  mkws s7 real; c=$(ls $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/*/vocab.json); rm -f "$c"; echo '{"x": 1}' > "$c"; ONLY=internvl35_8b q
  ck "tokenizer without the expected A-D ids / unusable: model step FAILED" done_starts "FAILED steps: model_internvl35_8b"
  mkws s8 real; rm -rf "${W:?}/mpr-ivl/followup_rel_4lang/assets/images/1/ru"; ONLY=internvl35_8b q; ck "missing 4-language image (offline): setup FAILED" done_starts "FAILED steps: setup_internvl35_8b"
  mkws s9 real; f=$(ls $W/mpr-ivl/followup_rel_6lang/assets/images/*/fr/* | head -1); rm -f "$f"; echo bad > "$f"; ONLY=internvl35_8b q; ck "corrupted fr image (sha256): setup FAILED" done_starts "FAILED steps: setup_internvl35_8b"
  mkws s10 real; sed 's/^ADDON_SHA=.*/ADDON_SHA=0000/' "$W/queue_b2_frja.sh" > "$W/qq.sh"; rm -f "$W/queue_b2_frja.sh"; cp "$W/qq.sh" "$W/queue_b2_frja.sh"; q; ck "addon sha mismatch in the real script: STOP addon" done_is "FAILED steps: addon"
  mkws s11 real; rm -rf "${W:?}/mpr-ivl/.venv"; q; ck "InternVL repo without venv: STOP repo for InternVL2.5; the queue ends (environmental)" bash -c "grep -q 'repo_internvl25_8b' $W/QUEUE_B2_DONE"
  mkws s12 real; ( cd "$W/mpr-ivl" && tar -xzf "$M/b2_frja_addon_20261116.tar.gz" && echo "# extra" >> followup_rel_6lang/b2_frja/models/internvl3_5_8b_4090.yaml ); ONLY=internvl35_8b q
  ck "addon file on the server modified after an earlier unpack (config yaml): refused (differs / SHA256SUMS)" done_starts "FAILED steps: setup_internvl35_8b"
}
case "${1:-all}" in all) for t in plan stub guards real resume setup; do "t_$t"; done ;; *) "t_$1" ;; esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
