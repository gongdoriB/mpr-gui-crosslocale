#!/usr/bin/env bash
# Flow tests for queue_b2_step1_ivl35_rel.sh in a fake /workspace. No GPU, no network (proxy points to a closed port), no pip, no model download.
# real = real setup()/model() (offline fake HF cache, stand-in tokenizer)/smoke/full with --mock-model (LOCAL_MOCK=1)/real gates/real packaging.
# stub = setup/model/smoke/full replaced by stubs (make_stub_step1.py).
# usage: run_step1_tests.sh all|plan|real|resume|guards|setup|stub
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
ROOT="${PROJECT_ROOT}"
TEMPLATE=$S/ivl25_repo; FAKE_HF=$S/b2/hf_fake35; WS=$S/b2/ws1
ADDON25="$ROOT/model_expansion_20261116/internvl25_rel6/internvl25_rel6_addon_20261116.tar.gz"
FRJA="$ROOT/followup_rel_6lang/assets"
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 PIP_INDEX_URL=http://127.0.0.1:9/simple
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 EXPECT_SKIP=torchvision,sentencepiece OTHERS_RE='run_all_overnight\.sh|queue_6x6\.sh|queue_sizes_uitars\.sh|queue_other_test\.sh'
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
OUTR=followup_rel_6lang/results/rel6new_internvl35_8b
mkws() {  # $1 name  $2 = real|stub
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state
  rm -rf "${WS:?}/ws_$1"; mkdir -p "$W" "$FAKE_STATE"
  if [ "$2" = stub ]; then
    python3 "$HERE/make_stub_step1.py" "$M/queue_b2_step1_ivl35_rel.sh" "$W/queue_b2_step1_ivl35_rel.sh"; mkdir -p "$W/mpr-ivl/.git" "$W/mpr-ivl/.venv/bin"; touch "$W/mpr-ivl/.venv/bin/activate"
  else
    cp "$M/queue_b2_step1_ivl35_rel.sh" "$W/"
    cp -Rl "$TEMPLATE" "$W/mpr-ivl"
    rm -rf "${W:?}/mpr-ivl/followup_rel_4lang/expA_ivl25" "${W:?}/mpr-ivl/followup_rel_4lang/results"
    printf '#!/bin/bash\nPYTHONPATH="%s/mpr-ivl/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$W" "$S" > "$W/mpr-ivl/.venv/bin/python.new"
    rm -f "$W/mpr-ivl/.venv/bin/python" "$W/mpr-ivl/.venv/bin/activate" "$W/mpr-ivl/.venv/bin/python3"
    mv "$W/mpr-ivl/.venv/bin/python.new" "$W/mpr-ivl/.venv/bin/python"; chmod +x "$W/mpr-ivl/.venv/bin/python"; ln -s python "$W/mpr-ivl/.venv/bin/python3"
    printf 'export PATH="%s/mpr-ivl/.venv/bin:$PATH"\n' "$W" > "$W/mpr-ivl/.venv/bin/activate"
    mkdir -p "$W/mpr-ivl/.cache"; cp -R "$FAKE_HF" "$W/mpr-ivl/.cache/huggingface"
    ( cd "$W/mpr-ivl" && tar -xzf "$ADDON25" && mkdir -p followup_rel_6lang && cp -Rl "$FRJA" followup_rel_6lang/assets )   # server state after the InternVL2.5 REL6 queue
    ( cd "$W/mpr-ivl" && find followup_rel_4lang -type f | sort > "$W/files4_before.txt"; shasum -a 256 01_code/src/mpr_crosslocale/inference/run_rel_followup.py 01_code/src/mpr_crosslocale/data/rel_followup.py 01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py > "$W/orig_sha_before.txt" )
  fi
}
q() { ( cd "$W" && W="$W" bash queue_b2_step1_ivl35_rel.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_B2_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_B2_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
RL() { echo "$W/queue_results/b2_step1_ivl35_rel/logs"; }
t_plan() { echo "### plan"; mkws plan stub; q --plan
  ck "plan states 26,057 / 29148 / no --strict-batch-size" bash -c "grep -q '26,057' $W/out.txt && grep -q 29148 $W/out.txt && grep -q 'no --strict-batch-size' $W/out.txt"
  ck "plan labels the ETA an estimate" has "$W/out.txt" 'ETA (estimate)'
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE_B2_STATUS ] && [ ! -e $W/QUEUE_B2_DONE ] && [ ! -d $W/queue_results ]"
  ck "bash -n" bash -n "$M/queue_b2_step1_ivl35_rel.sh"; }
t_real() { echo "### real flow (mock model)"; mkws real real; q
  local R_=$W/mpr-ivl
  ck "QUEUE_B2_DONE = OK" done_is OK
  ck "setup log: venv/code/system prompt/config equal earlier run" has "$(RL)/setup.log" 'equal the earlier InternVL3.5 REL run'
  ck "ivl25 REL6 files verified by SHA256SUMS" has "$(RL)/setup.log" 'internvl2_5_8b.yaml: OK'
  ck "model ok, label ids 32..35" has "$(RL)/model.log" "'A': \[32\]"
  ck "smoke plan: 160 rows / 112 inferences" bash -c "grep -q '\"condition_rows\": 160' $R_/$OUTR/smoke/plan.json && grep -q '\"actual_inferences\": 112' $R_/$OUTR/smoke/plan.json"
  ck "smoke gate line in status" has "$W/QUEUE_B2_STATUS" 'smoke gate: acc'
  ck "full gate passed, 26,057 rows" bash -c "grep -q '\"status\": \"passed\"' $R_/$OUTR/full/gate_rel_complete.json && grep -q '\"rows\": 26057' $R_/$OUTR/full/gate_rel_complete.json"
  ck "summary: 29,148 condition rows all success" bash -c "grep -q '\"condition_rows\": 29148' $R_/$OUTR/full/summary.json && grep -q '\"success\": 29148' $R_/$OUTR/full/summary.json"
  ck "run contract: batch 1, strict false, internvl3_5 eager, new_only" bash -c "python3 -c \"import json;d=json.load(open('$R_/$OUTR/full/run_contract.json'));assert d['batch_size']==1 and d['strict_batch_size']==False and d['model_config']['model_family']=='internvl3_5' and d['model_config']['attn_implementation']=='eager' and d['languages_subset']=='new_only'\""
  ck "fr and ja in gate language report" bash -c "grep -q '\"fr\"' $R_/$OUTR/full/gate_rel_complete.json && grep -q '\"ja\"' $R_/$OUTR/full/gate_rel_complete.json"
  ck "package + sha256 + per-file list" bash -c "cd $W/queue_results/b2_step1_ivl35_rel && ls b2_step1_ivl35_rel_results_*.tar.gz && shasum -a 256 -c b2_step1_ivl35_rel_results_*.tar.gz.sha256 && tar -tzf b2_step1_ivl35_rel_results_*.tar.gz | grep -q results_files.sha256"
  ck "model cache deleted after success" test ! -d "$R_/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B"
  ck "4-language folder file list unchanged, originals byte-identical" bash -c "cd $R_ && find followup_rel_4lang -type f | sort | diff -q - $W/files4_before.txt && shasum -a 256 -c $W/orig_sha_before.txt"
  ck "result folder new, not the InternVL2.5 one" bash -c "[ ! -e $R_/followup_rel_6lang/results/rel6_internvl25_8b ] && [ -d $R_/$OUTR ]"
  local n; n=$(wc -l < "$R_/$OUTR/full/predictions.jsonl"); mkdir -p "$R_/.cache"; cp -R "$FAKE_HF/hub" "$R_/.cache/huggingface/" 2>/dev/null; q
  ck "rerun: OK, 'already done', no new predictions" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ] && grep -q 'already done' $(RL)/full.log && [ $n -eq \$(wc -l < $R_/$OUTR/full/predictions.jsonl) ]"
  mkws keep real; KEEP_MODEL_CACHE=1 q; ck "KEEP_MODEL_CACHE=1 keeps the cache" test -d "$W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B"; }
t_resume() { echo "### resume"; mkws res real; q; local R_=$W/mpr-ivl
  head -n 9000 "$R_/$OUTR/full/predictions.jsonl" > "$W/head.jsonl"; cp "$W/head.jsonl" "$R_/$OUTR/full/predictions.jsonl"
  rm -f "${R_:?}/${OUTR:?}/full/gate_rel_complete.json" "${R_:?}/${OUTR:?}/full/results.jsonl" "${R_:?}/${OUTR:?}/full/summary.json"
  mkdir -p "$R_/.cache/huggingface"; cp -R "$FAKE_HF/hub" "$R_/.cache/huggingface/" 2>/dev/null; q
  ck "second run OK; 26,057 unique rows; first 9,000 lines unchanged" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = OK ] && [ \$(wc -l < $R_/$OUTR/full/predictions.jsonl) -eq 26057 ] && head -n 9000 $R_/$OUTR/full/predictions.jsonl | cmp - $W/head.jsonl"; }
t_guards() { echo "### guards"
  mkws g1 real; FAKE_GPU_APPS="4242, python, 9000 MiB" q; ck "GPU process: FAILED steps: gpu, nothing started" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: gpu' ] && ! grep -q 'step setup started' $W/QUEUE_B2_STATUS"
  mkws g2 real; FAKE_GPU_USED=20000 q; ck "GPU memory used: refused" done_starts "FAILED steps: gpu"
  mkws g3 real; FAKE_GPU_NAME="NVIDIA A100" q; ck "not a 4090: refused" done_starts "FAILED steps: gpu"
  mkws g4 real; mkdir -p "$S/b2/fp"; printf '#!/bin/bash\nsleep 40\n' > "$S/b2/fp/queue_other_test.sh"; bash "$S/b2/fp/queue_other_test.sh" & local p=$!; sleep 1; q; kill $p 2>/dev/null; wait $p 2>/dev/null
  ck "another queue script running: refused (others)" done_is "FAILED steps: others"
  mkws g5 real; FAKE_NET_HF="6000000 10.0 206" q; ck "slow network: STOP network, no setup" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: network' ] && ! grep -q 'step setup started' $W/QUEUE_B2_STATUS"
  mkws g6 real; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q; ck "FORCE_SLOW_NET=1 continues to OK" done_is OK
  mkws g7 real; rm -rf "${W:?}/mpr-ivl/.cache/huggingface/hub"; FAKE_FREE=10 q; ck "10 GB free, no cache: STOP disk" done_starts "FAILED steps: disk"
  mkws g8 real; FAKE_FREE=10 q; ck "10 GB free but cached: continues" done_is OK
  mkws g9 real; rm -rf "${W:?}/mpr-ivl/.venv"; q; ck "no venv: STOP repo" done_is "FAILED steps: repo"; }
t_setup() { echo "### setup negatives"
  mkws s1 real; cp --help >/dev/null 2>&1; rm -f "$W/mpr-ivl/followup_rel_6lang/ivl25_rel6/check_rel6_gate.py.new"; { cat "$W/mpr-ivl/followup_rel_6lang/ivl25_rel6/check_rel6_gate.py"; echo "# x"; } > "$W/g.py"; rm -f "$W/mpr-ivl/followup_rel_6lang/ivl25_rel6/check_rel6_gate.py"; cp "$W/g.py" "$W/mpr-ivl/followup_rel_6lang/ivl25_rel6/check_rel6_gate.py"; q
  ck "tampered ivl25 REL6 gate file: setup FAILED, no model step" bash -c "grep -q 'FAILED steps: setup' $W/QUEUE_B2_DONE && ! grep -q 'step model started' $W/QUEUE_B2_STATUS"
  mkws s2 real; f="$W/mpr-ivl/01_code/src/mpr_crosslocale/inference/system_prompts.py"; sed 's/def build_system_prompt/def build_system_promptX/' "$f" > "$W/sp.py"; rm -f "$f"; cp "$W/sp.py" "$f"; q; ck "repo code differs from earlier run: setup FAILED" done_starts "FAILED steps: setup"
  mkws s3 real; EXPECT_TORCH=9.9.9 q; ck "wrong torch version: setup FAILED, nothing installed" done_starts "FAILED steps: setup"
  mkws s4 real; f=$(ls $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B/snapshots/*/model-00004*); rm -f "$f"; truncate -s 100 "$f"; q; ck "wrong safetensors size: model FAILED" done_starts "FAILED steps: model"
  mkws s5 real; MODEL_MIN_BYTES=17070000001 q; ck "size one byte below the allowed range: model FAILED" done_starts "FAILED steps: model"
  mkws s6 real; rm -rf "${W:?}/mpr-ivl/followup_rel_4lang/assets/images/1/ru"; q; ck "missing 4-language image: setup FAILED (offline)" done_starts "FAILED steps: setup"
  mkws s7 real; f=$(ls $W/mpr-ivl/followup_rel_6lang/assets/images/*/fr/* | head -1); rm -f "$f"; echo bad > "$f"; q; ck "corrupted fr image (sha256): setup FAILED" done_starts "FAILED steps: setup"
  mkws s8 real; sed 's/^CFG_SHA=.*/CFG_SHA=0000/' "$W/queue_b2_step1_ivl35_rel.sh" > "$W/qq.sh"; rm -f "$W/queue_b2_step1_ivl35_rel.sh"; cp "$W/qq.sh" "$W/queue_b2_step1_ivl35_rel.sh"; q; ck "config sha differs: setup FAILED" done_starts "FAILED steps: setup"; }
t_stub() { echo "### stub flow"
  mkws t1 stub; q; ck "stub success OK" done_is OK
  mkws t2 stub; FAKE_FAIL_STEP=smoke q; ck "smoke fails: full not run, FAILED steps: smoke" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: smoke' ] && ! grep -q full $W/mpr-ivl/steps_run.txt"
  mkws t3 stub; FAKE_FAIL_STEP=full_eta q; ck "ETA stop (exit 3): FAILED steps: full_eta + message" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: full_eta' ] && grep -q 'smoke ETA longer' $W/QUEUE_B2_STATUS"
  mkws t4 stub; FAKE_FAIL_STEP=full_sleep FAKE_TIMEOUT_MATCH="full" FAKE_TIMEOUT_SECS=2 q; ck "timeout reported" has "$W/QUEUE_B2_STATUS" 'step full TIMEOUT'
  mkws t5 stub; FAKE_FAIL_STEP=full q; ck "full fails: cache kept, FAILED steps: full" bash -c "[ \"\$(cat $W/QUEUE_B2_DONE)\" = 'FAILED steps: full' ] && [ -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B ]"
  mkws t6 stub; FAIL_DL_FIRST=2 q; ck "2 transient download failures then OK" done_is OK
  mkws t7 stub; FAIL_DL_FIRST=3 q; ck "3 download failures: FAILED steps: model" done_is "FAILED steps: model"
  mkws t8 stub; ( cd "$W" && W="$W" FAKE_FAIL_STEP=smoke_sleep bash queue_b2_step1_ivl35_rel.sh > out1.txt 2>&1 ) & local p=$!; sleep 4; q
  ck "duplicate start refused (already running), DONE untouched" bash -c "grep -q 'already running' $W/out.txt && [ \$(cat $W/rc.txt) = 1 ] && [ ! -e $W/QUEUE_B2_DONE ]"
  kill -TERM $p 2>/dev/null; pkill -TERM -P $p 2>/dev/null; wait $p 2>/dev/null
  ck "static: no pip install in the queue" bash -c "! grep -q 'pip install' $M/queue_b2_step1_ivl35_rel.sh"
  ck "static: no --strict-batch-size in the REL command" bash -c "! grep 'python followup_rel_6lang/run_inference_6lang.py' $M/queue_b2_step1_ivl35_rel.sh | grep -q strict-batch-size"; }
case "${1:-all}" in all) for t in plan real resume guards setup stub; do "t_$t"; done ;; *) "t_$1" ;; esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
