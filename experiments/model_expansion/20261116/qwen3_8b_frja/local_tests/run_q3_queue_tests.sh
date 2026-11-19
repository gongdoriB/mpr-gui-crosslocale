#!/usr/bin/env bash
# Flow tests for queue_qwen3_8b_frja.sh in a fake /workspace (W=...). Nothing leaves this machine: no GPU, no pip, no model download, no network
# (https_proxy points to a closed port as a second safety net; the fake curl/hf answer instead of the real ones).
#  "real" tests: real setup() (addon + fr/ja image tar unpack, verification, venv checks, code-identity check), real model() (offline, against a fake HF cache: tokenizer files of the
#                local Qwen3-VL-4B snapshot as a STAND-IN, config rewritten to the 8B architecture, sparse safetensors of 17.5 GB), real run scripts with --mock-model (LOCAL_MOCK=1), real gates.
#  "stub" tests: setup/model/smoke/consistency/full/retry replaced by stubs (make_stub_queue_q3.py) to exercise the GPU check, network check, retry, timeout, ETA exit code,
#                non-fatal consistency step, order of steps, cache rule, packaging, duplicate start.
# Stand-ins in shims/: hf, nvidia-smi, curl, timeout, df, flock, sha256sum, python3.
# usage: run_q3_queue_tests.sh <name>|all     names: plan real batch resume stubok gpu net fail setup imgtar sigint dup cache
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
TEMPLATE_REPO=$S/q3frja/template     # tools/make_q3_template.sh: Qwen repo stand-in (patches reversed), 4-language images hard-linked, venv stub
FAKE_HF=$S/q3frja/hf_fake            # make_fake_hf.py
FRJA=${CUAI_ROOT}/CUAI_하계\ 컨퍼런스/followup_rel_6lang/assets   # source of the fr/ja images (read only; hard links)
IMGTAR=${CUAI_ROOT}/CUAI_하계\ 컨퍼런스/model_expansion_20261116/internvl25_rel6/internvl25_rel6_frja_images_20261116.tar
WS=$S/q3frja/ws
export PIP_INDEX_URL=http://127.0.0.1:9/simple PIP_EXTRA_INDEX_URL= PIP_RETRIES=0 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 POLL_S=1 HF_HUB_OFFLINE=1 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 EXPECT_TV=0.24.1 EXPECT_SKIP=qwen-vl-utils
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
RR=followup_rel_6lang/results/rel6new_qwen3_vl_8b
RC=followup_rel_6lang/results/consistency_qwen3_vl_8b
RX=followup_rel_6lang/results/expA6_qwen3_vl_8b
mkws() {  # $1 name $2 = real|stub  [$3 = tar  -> also link the fr/ja image tar]
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state; rm -rf "$W"; mkdir -p "$W" "$FAKE_STATE"
  if [ "$2" = stub ]; then python3 "$HERE/make_stub_queue_q3.py" "$M/queue_qwen3_8b_frja.sh" "$W/queue_qwen3_8b_frja.sh"
  else cp "$M/queue_qwen3_8b_frja.sh" "$W/"; fi
  ln "$M/qwen3_8b_frja_addon_20261116.tar.gz" "$W/"
  if [ "${3:-}" = tar ]; then ln "$IMGTAR" "$W/internvl25_rel6_frja_images_20261116.tar"; fi
  if [ "$2" = real ]; then
    cp -Rl "$TEMPLATE_REPO" "$W/mpr-gui-crosslocale"
    printf '#!/bin/bash\nPYTHONPATH="%s/mpr-gui-crosslocale/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$W" "$S" > "$W/mpr-gui-crosslocale/.venv/bin/python.new"
    rm "$W/mpr-gui-crosslocale/.venv/bin/python" "$W/mpr-gui-crosslocale/.venv/bin/activate" "$W/mpr-gui-crosslocale/.venv/bin/python3" 2>/dev/null
    mv "$W/mpr-gui-crosslocale/.venv/bin/python.new" "$W/mpr-gui-crosslocale/.venv/bin/python"; chmod +x "$W/mpr-gui-crosslocale/.venv/bin/python"; ln -s python "$W/mpr-gui-crosslocale/.venv/bin/python3"
    printf 'export PATH="%s/mpr-gui-crosslocale/.venv/bin:$PATH"\n' "$W" > "$W/mpr-gui-crosslocale/.venv/bin/activate"
    mkdir -p "$W/mpr-gui-crosslocale/.cache"; cp -Rl "$FAKE_HF" "$W/mpr-gui-crosslocale/.cache/huggingface"
    if [ "${3:-}" != tar ]; then   # no image tar: the images are "already there" (hard links), as after an earlier fetch
      mkdir -p "$W/mpr-gui-crosslocale/followup_rel_6lang"; cp -Rl "$FRJA" "$W/mpr-gui-crosslocale/followup_rel_6lang/assets"
    fi
    ( cd "$W/mpr-gui-crosslocale" && find followup_rel_4lang -type f | sort > "$W/files4_before.txt"; shasum -a 256 01_code/src/mpr_crosslocale/inference/run_rel_followup.py 01_code/src/mpr_crosslocale/data/rel_followup.py 01_code/src/mpr_crosslocale/inference/run_cross_locale.py followup_rel_4lang/expA/run_rel_T.py > "$W/orig_sha_before.txt" )
  else
    mkdir -p "$W/mpr-gui-crosslocale/.venv/bin" "$W/mpr-gui-crosslocale/01_code"; touch "$W/mpr-gui-crosslocale/.venv/bin/activate"
  fi
}
q() { ( cd "$W" && W="$W" bash queue_qwen3_8b_frja.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_C_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_C_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
ST() { echo "$W/QUEUE_C_STATUS"; }
R() { echo "$W/mpr-gui-crosslocale"; }
RES() { echo "$W/queue_results/qwen3_8b_frja"; }

t_plan() {
  echo "### plan: --plan prints and changes nothing"
  mkws plan stub; q --plan
  ck "plan lists consistency, REL, Exp A and the numbers (26,057 / 9180 / 477)" bash -c "grep -q 'consistency check' $W/out.txt && grep -q '26,057' $W/out.txt && grep -q '9180' $W/out.txt && grep -q '477' $W/out.txt"
  ck "plan states the batch sizes (REL 1, Exp A 3) and the disk budget" bash -c "grep -q 'batch 1' $W/out.txt && grep -q 'batch 3' $W/out.txt && grep -q 'Disk (65 GB' $W/out.txt"
  ck "plan states estimates as estimates" has "$W/out.txt" 'estimate, not measured'
  ck "plan creates no status/done/result files" bash -c "[ ! -e $W/QUEUE_C_STATUS ] && [ ! -e $W/QUEUE_C_DONE ] && [ ! -d $W/queue_results ]"
  mkws plan2 stub; REL_BATCH=4 EXPA_BATCH=1 EXPA_SPLIT=main q --plan
  ck "env batch/split reflected in the plan (REL 4, Exp A 1, 8532)" bash -c "grep -q 'batch 4' $W/out.txt && grep -q '8532' $W/out.txt"
  ck "bash -n on every shipped script" bash -c 'cd "$1" && bash -n queue_qwen3_8b_frja.sh && for f in addon/followup_rel_6lang/q3_frja/*.sh build_addon.sh; do bash -n "$f" || exit 1; done' _ "$M"
  mkws plan3 stub; REL_BATCH=9 q
  ck "REL_BATCH=9 refused" bash -c "[ \$(cat $W/rc.txt) = 2 ]"
  mkws plan4 stub; EXPA_BATCH=2 q
  ck "EXPA_BATCH=2 refused" bash -c "[ \$(cat $W/rc.txt) = 2 ]"
}
t_real() {
  echo "### real: real setup (addon + image tar) / model (offline fake cache) / smokes / consistency / REL full / Exp A full / T_retry with --mock-model, real gates, real packaging"
  mkws real real tar; q
  local R_; R_=$(R); local RS; RS=$(RES)
  ck "QUEUE_C_DONE = OK" done_is OK
  ck "GPU idle line" has "$(ST)" 'GPU idle'
  ck "network line with MB/s" has "$(ST)" 'huggingface.co 40.0 MB/s'
  ck "setup ok" has "$(ST)" 'step setup ok'
  ck "setup log: 12 inference-code hashes + config equal the 2026-11-07 contract" has "$RS/logs/setup.log" '12 inference-code hashes and the model config equal'
  ck "setup log: venv ok with the pinned package list" has "$RS/logs/setup.log" 'venv ok:'
  ck "model ok (real checks: 4 shards, byte range, Qwen3VL architecture, A-D ids 32-35)" has "$(ST)" 'step model ok'
  ck "model log shows label ids" has "$RS/logs/model.log" "'A': \[32\]"
  ck "image tar unpacked (729 fr/ja images)" bash -c "[ \$(find $R_/followup_rel_6lang/assets -type f | wc -l) -eq 729 ]"
  ck "6-language modules import" has "$RS/logs/setup.log" '6-language modules import ok'
  ck "bundle verify passed (conditions sha256 locked)" has "$R_/$RR/logs/bundle_verify.log" '63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed'
  ck "preflight: config/system prompt/12 code hashes equal contract (run script)" has "$RS/logs/smoke_rel.log" 'equal the 2026-11-07 Qwen3-VL-8B REL contract'
  ck "legacy subset check ran on the real 4-language file (not skipped)" bash -c "! grep -q skipped $R_/$RR/logs/legacy_subset_check.log"
  ck "REL smoke gate line" has "$(ST)" 'smoke gate: rel fr/ja: acc'
  ck "Exp A smoke gate line" has "$(ST)" 'smoke gate: expA fr/ja: acc'
  ck "REL smoke plan: --max-items 2 -> 160 condition rows / 112 inferences (new_only)" bash -c "grep -q '\"condition_rows\": 160' $R_/$RR/smoke/plan.json && grep -q '\"actual_inferences\": 112' $R_/$RR/smoke/plan.json"
  ck "Exp A smoke plan: 162 inferences" bash -c "grep -q '\"inferences\": 162' $R_/$RX/smoke/plan.json"
  ck "order of status lines: smoke_rel < smoke_expa < consistency < full_rel < full_expa < retry_expa" bash -c "
    n() { grep -n \"\$1\" $W/QUEUE_C_STATUS | head -1 | cut -d: -f1; }
    a=\$(n 'step smoke_rel started'); b=\$(n 'step smoke_expa started'); c=\$(n 'step consistency started'); d=\$(n 'step full_rel started'); e=\$(n 'step full_expa started'); f=\$(n 'step retry_expa started')
    [ \$a -lt \$b ] && [ \$b -lt \$c ] && [ \$c -lt \$d ] && [ \$d -lt \$e ] && [ \$e -lt \$f ]"
  ck "consistency: 600 rows, own folder, agreement json present" bash -c "test -s $R_/$RC/consistency_agreement.json && grep -q '\"actual_inferences\": 600' $R_/$RC/run/plan.json && test ! -e $R_/$RR/consistency_agreement.json"
  ck "consistency status line (label agreement, accuracy old/new)" has "$(ST)" 'consistency: label agreement'
  ck "REL full: 26,057 inferences / 29,148 rows, gate passed" bash -c "grep -q '\"status\": \"passed\"' $R_/$RR/full/gate_rel_complete.json && grep -q '\"rows\": 26057' $R_/$RR/full/gate_rel_complete.json && grep -q '\"success\": 29148' $R_/$RR/full/summary.json"
  ck "REL full gate lists fr and ja per language" bash -c "grep -q '\"fr\"' $R_/$RR/full/gate_rel_complete.json && grep -q '\"ja\"' $R_/$RR/full/gate_rel_complete.json"
  ck "REL full: every batch size is 1" bash -c "grep -q '\"batch_sizes\": {' $R_/$RR/full/gate_rel_complete.json && python3 -c \"import json;d=json.load(open('$R_/$RR/full/gate_rel_complete.json'));assert d['batch_sizes']=={'1':26057},d['batch_sizes']\""
  ck "Exp A full: 9,180 rows, gate passed, anchors compared with the REL run" bash -c "python3 -c \"import json;d=json.load(open('$R_/$RX/full/gate_complete.json'));assert d['status']=='passed' and d['rows']==9180 and d['anchor_vs_rel_new_only_run']['anchor_rows']==6120,d\""
  ck "T_retry: 477 rows, gate passed" bash -c "python3 -c \"import json;d=json.load(open('$R_/$RX/retry/gate_complete.json'));assert d['status']=='passed' and d['rows']==477,d\""
  ck "addon files unpacked in repo" test -s "$R_/followup_rel_6lang/q3_frja/run_rel6_qwen3.sh"
  ck "results tar + sha256 + per-file list inside" bash -c "ls $RS/qwen3_8b_frja_results_*.tar.gz && ls $RS/*.sha256 && tar -tzf $RS/qwen3_8b_frja_results_*.tar.gz | grep -q results_files.sha256 && tar -tzf $RS/qwen3_8b_frja_results_*.tar.gz | grep -q 'rel6new_qwen3_vl_8b/full/predictions.jsonl' && tar -tzf $RS/qwen3_8b_frja_results_*.tar.gz | grep -q 'consistency_qwen3_vl_8b/consistency_agreement.json' && tar -tzf $RS/qwen3_8b_frja_results_*.tar.gz | grep -q 'expA6_qwen3_vl_8b/retry/predictions.jsonl'"
  ck "results tar sha256 file verifies" bash -c "cd $RS && shasum -a 256 -c qwen3_8b_frja_results_*.tar.gz.sha256"
  ck "model cache of this model KEPT after success (default)" test -d "$R_/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  ck "status says cache kept" has "$(ST)" 'model cache KEPT'
  ck "4-language folder: same file list as before (no result written there)" bash -c "cd $R_ && find followup_rel_4lang -type f | sort | diff -q - $W/files4_before.txt"
  ck "no results folder under followup_rel_4lang" test ! -e "$R_/followup_rel_4lang/results"
  ck "original runner/data/code files and the old Exp A runner byte-identical after the run" bash -c "cd $R_ && shasum -a 256 -c $W/orig_sha_before.txt"
  local before; before=$(wc -l < "$R_/$RR/full/predictions.jsonl")
  q
  ck "rerun: QUEUE_C_DONE OK" done_is OK
  ck "rerun: REL full already done / Exp A already done / retry already done" bash -c "grep -q 'REL fr/ja full already done' $R_/$RR/STATUS && grep -q 'Exp A fr/ja full already done' $R_/$RX/STATUS && grep -q 'Exp A T_retry already done' $R_/$RX/STATUS"
  ck "rerun: no prediction added" bash -c "[ $before -eq \$(wc -l < $R_/$RR/full/predictions.jsonl) ]"
}
t_resume() {
  echo "### resume: a half-finished REL full run (cut predictions, no gate) is completed by the same command; the earlier predictions are kept"
  mkws resume real tar; q
  local R_; R_=$(R)
  ck "first run OK" done_is OK
  head -n 9000 "$R_/$RR/full/predictions.jsonl" > "$W/pred_head.jsonl"; head -n 9000 "$R_/$RR/full/predictions.jsonl" | tail -1 > "$W/last_kept.txt"
  cp "$W/pred_head.jsonl" "$R_/$RR/full/predictions.jsonl"; rm -f "$R_/$RR/full/gate_rel_complete.json" "$R_/$RR/full/results.jsonl" "$R_/$RR/full/summary.json"
  q
  ck "second run OK" done_is OK
  ck "REL full completed: 26,057 unique rows, first 9,000 lines unchanged" bash -c "[ \$(wc -l < $R_/$RR/full/predictions.jsonl) -eq 26057 ] && head -n 9000 $R_/$RR/full/predictions.jsonl | cmp - $W/pred_head.jsonl"
  ck "gate passed again" bash -c "grep -q '\"status\": \"passed\"' $R_/$RR/full/gate_rel_complete.json"
  # a different batch size in the same folder is refused (fingerprint)
  rm -f "$R_/$RR/smoke/gate_rel_smoke.json"   # smoke gate removed so that the smoke stage runs again, now with batch 4 in the folder of the batch-1 run
  ( cd "$R_" && REL_BATCH=4 LOCAL_MOCK=1 EXPECT_TORCH=2.9.1 bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh smoke ) > "$W/b4.txt" 2>&1
  ck "same folder with another batch size (smoke dir reused): refused by the fingerprint" has "$W/b4.txt" 'belongs to a different'
}
t_batch() {
  echo "### batch: REL_BATCH=4 (strict) / EXPA_BATCH=1 / EXPA_SPLIT=main / CC_BATCH separate; SKIP_CONSISTENCY"
  mkws batch real; REL_BATCH=4 EXPA_BATCH=1 EXPA_SPLIT=main CC_BATCH=2 SKIP_CONSISTENCY=1 q
  local R_; R_=$(R)
  ck "QUEUE_C_DONE = OK" done_is OK
  ck "consistency skipped and said so" has "$(ST)" 'consistency check skipped'
  ck "REL batch 4: plan batch_size 4, 26,056 rows in batches of 4 + 1 tail batch of 1 (26,057 = 4 x 6,514 + 1), gate passed" bash -c "python3 -c \"import json;p=json.load(open('$R_/$RR/full/plan.json'));g=json.load(open('$R_/$RR/full/gate_rel_complete.json'));assert p['batch_size']==4 and g['batch_sizes']=={'4':26056,'1':1} and g['expected_batch_size']==4 and g['status']=='passed',(p,g['batch_sizes'])\""
  ck "REL batch 4: contract has strict_batch_size true" bash -c "grep -q '\"strict_batch_size\": true' $R_/$RR/full/run_contract.json"
  ck "Exp A batch 1, split main: 8,532 rows, every batch size 1, gate passed" bash -c "python3 -c \"import json;g=json.load(open('$R_/$RX/full/gate_complete.json'));assert g['rows']==8532 and set(g['batch_sizes'])=={'1'} and g['status']=='passed',g['batch_sizes']\""
  ck "T_retry split main: 444 rows" bash -c "python3 -c \"import json;g=json.load(open('$R_/$RX/retry/gate_complete.json'));assert g['rows']==444,g['rows']\""
  ck "consistency folder not created when skipped" test ! -d "$R_/$RC"
  mkws batch2 real; CC_BATCH=2 q
  R_=$(R)
  ck "CC_BATCH=2: consistency run used batch 2 (gate checks it) and compare json says batch size 2" bash -c "python3 -c \"import json;d=json.load(open('$R_/$RC/consistency_agreement.json'));assert d['new_run']['batch_sizes']=={'2':600},d['new_run']['batch_sizes']\""
}
t_gpu() {
  echo "### gpu: a process on the GPU, or memory in use -> refuses at once, nothing started"
  mkws gpu1 stub; FAKE_GPU_APPS="4242, python, 9000 MiB" q
  ck "compute process present: FAILED steps: gpu" done_starts "FAILED steps: gpu"
  ck "status names the process" has "$(ST)" 'processes: 4242, python'
  ck "no step started" test ! -e "$(R)/steps_run.txt"
  ck "it did not wait (finished within seconds, no 'waiting')" bash -c "! grep -qi waiting $(ST)"
  mkws gpu2 stub; FAKE_GPU_USED=20000 q
  ck "GPU memory 20000 MiB in use, no listed process: FAILED steps: gpu" done_starts "FAILED steps: gpu"
  mkws gpu3 stub; q
  ck "control: idle GPU starts" done_is OK
  mkws gpu4 stub; FAKE_GPU_NAME="NVIDIA A100" q
  ck "not an RTX 4090: FAILED steps: gpu" done_starts "FAILED steps: gpu"
  mkws gpu5 stub; FAKE_GPU_USED=1400 q
  ck "1400 MiB (below the 1500 MiB limit) does not block" done_is OK
}
t_net() {
  echo "### net: slow -> STOP; FORCE_SLOW_NET=1 continues; HTTP 404 -> warning only; unreachable -> STOP"
  mkws net stub; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow: FAILED steps: network" done_starts "FAILED steps: network"
  ck "MB/s in status" has "$(ST)" 'huggingface.co 0.6 MB/s'
  ck "nothing downloaded or set up" test ! -d "$(R)/.cache"
  mkws net2 stub; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1 continues" done_is OK
  mkws net3 stub; FAKE_NET_HF="0 1.0 404" q
  ck "404: warning, continues" done_is OK
  mkws net4 stub; FAKE_NET_HF="0 1.0 000" q
  ck "unreachable: STOP" done_starts "FAILED steps: network"
}
t_stubok() {
  echo "### stubok: step order, success verdict, nothing skipped"
  mkws ok stub; q
  ck "QUEUE_C_DONE = OK" done_is OK
  ck "six stub steps ran in the order smoke_rel smoke_expa consistency full_rel full_expa retry_expa" bash -c "[ \"\$(tr '\n' ' ' < $(R)/steps_run.txt)\" = 'smoke_rel smoke_expa consistency full_rel full_expa retry_expa ' ]"
  ck "no waiting for any other queue (no MINICPM/QUEUE_DONE needed)" bash -c "! grep -qi 'waiting for' $(ST)"
  mkws ok2 stub; FAKE_FAIL_STEP=smoke_rel q
  ck "REL smoke fails: REL full and consistency skipped, Exp A still runs" bash -c "[ \"\$(tr '\n' ' ' < $(R)/steps_run.txt)\" = 'smoke_rel smoke_expa full_expa retry_expa ' ]"
  ck "verdict FAILED steps: smoke_rel" done_is "FAILED steps: smoke_rel"
  ck "status reports the failed REL smoke" has "$(ST)" 'REL smoke failed'
  mkws ok3 stub; FAKE_FAIL_STEP=smoke_expa q
  ck "Exp A smoke fails: Exp A full/retry skipped, REL still runs" bash -c "[ \"\$(tr '\n' ' ' < $(R)/steps_run.txt)\" = 'smoke_rel smoke_expa consistency full_rel ' ]"
  ck "verdict FAILED steps: smoke_expa" done_is "FAILED steps: smoke_expa"
  mkws ok4 stub; FAKE_FAIL_STEP=consistency q
  ck "consistency fails: non-fatal; REL and Exp A continue" bash -c "[ \"\$(tr '\n' ' ' < $(R)/steps_run.txt)\" = 'smoke_rel smoke_expa consistency full_rel full_expa retry_expa ' ]"
  ck "verdict is OK (non-fatal: consistency failed) and the exit code is 0" bash -c "[ \"\$(cat $W/QUEUE_C_DONE)\" = 'OK (non-fatal: consistency failed)' ] && [ \$(cat $W/rc.txt) = 0 ]"
  mkws ok5 stub; FAKE_FAIL_STEP=full_rel q
  ck "REL full fails: Exp A still runs; FAILED steps: full_rel" bash -c "[ \"\$(cat $W/QUEUE_C_DONE)\" = 'FAILED steps: full_rel' ] && grep -q full_expa $(R)/steps_run.txt"
  mkws ok6 stub; FAKE_FAIL_STEP=full_expa q
  ck "Exp A full fails: retry not run" bash -c "[ \"\$(cat $W/QUEUE_C_DONE)\" = 'FAILED steps: full_expa' ] && ! grep -q retry_expa $(R)/steps_run.txt"
}
t_fail() {
  echo "### fail: retries, timeout, ETA exit 3"
  mkws fail1 stub; FAIL_DL_FIRST=2 q
  ck "2 transient download failures, then ok (retry x3)" done_is OK
  mkws fail2 stub; FAIL_DL_FIRST=3 q
  ck "3 failures: model step FAILED, nothing after it ran" bash -c "[ \"\$(cat $W/QUEUE_C_DONE)\" = 'FAILED steps: model' ] && [ ! -e $(R)/steps_run.txt ]"
  ck "partial cache kept" test -d "$(R)/.cache/huggingface"
  mkws fail3 stub; FAKE_FAIL_STEP=full_rel_sleep FAKE_TIMEOUT_MATCH="full_rel" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported (step full_rel TIMEOUT)" has "$(ST)" 'step full_rel TIMEOUT'
  mkws fail4 stub; FAKE_FAIL_STEP=full_rel_eta q
  ck "ETA stop (exit 3) -> FAILED steps: full_rel_eta" done_is "FAILED steps: full_rel_eta"
  ck "ETA stop message" has "$(ST)" 'STOP step full_rel'
  mkws fail5 stub; FAKE_FAIL_STEP=full_expa_eta q
  ck "Exp A ETA stop -> FAILED steps: full_expa_eta" done_is "FAILED steps: full_expa_eta"
  ck "static: model() uses retry 3 hf download" grep -q 'retry 3 hf download' "$M/queue_qwen3_8b_frja.sh"
  ck "static: no pip install anywhere in the queue" bash -c "! grep -q 'pip install' $M/queue_qwen3_8b_frja.sh"
}
t_cache() {
  echo "### cache: kept by default; DELETE_MODEL_CACHE=1 deletes after success only"
  mkws c1 stub; q
  ck "default: cache kept on success" test -d "$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  mkws c2 stub; DELETE_MODEL_CACHE=1 q
  ck "DELETE_MODEL_CACHE=1: this model's cache deleted after success" test ! -d "$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  ck "rest of the HF cache dir untouched" test -d "$(R)/.cache/huggingface/hub"
  mkws c3 stub; DELETE_MODEL_CACHE=1 FAKE_FAIL_STEP=full_rel q
  ck "DELETE_MODEL_CACHE=1 but a step failed: cache kept" test -d "$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
}
t_setup() {
  echo "### setup: real setup()/model()/preflight must refuse a wrong repo / config / venv / model (negative cases)"
  mkws s1 real; sed -i '' 's/def build_system_prompt/def build_system_promptX/' "$(R)/01_code/src/mpr_crosslocale/inference/system_prompts.py" 2>/dev/null
  q; ck "repo code differs from the 2026-11-07 contract (system_prompts.py edited): setup FAILED before any download" done_starts "FAILED steps: setup"
  ck "no model step ran" bash -c "! grep -q 'step model started' $(ST)"
  mkws s2 real tar; cp "$(R)/01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml" "$W/y"; rm "$(R)/01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml"; sed 's/max_pixels: 2097152/max_pixels: 1048576/' "$W/y" > "$(R)/01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml"
  q; ck "repo config differs from the addon config (max_pixels): the smoke step FAILED (cmp in preflight), no full run" bash -c "grep -q 'FAILED steps' $W/QUEUE_C_DONE && grep -q smoke_rel $W/QUEUE_C_DONE && ! grep -q full_rel $(R)/steps_run.txt 2>/dev/null"
  mkws s3 real; EXPECT_TORCH=9.9.9 q
  ck "wrong torch version: venv check fails -> setup FAILED, nothing installed" done_starts "FAILED steps: setup"
  mkws s3b real; EXPECT_SKIP="" q
  ck "qwen-vl-utils missing from the venv (EXPECT_SKIP cleared): setup FAILED" done_starts "FAILED steps: setup"
  mkws s4 real; rm -rf "$W/mpr-gui-crosslocale"; q
  ck "no repo: STOP before any step" done_starts "FAILED steps: repo"
  mkws s5 real; sed -i '' 's/^ADDON_SHA=.*/ADDON_SHA=0000/' "$W/queue_qwen3_8b_frja.sh"; q
  ck "addon sha mismatch: STOP" done_starts "FAILED steps: addon"
  ck "addon sha mismatch: nothing unpacked" test ! -d "$(R)/followup_rel_6lang/q3_frja"
  mkws s6 real; rm -rf "$(R)/.cache/huggingface/hub"; FAKE_FREE=10 q
  ck "10 GB free and no cache: STOP (need 30)" done_starts "FAILED steps: disk"
  mkws s6b real; FAKE_FREE=10 q
  ck "10 GB free but the model is already cached: continues (resume case)" done_is OK
  mkws s7 real; f="$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/model-00004-of-00004.safetensors"; rm "$f"; truncate -s 100 "$f"; q
  ck "wrong safetensors size (100 bytes): model step FAILED" done_starts "FAILED steps: model"
  mkws s7b real; f="$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/model-00004-of-00004.safetensors"; rm "$f"; truncate -s 4334000000 "$f"; MODEL_BYTES=17534000001 q
  ck "MODEL_BYTES given and one byte off: model step FAILED" done_starts "FAILED steps: model"
  mkws s7c real; MODEL_BYTES=17534000000 q
  ck "MODEL_BYTES given and exact: OK" done_is OK
  mkws s7d real; c="$(R)/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json"; sed 's/Qwen3VLForConditionalGeneration/Qwen2VLForConditionalGeneration/' "$c" > "$W/c.json"; rm "$c"; cp "$W/c.json" "$c"; q
  ck "wrong architecture in config.json: model step FAILED" done_starts "FAILED steps: model"
  mkws s8 real; rm -rf "$(R)/followup_rel_4lang/assets/images/1/ru"; q
  ck "a 4-language image missing: setup does not silently continue (verify/fetch fails offline)" done_starts "FAILED steps: setup"
  mkws s9 real; q
  ck "control: images pre-placed (no tar) -> OK" done_is OK
  ck "control: status says images will be downloaded when no tar is present" has "$(ST)" 'no internvl25_rel6_frja_images'
  mkws s10 real tar; rm -rf "$(R)/.venv"; q
  ck "no venv: STOP (repo)" done_starts "FAILED steps: repo"
  mkws s11 real; mv "$(R)/followup_rel_4lang/assets/images/1/zh/booking_zh_1.jpg" "$W/x.jpg"; echo corrupt > "$(R)/followup_rel_4lang/assets/images/1/zh/booking_zh_1.jpg"; q
  ck "a corrupted 4-language image (sha256) is detected: setup FAILED" done_starts "FAILED steps: setup"
}
t_imgtar() {
  echo "### imgtar: bad image tar / addon with a path outside its own files"
  mkws it1 real tar; sed -i '' 's/^IMAGES_SHA=.*/IMAGES_SHA=0000/' "$W/queue_qwen3_8b_frja.sh"; q
  ck "image tar sha mismatch: STOP" done_starts "FAILED steps: addon"
  mkws it2 real
  # malicious addon: same content plus one overwrite of an existing module of the repo
  rm -rf "$WS/evil"; mkdir -p "$WS/evil"; tar -xzf "$M/qwen3_8b_frja_addon_20261116.tar.gz" -C "$WS/evil"
  mkdir -p "$WS/evil/01_code/src/mpr_crosslocale/inference"; echo "# overwritten" > "$WS/evil/01_code/src/mpr_crosslocale/inference/run_rel_followup.py"
  rm -f "$W/qwen3_8b_frja_addon_20261116.tar.gz"; tar -czf "$W/qwen3_8b_frja_addon_20261116.tar.gz" -C "$WS/evil" 01_code followup_rel_6lang
  sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$(shasum -a 256 "$W/qwen3_8b_frja_addon_20261116.tar.gz" | cut -d' ' -f1)/" "$W/queue_qwen3_8b_frja.sh"; q
  ck "addon with a file outside its own paths: setup FAILED" done_starts "FAILED steps: setup"
  ck "the existing 4-language runner was NOT overwritten" bash -c "cd $(R) && shasum -a 256 -c $W/orig_sha_before.txt"
  mkws it3 real tar
  rm -rf "$WS/evil2"; mkdir -p "$WS/evil2/followup_rel_6lang/assets/images/1/en"; echo x > "$WS/evil2/followup_rel_6lang/assets/images/1/en/x.png"
  rm -f "$W/internvl25_rel6_frja_images_20261116.tar"; tar -cf "$W/internvl25_rel6_frja_images_20261116.tar" -C "$WS/evil2" followup_rel_6lang
  sed -i '' "s/^IMAGES_SHA=.*/IMAGES_SHA=$(shasum -a 256 "$W/internvl25_rel6_frja_images_20261116.tar" | cut -d' ' -f1)/" "$W/queue_qwen3_8b_frja.sh"; q
  ck "image tar with a non fr/ja path: setup FAILED" done_starts "FAILED steps: setup"
  mkws it4 real
  rm -rf "$WS/evil3"; mkdir -p "$WS/evil3"; tar -xzf "$M/qwen3_8b_frja_addon_20261116.tar.gz" -C "$WS/evil3"; echo tampered >> "$WS/evil3/followup_rel_6lang/q3_frja/common_q3.sh"
  rm -f "$W/qwen3_8b_frja_addon_20261116.tar.gz"; tar -czf "$W/qwen3_8b_frja_addon_20261116.tar.gz" -C "$WS/evil3" 01_code followup_rel_6lang
  sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$(shasum -a 256 "$W/qwen3_8b_frja_addon_20261116.tar.gz" | cut -d' ' -f1)/" "$W/queue_qwen3_8b_frja.sh"; q
  ck "addon whose own SHA256SUMS does not match its files (tampered script): setup FAILED" done_starts "FAILED steps: setup"
}
t_sigint() {
  echo "### sigint: TERM during a step -> no QUEUE_C_DONE"
  mkws sigint stub
  ( cd "$W" && W="$W" FAKE_FAIL_STEP=smoke_rel_sleep bash queue_qwen3_8b_frja.sh > "$W/out.txt" 2>&1 ) & local p=$!
  sleep 4; pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; wait $p 2>/dev/null
  # bash runs its trap after the foreground child (the 30 s stub step) has ended: wait for the status line
  for _ in $(seq 1 45); do has "$(ST)" 'interrupted by signal' && break; sleep 1; done
  ck "no QUEUE_C_DONE" test ! -e "$W/QUEUE_C_DONE"
  ck "status says interrupted" has "$(ST)" 'interrupted by signal'
}
t_dup() {
  echo "### dup: second copy refuses (flock)"
  mkws dup stub
  ( cd "$W" && W="$W" FAKE_FAIL_STEP=smoke_rel_sleep bash queue_qwen3_8b_frja.sh > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 4; q
  ck "second run says already running" has "$W/out.txt" 'already running'
  ck "second run exit code 1 and it did not touch QUEUE_C_DONE" bash -c "[ \$(cat $W/rc.txt) = 1 ] && [ ! -e $W/QUEUE_C_DONE ]"
  kill -TERM $p 2>/dev/null; pkill -TERM -P $p 2>/dev/null; wait $p 2>/dev/null
}
case "${1:-}" in
  all) for t in plan stubok gpu net fail cache setup imgtar sigint dup real batch resume; do "t_$t"; done ;;
  plan|real|batch|resume|stubok|gpu|net|fail|cache|setup|imgtar|sigint|dup) "t_$1" ;;
  *) echo "usage: $0 <plan|real|batch|resume|stubok|gpu|net|fail|cache|setup|imgtar|sigint|dup|all>"; exit 2 ;;
esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
