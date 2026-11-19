#!/usr/bin/env bash
# Flow tests for queue_rstem6_c.sh in a fake /workspace (W=...). No GPU, no pip, no model, no network (fake curl/hf/nvidia-smi/df/timeout/flock in shims/).
# Heavy per-model steps (setup, model, smoke, full) are stubs (make_stub_queue.py); everything else is the real script (checks, step/timeout/ETA handling, failure rules, packaging, cache deletion).
# NOT run here (미검증): the real setup (git clone, patches, pip, image copy/check), the real model download + tokenizer check, real GPU inference. The runner/gate/env-check have their own tests.
# usage: run_queue_tests.sh all        env: TESTWS (scratch dir for the fake workspaces)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"; export M
WS=${TESTWS:?set TESTWS to a scratch dir}
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 GPU_RECHECK_TRIES=2 GPU_RECHECK_S=0
# the dev Mac still has old queue_internvl25_rel6.sh processes from earlier tests: the "other queue" rule only looks for our own marker script name here
export OTHERS_RE='queue_other_test\.sh' 
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {  # $1 name; fake workspace with the two existing repos (qwen/uitars repo, minicpm repo incl. its model cache)
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state; rm -rf "$W"; mkdir -p "$W" "$FAKE_STATE"
  python3 -I "$HERE/make_stub_queue.py" "$M/queue_rstem6_c.sh" "$W/queue_rstem6_c.sh"
  cp "$M/rstem6_addon_20261118.tar.gz" "$W/"
  for r in mpr-gui-crosslocale mpr-minicpm; do mkdir -p "$W/$r/.git" "$W/$r/01_code" "$W/$r/.venv/bin"; touch "$W/$r/.venv/bin/activate"; done
  mkdir -p "$W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5/snapshots/14a2004908140699317e2109e90573c1e6323c36"; echo '{}' > "$W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5/snapshots/14a2004908140699317e2109e90573c1e6323c36/config.json"
}
q() { ( cd "$W" && W="$W" bash queue_rstem6_c.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
ST() { echo "$W/QUEUE_RSTEM6_STATUS"; }
RES() { echo "$W/queue_results/rstem6"; }
steps() { tr '\n' ' ' < "$W/steps_run.txt" 2>/dev/null; }
full_order() { local s="" k; for k in "$@"; do s="$s setup_$k model_$k smoke_$k full_$k"; done; echo "${s# } "; }
ALL5="qwen3_vl_8b minicpm_v45 uitars_15_7b internvl35_8b internvl25_8b"
t_plan() {
  echo "### plan: prints, changes nothing"
  mkws plan; ( cd "$W" && W="$W" bash queue_rstem6_c.sh --plan ) > "$W/out.txt" 2>&1
  ck "plan lists the five models in the required order" bash -c "grep -n '^ (' $W/out.txt | cut -d'(' -f2 | cut -d')' -f1 | tr '\n' ' ' | grep -q '^qwen3_vl_8b minicpm_v45 uitars_15_7b internvl35_8b internvl25_8b $'"
  ck "plan shows 18,960 / batch / limits" bash -c "grep -q '18,960' $W/out.txt && grep -q 'batch 4' $W/out.txt && grep -q 'full limit 10 h' $W/out.txt"
  ck "plan created nothing in the workspace" bash -c "[ ! -e $W/QUEUE_RSTEM6_STATUS ] && [ ! -d $W/queue_results ] && [ ! -e $W/.queue_rstem6.lock ]"
  mkws plan2; ( cd "$W" && ONLY="internvl25_8b qwen3_vl_8b" W="$W" bash queue_rstem6_c.sh --plan ) > "$W/out.txt" 2>&1
  ck "ONLY keeps the fixed order (qwen3_vl_8b before internvl25_8b)" bash -c "grep -q 'Models in this order: qwen3_vl_8b internvl25_8b' $W/out.txt"
}
t_ok() {
  echo "### ok: five models, order, packages, COMPLETE, cache deletion"
  mkws ok; q
  ck "QUEUE_RSTEM6_DONE = OK, exit 0" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ] && [ \$(cat $W/rc.txt) = 0 ]"
  ck "step order = setup, model, smoke, full for the five models in order" bash -c "[ \"$(steps)\" = '$(full_order $ALL5)' ]"
  ck "every model has COMPLETE, package .tar.gz + .sha256 + .files.sha256, sha verifies" bash -c "cd $(RES) && for k in $ALL5; do [ -f \$k/COMPLETE ] && ls \$k/rstem6_\${k}_results_*.tar.gz >/dev/null && ls \$k/rstem6_\${k}_*.files.sha256 >/dev/null && (cd \$k && shasum -a 256 -c rstem6_\${k}_results_*.tar.gz.sha256); done"
  ck "package holds the result folder and a results_files.sha256" bash -c "tar -tzf $(RES)/qwen3_vl_8b/rstem6_qwen3_vl_8b_results_*.tar.gz | grep -q 'results/rstem6_qwen3_vl_8b/full/predictions.jsonl' && tar -tzf $(RES)/qwen3_vl_8b/rstem6_qwen3_vl_8b_results_*.tar.gz | grep -q results_files.sha256"
  ck "model caches deleted after each finished model (qwen/uitars repo, minicpm, ivl)" bash -c "[ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct ] && [ ! -d $W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5 ] && [ ! -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B ] && [ ! -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B ] && [ ! -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B ]"
  ck "status lines: smoke gate and complete gate per model" bash -c "[ \$(grep -c 'smoke gate:' $(ST)) = 5 ] && [ \$(grep -c 'complete gate:' $(ST)) = 5 ]"
  ck "PACKAGES_SHA256.txt has 5 model packages + queue logs" bash -c "[ \$(wc -l < $(RES)/PACKAGES_SHA256.txt) -ge 6 ]"
  ck "old queue names untouched (no QUEUE_C_*, QUEUE6_*)" bash -c "[ ! -e $W/QUEUE_C_STATUS ] && [ ! -e $W/QUEUE6_STATUS ]"
  q
  ck "rerun: every model skipped as complete, verdict OK, no step ran again" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ] && [ \$(grep -c 'already complete' $(ST)) = 5 ] && [ \"$(steps)\" = '$(full_order $ALL5)' ]"
  mkws keep; KEEP_MODEL_CACHE=1 q
  ck "KEEP_MODEL_CACHE=1 keeps the caches" bash -c "[ -d $W/mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B ] && [ -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B ]"
  mkws only; ONLY="minicpm_v45 uitars_15_7b" q
  ck "ONLY subset: two models run, DONE OK" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ] && [ \"$(steps)\" = '$(full_order minicpm_v45 uitars_15_7b)' ]"
}
t_fail() {
  echo "### failure rules"
  mkws f1; FAKE_FAIL=smoke_minicpm_v45 q
  ck "minicpm smoke fails: full skipped, other four complete, verdict FAILED steps: smoke_minicpm_v45" bash -c "grep -q '^FAILED steps: smoke_minicpm_v45' $W/QUEUE_RSTEM6_DONE && ! grep -q full_minicpm_v45 $W/steps_run.txt && grep -q full_internvl25_8b $W/steps_run.txt && [ ! -f $(RES)/minicpm_v45/COMPLETE ] && ls $(RES)/minicpm_v45/*_PARTIAL.tar.gz >/dev/null"
  ck "failed model keeps its cache; later models run" bash -c "[ -d $W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5 ] && [ -f $(RES)/internvl25_8b/COMPLETE ]"
  mkws f2; FAKE_FAIL=setup_internvl35_8b q
  ck "ivl35 setup fails: model-specific, uitars and ivl25 continue (ivl25 setup runs again and passes)" bash -c "grep -q '^FAILED steps: setup_internvl35_8b' $W/QUEUE_RSTEM6_DONE && ! grep -q model_internvl35_8b $W/steps_run.txt && [ -f $(RES)/uitars_15_7b/COMPLETE ] && [ -f $(RES)/internvl25_8b/COMPLETE ]"
  mkws f3; FAKE_FAIL=model_qwen3_vl_8b q
  ck "model download/check fails: no smoke/full for that model, others run" bash -c "! grep -q smoke_qwen3_vl_8b $W/steps_run.txt && [ -f $(RES)/minicpm_v45/COMPLETE ]"
  mkws f4; FAKE_FAIL=full_eta_uitars_15_7b q
  ck "ETA stop (exit 3): FAILED steps: full_uitars_15_7b_eta, partial package, ivl25 still runs" bash -c "grep -q 'full_uitars_15_7b_eta' $W/QUEUE_RSTEM6_DONE && ls $(RES)/uitars_15_7b/*_PARTIAL.tar.gz >/dev/null && [ -f $(RES)/internvl25_8b/COMPLETE ]"
  mkws f5; FAKE_FAIL=full_sleep_qwen3_vl_8b FAKE_TIMEOUT_MATCH="full" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported, queue continues" bash -c "grep -q 'step full_qwen3_vl_8b TIMEOUT' $(ST) && [ -f $(RES)/minicpm_v45/COMPLETE ]"
  mkws f6; FAKE_FAIL=full_cuda_minicpm_v45 q
  ck "CUDA fatal text in a step log: queue stops, later models not started" bash -c "grep -q 'STOP QUEUE (environmental failure: CUDA fatal' $(ST) && ! grep -q setup_internvl35_8b $W/steps_run.txt && grep -q '^FAILED steps:' $W/QUEUE_RSTEM6_DONE"
  mkws f7; FAKE_FAIL=full_qwen3_vl_8b q
  ck "full fails without CUDA text: partial package, queue continues" bash -c "ls $(RES)/qwen3_vl_8b/*_PARTIAL.tar.gz >/dev/null && [ -f $(RES)/internvl25_8b/COMPLETE ]"
  mkws f8; FAKE_FAIL=full_qwen3_vl_8b FAKE_GPU_USED_AFTER=1 q
  ck "failed run: GPU re-check passes with idle GPU (message in status)" bash -c "grep -q 'environment re-check ok' $(ST)"
}
t_gate() {
  echo "### start conditions: GPU, other queues, net, disk, sha"
  mkws g1; FAKE_GPU_APPS="4242, python, 9000 MiB" q
  ck "GPU process: FAILED steps: gpu, nothing started" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_RSTEM6_DONE && [ ! -e $W/steps_run.txt ]"
  mkws g2; FAKE_GPU_USED=1500 q
  ck "GPU memory 1500 MiB (limit): refused" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_RSTEM6_DONE"
  mkws g2b; FAKE_GPU_USED=1499 q
  ck "GPU memory 1499 MiB: starts" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
  mkws g3; FAKE_GPU_NAME="NVIDIA A100" q
  ck "not a 4090: refused" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_RSTEM6_DONE"
  mkws g4; ( sleep 25 & ) ; cp "$HERE/shims/sleeper_queue.sh" "$W/queue_other_test.sh" 2>/dev/null || printf '#!/bin/bash\nsleep 25\n' > "$W/queue_other_test.sh"; ( bash "$W/queue_other_test.sh" & ) ; sleep 1; q; pkill -f queue_other_test.sh 2>/dev/null
  ck "another queue_*.sh running: refused (FAILED steps: others)" bash -c "grep -q '^FAILED steps: others' $W/QUEUE_RSTEM6_DONE"
  mkws n1; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow net: FAILED steps: network" bash -c "grep -q '^FAILED steps: network' $W/QUEUE_RSTEM6_DONE"
  mkws n2; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1: OK" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
  mkws n3; FAKE_NET_PT="6000000 10.0 206" q
  ck "slow download.pytorch.org (mpr-ivl needs a venv): refused; not checked when ONLY has no InternVL" bash -c "grep -q '^FAILED steps: network' $W/QUEUE_RSTEM6_DONE"
  mkws n4; FAKE_NET_PT="6000000 10.0 206" ONLY="qwen3_vl_8b minicpm_v45" q
  ck "ONLY without InternVL: pytorch speed not needed, OK" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
  mkws d1; FAKE_FREE=19 ONLY="qwen3_vl_8b" q
  ck "19 GB free for qwen3 (needs 20): refused (FAILED steps: disk_qwen3_vl_8b)" bash -c "grep -q 'FAILED steps: disk_qwen3_vl_8b' $W/QUEUE_RSTEM6_DONE && [ ! -e $W/steps_run.txt ]"
  mkws d2; FAKE_FREE=29 q
  ck "29 GB free (df is constant in the test): qwen3 + minicpm run; uitars needs 36 GB: STOP disk_uitars_15_7b, queue stops (order changed 2026-11-18)" bash -c "grep -q 'need 36 GB for uitars_15_7b' $(ST) && grep -q 'FAILED steps: disk_uitars_15_7b' $W/QUEUE_RSTEM6_DONE && [ -f $(RES)/minicpm_v45/COMPLETE ] && ! grep -q setup_uitars $W/steps_run.txt && ! grep -q setup_internvl35 $W/steps_run.txt"
  mkws d3; FAKE_FREE=36 q
  ck "36 GB free: all five models pass their disk rule (uitars needs exactly 36)" bash -c "[ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
  mkws d4; FAKE_FREE=35 q
  ck "35 GB free: stops at uitars_15_7b (needs 36 GB), earlier models complete" bash -c "grep -q 'need 36 GB for the uitars_15_7b download\|need 36 GB for uitars_15_7b' $(ST) && grep -q 'disk_uitars_15_7b' $W/QUEUE_RSTEM6_DONE && [ -f $(RES)/minicpm_v45/COMPLETE ] && [ ! -f $(RES)/internvl35_8b/COMPLETE ]"
  mkws s1; echo x >> "$W/rstem6_addon_20261118.tar.gz"; q
  ck "addon tar modified: FAILED steps: addon, nothing started" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_RSTEM6_DONE && [ ! -e $W/steps_run.txt ]"
  mkws s2; rm "$W/rstem6_addon_20261118.tar.gz"; q
  ck "addon tar missing: refused" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_RSTEM6_DONE"
  mkws s3; echo x > "$W/internvl25_rel6_frja_images_20261116.tar"; q
  ck "wrong images tar sha: refused" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_RSTEM6_DONE"
  mkws s4; rm -rf "$W/mpr-minicpm/.venv"; q
  ck "existing repo without .venv: STOP repo_minicpm_v45 (environmental), later models not started" bash -c "grep -q 'repo_minicpm_v45' $W/QUEUE_RSTEM6_DONE && ! grep -q setup_internvl35_8b $W/steps_run.txt"
}
t_cache() {
  echo "### cache deletion is scoped"
  mkws c1; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs"; echo keep > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/b"
  mkdir -p "$W/queue_results/qwen3_8b_frja" "$W/mpr-gui-crosslocale/followup_rel_6lang/results/expA6_qwen3_vl_8b"; echo keep > "$W/queue_results/qwen3_8b_frja/k"; echo keep > "$W/mpr-gui-crosslocale/followup_rel_6lang/results/expA6_qwen3_vl_8b/k"; echo OK > "$W/QUEUE_C2_DONE"
  q
  ck "other model caches, older results and old DONE files untouched" bash -c "[ -s $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/b ] && [ -s $W/queue_results/qwen3_8b_frja/k ] && [ -s $W/mpr-gui-crosslocale/followup_rel_6lang/results/expA6_qwen3_vl_8b/k ] && [ \"\$(cat $W/QUEUE_C2_DONE)\" = OK ]"
  ck "own result folders are kept after the cache deletion" bash -c "[ -f $W/mpr-minicpm/followup_rel_6lang/results/rstem6_minicpm_v45/full/predictions.jsonl ]"
}
t_dup() {
  echo "### dup and signal"
  mkws dup; ( cd "$W" && W="$W" FAKE_FAIL=smoke_sleep_qwen3_vl_8b bash "$W/queue_rstem6_c.sh" > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 4; q
  ck "second copy refuses (already running), exit 1" bash -c "grep -q 'already running' $W/out.txt && [ \$(cat $W/rc.txt) = 1 ]"
  pkill -TERM -f "bash $W/queue_rstem6_c.sh" 2>/dev/null; wait $p 2>/dev/null
  for _ in $(seq 1 45); do has "$(ST)" 'interrupted by signal' && break; sleep 1; done
  ck "SIGTERM: no QUEUE_RSTEM6_DONE, status says interrupted" bash -c "[ ! -e $W/QUEUE_RSTEM6_DONE ] && grep -q 'interrupted by signal' $(ST)"
}
t_static() {
  echo "### static"
  Q="$M/queue_rstem6_c.sh"
  ck "bash -n (queue, runner sh, tests)" bash -c 'bash -n "$1" && bash -n "$2" && bash -n "$3"' _ "$Q" "$M/addon/followup_rel_6lang/rstem6/run_rstem6.sh" "$HERE/run_queue_tests.sh"
  ck "ADDON_SHA in the queue equals the real tar" bash -c 'grep -q "^ADDON_SHA=$(shasum -a 256 "$2" | cut -d" " -f1)" "$1"' _ "$Q" "$M/rstem6_addon_20261118.tar.gz"
  ck "addon tar holds only followup_rel_6lang/rstem6/" bash -c '! tar -tzf "$1" | grep -v -E "^followup_rel_6lang/(rstem6(/.*)?)?$" | grep -q .' _ "$M/rstem6_addon_20261118.tar.gz"
  ck "rm -rf occurrences: exactly 4 and each one is a stage dir, the guarded model-cache delete or the .venv of the new mpr-ivl" bash -c 'n=$(grep -c "rm -rf" "$1"); [ "$n" = 4 ] && [ "$(grep "rm -rf" "$1" | grep -c -F -e "rm -rf \"\$stage\"" -e "rm -rf \"\${d:?}\"" -e "rm -rf .venv")" = 4 ]' _ "$Q"
  ck "only delete_cache removes a model cache and it checks the path pattern" bash -c 'grep -c "rm -rf \"\${d:?}\"" "$1" | grep -q "^1$" && grep -q "models--\*) ;;" "$1"' _ "$Q"
  ck "SHA256SUMS inside the addon folder verifies" bash -c 'cd "$1/addon/followup_rel_6lang/rstem6" && shasum -a 256 -c SHA256SUMS' _ "$M"
  ck "queue never reads AGENTS.md / CLAUDE.md of the server" bash -c '! grep -E "cat .*(AGENTS|CLAUDE)\.md|vast-agents-guide" "$1"' _ "$Q"
}
case "${1:-}" in
  all) for t in plan ok fail gate cache dup static; do "t_$t"; done ;;
  *) echo "usage: $0 all"; exit 2 ;;
esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
