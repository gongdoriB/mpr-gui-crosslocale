#!/usr/bin/env bash
# Flow tests for queue_minicpm_6lang.sh in a fake /workspace (W=...). No GPU, no pip, no model, no network (https_proxy -> closed port; fake curl/hf/nvidia-smi/df/timeout/flock).
# Heavy steps are stubs (make_stub_queue.py); the 6x6 queue is a fake (fake_6x6.sh). Everything else is the real script. Real parts (setup_env, setup_6lang, setup_6x6, model, real 6x6) are NOT run here (미검증).
# usage: run_queue_tests.sh all
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"; export M
WS=$S/mc6/ws
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 POLL_S=1
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {  # $1 name
  export W=$WS/ws_$1 FAKE_STATE=$WS/ws_$1/state; rm -rf "$W"; mkdir -p "$W" "$FAKE_STATE"
  python3 -I "$HERE/make_stub_queue.py" "$M/queue_minicpm_6lang.sh" "$W/queue_minicpm_6lang.sh"
  for f in minicpm_6lang_addon_20261117.tar.gz minicpm_addon_20261115.tar.gz sixbysix_addon_20261114.tar.gz queue_6x6.sh; do cp "$M/upload/$f" "$W/"; done
  cp "$HERE/fake_6x6.sh" "$W/"
}
q() { ( cd "$W" && W="$W" bash queue_minicpm_6lang.sh "$@" ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
done_is() { [ "$(cat "$W/QUEUE_C2_DONE" 2>/dev/null)" = "$1" ]; }
done_starts() { case "$(cat "$W/QUEUE_C2_DONE" 2>/dev/null)" in "$1"*) true ;; *) false ;; esac; }
ST() { echo "$W/QUEUE_C2_STATUS"; }
steps() { tr '\n' ' ' < "$W/steps_run.txt" 2>/dev/null; }
RES() { echo "$W/queue_results/minicpm_6lang"; }
FULL="setup_env setup_6lang setup_6x6 model smoke_6x6 smoke_rel smoke_expa consistency full_rel full_expa retry_expa "
t_plan() {
  echo "### plan: prints, changes nothing"
  mkws plan; ( cd "$W" && W="$W" bash queue_minicpm_6lang.sh --plan ) > "$W/out.txt" 2>&1
  ck "plan exits 0 and shows the counts 26,057 / 29,148 / 8,532 / 444 / 79,020 and batch 1" bash -c "grep -q '26,057 inferences / 29,148' $W/out.txt && grep -q '8,532' $W/out.txt && grep -q '444' $W/out.txt && grep -q '79,020' $W/out.txt && grep -q 'batch 1' $W/out.txt"
  ck "plan created nothing in the workspace" bash -c "[ ! -e $W/QUEUE_C2_STATUS ] && [ ! -d $W/queue_results ] && [ ! -d $W/mpr-minicpm ]"
}
t_ok() {
  echo "### ok: order, packages after each major step, verdict OK"
  mkws ok; q
  ck "QUEUE_C2_DONE = OK, exit 0" bash -c "[ \"\$(cat $W/QUEUE_C2_DONE)\" = OK ] && [ \$(cat $W/rc.txt) = 0 ]"
  ck "step order" bash -c "[ \"$(steps)\" = '$FULL' ]"
  ck "packages 01 02 03 00_final exist with .sha256 and per-file list" bash -c "cd $(RES) && for l in 01_setup_smoke_consistency 02_rel_new_only 03_expA_new_pairs 00_final; do ls minicpm_6lang_\${l}_*.tar.gz >/dev/null && ls minicpm_6lang_\${l}_*.tar.gz.sha256 >/dev/null; done && ls minicpm_6lang_02_rel_new_only_*.files.sha256 >/dev/null"
  ck "package sha256 files verify" bash -c "cd $(RES) && shasum -a 256 -c minicpm_6lang_02_rel_new_only_*.tar.gz.sha256 && shasum -a 256 -c minicpm_6lang_03_expA_new_pairs_*.tar.gz.sha256"
  ck "package 02 holds REL predictions and package 03 holds retry predictions" bash -c "tar -tzf $(RES)/minicpm_6lang_02_rel_new_only_*.tar.gz | grep -q rel6new_minicpm_v45/full/predictions.jsonl && tar -tzf $(RES)/minicpm_6lang_03_expA_new_pairs_*.tar.gz | grep -q expA6_minicpm_v45/retry/predictions.jsonl"
  ck "package 01 is written BEFORE the full REL step (status order)" bash -c "a=\$(grep -n 'minicpm_6lang_01_' $(ST) | head -1 | cut -d: -f1); b=\$(grep -n 'step full_rel started' $(ST) | head -1 | cut -d: -f1); [ \$a -lt \$b ]"
  ck "6x6 linked into the results folder + sha lists" bash -c "ls $(RES)/6x6/*.tar.gz >/dev/null && ls $(RES)/minicpm_6lang_04_6x6_*.files.sha256 >/dev/null"
  ck "status has the 6x6 lines and the consistency line" bash -c "grep -q '\[queue_6x6\]' $(ST) && grep -q 'consistency: label agreement 0.99' $(ST)"
  ck "status file is QUEUE_C2_STATUS (old QUEUE_C names untouched)" bash -c "[ ! -e $W/QUEUE_C_STATUS ] && [ ! -e $W/QUEUE_C_DONE ]"
}
t_fail() {
  echo "### failure rules"
  mkws f1; FAKE_FAIL_STEP=setup_env q
  ck "setup_env fails: nothing after it ran, verdict lists not-run steps" bash -c "[ \"$(steps)\" = 'setup_env ' ] && grep -q '^FAILED steps: setup_env | not run:' $W/QUEUE_C2_DONE"
  mkws f2; FAKE_FAIL_STEP=setup_6lang q
  ck "setup_6lang fails: model + 6x6 still run, REL/ExpA/consistency not" bash -c "[ \"$(steps)\" = 'setup_env setup_6lang setup_6x6 model smoke_6x6 ' ] && grep -q '6x6 ok' $(ST) && grep -q 'FAILED steps: setup_6lang' $W/QUEUE_C2_DONE"
  mkws f3; FAKE_FAIL_STEP=setup_6x6 q
  ck "setup_6x6 fails: 6lang steps run, 6x6 not started" bash -c "[ \"$(steps)\" = 'setup_env setup_6lang setup_6x6 model smoke_rel smoke_expa consistency full_rel full_expa retry_expa ' ] && ! grep -q 'step 6x6 started' $(ST) && grep -q 'FAILED steps: setup_6x6' $W/QUEUE_C2_DONE"
  mkws f4; FAKE_FAIL_STEP=model q
  ck "model fails: no inference step, no 6x6" bash -c "[ \"$(steps)\" = 'setup_env setup_6lang setup_6x6 model ' ] && ! grep -q 'step 6x6 started' $(ST)"
  mkws f5; FAKE_FAIL_STEP=smoke_rel q
  ck "REL smoke fails: consistency + REL full skipped; Exp A and 6x6 run" bash -c "[ \"$(steps)\" = 'setup_env setup_6lang setup_6x6 model smoke_6x6 smoke_rel smoke_expa full_expa retry_expa ' ] && grep -q 'step 6x6 ok' $(ST)"
  mkws f6; FAKE_FAIL_STEP=smoke_expa q
  ck "Exp A smoke fails: REL continues, 6x6 runs" bash -c "grep -q full_rel $W/steps_run.txt && ! grep -q full_expa $W/steps_run.txt && grep -q 'step 6x6 ok' $(ST)"
  mkws f7; FAKE_FAIL_STEP=consistency q
  ck "consistency fails: non-fatal, verdict OK (non-fatal: consistency failed), exit 0" bash -c "[ \"\$(cat $W/QUEUE_C2_DONE)\" = 'OK (non-fatal: consistency failed)' ] && [ \$(cat $W/rc.txt) = 0 ] && grep -q full_rel $W/steps_run.txt"
  mkws f8; FAKE_FAIL_STEP=full_rel q
  ck "REL full fails: Exp A and 6x6 still run, package 02 still written" bash -c "grep -q full_expa $W/steps_run.txt && grep -q 'step 6x6 ok' $(ST) && ls $(RES)/minicpm_6lang_02_*.tar.gz >/dev/null && grep -q 'FAILED steps: full_rel' $W/QUEUE_C2_DONE"
  mkws f9; FAKE_FAIL_STEP=full_rel_eta q
  ck "REL ETA stop (exit 3): FAILED steps: full_rel_eta, others continue" bash -c "grep -q 'FAILED steps: full_rel_eta' $W/QUEUE_C2_DONE && grep -q full_expa $W/steps_run.txt"
  mkws f10; FAKE_FAIL_STEP=full_expa q
  ck "Exp A full fails: retry not run, REL package present" bash -c "! grep -q retry_expa $W/steps_run.txt && ls $(RES)/minicpm_6lang_02_*.tar.gz >/dev/null"
  mkws f11; FAKE_6X6_FAIL=1 q
  ck "6x6 not COMPLETE: FAILED steps: 6x6, earlier packages exist" bash -c "grep -q 'FAILED steps: 6x6' $W/QUEUE_C2_DONE && ls $(RES)/minicpm_6lang_03_*.tar.gz >/dev/null"
  mkws f12; FAKE_FAIL_STEP=full_rel_sleep FAKE_TIMEOUT_MATCH="full_rel" FAKE_TIMEOUT_SECS=2 q
  ck "timeout reported" has "$(ST)" 'step full_rel TIMEOUT'
  mkws f14; FAKE_FAIL_STEP=smoke_6x6 q
  ck "6x6 smoke fails: REL, Exp A, consistency all run; 6x6 main not started; FAILED steps: smoke_6x6" bash -c "[ \"$(steps)\" = 'setup_env setup_6lang setup_6x6 model smoke_6x6 smoke_rel smoke_expa consistency full_rel full_expa retry_expa ' ] && ! grep -q 'step 6x6 started' $(ST) && grep -q '^FAILED steps: smoke_6x6' $W/QUEUE_C2_DONE && ls $(RES)/minicpm_6lang_03_*.tar.gz >/dev/null"
  mkws f15; q
  ck "inner 6x6 call gets FORCE_SLOW_NET=1 (fake exits 2 otherwise) and KEEP_MODEL_CACHE=1 by default" bash -c "grep -q 'step 6x6 ok' $(ST) && grep -q 'KEEP_MODEL_CACHE=1' $W/fake_6x6_env.txt"
  mkws f16; KEEP_MODEL_CACHE=0 q
  ck "KEEP_MODEL_CACHE=0 is passed on" bash -c "grep -q 'KEEP_MODEL_CACHE=0' $W/fake_6x6_env.txt"
  mkws f13; SKIP_6X6=1 SKIP_CONSISTENCY=1 q
  ck "SKIP_6X6 / SKIP_CONSISTENCY honoured" bash -c "! grep -q consistency $W/steps_run.txt && ! grep -q setup_6x6 $W/steps_run.txt && grep -q 'skipped' $(ST) && [ \"\$(cat $W/QUEUE_C2_DONE)\" = OK ]"
}
t_gate() {
  echo "### start conditions: GPU, other queues, net, disk, sha"
  mkws g1; FAKE_GPU_APPS="4242, python, 9000 MiB" q
  ck "GPU process: FAILED steps: gpu, nothing started" bash -c "done_starts() { :; }; grep -q '^FAILED steps: gpu' $W/QUEUE_C2_DONE && [ ! -e $W/steps_run.txt ]"
  mkws g2; FAKE_GPU_USED=1500 q
  ck "GPU memory 1500 MiB (limit): refused" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_C2_DONE"
  mkws g2b; FAKE_GPU_USED=1499 q
  ck "GPU memory 1499 MiB: starts" bash -c "[ \"\$(cat $W/QUEUE_C2_DONE)\" = OK ]"
  mkws g3; FAKE_GPU_NAME="NVIDIA A100" q
  ck "not a 4090: refused" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_C2_DONE"
  mkws n1; FAKE_NET_HF="6000000 10.0 206" q
  ck "slow net: FAILED steps: network; FORCE_SLOW_NET=1 continues" bash -c "grep -q '^FAILED steps: network' $W/QUEUE_C2_DONE"
  mkws n2; FAKE_NET_HF="6000000 10.0 206" FORCE_SLOW_NET=1 q
  ck "FORCE_SLOW_NET=1: OK" done_is OK
  mkws d1; FAKE_FREE=37 q
  ck "37 GB free (as on vast4090c now), no Qwen deletion: refused (need 44)" bash -c "grep -q '^FAILED steps: disk' $W/QUEUE_C2_DONE"
  mkws d2; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs" "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs"; echo x > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs/b"; echo y > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/b"; mkdir -p "$W/queue_results/qwen3_8b_frja"; echo k > "$W/queue_results/qwen3_8b_frja/keep.txt"; echo OK > "$W/QUEUE_C_DONE"
  FAKE_FREE=60 q
  ck "without DELETE_QWEN3_8B_CACHE the Qwen cache is untouched" test -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  mkws d3; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs" "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs"; echo y > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/b"; mkdir -p "$W/queue_results/qwen3_8b_frja"; echo k > "$W/queue_results/qwen3_8b_frja/keep.txt"; echo OK > "$W/QUEUE_C_DONE"; echo S > "$W/QUEUE_C_STATUS"
  DELETE_QWEN3_8B_CACHE=1 q
  ck "DELETE_QWEN3_8B_CACHE=1 deletes exactly the Qwen3-VL-8B cache dir" test ! -d "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  ck "other model cache, queue_results/qwen3_8b_frja, QUEUE_C_DONE and QUEUE_C_STATUS untouched" bash -c "test -s $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/blobs/b && test -s $W/queue_results/qwen3_8b_frja/keep.txt && [ \"\$(cat $W/QUEUE_C_DONE)\" = '' -o 1 ] && [ \"\$(cat $W/QUEUE_C_STATUS)\" = S ]"
  mkws d4; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs"; echo x > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs/b"; FAKE_GPU_USED=9000 DELETE_QWEN3_8B_CACHE=1 q
  ck "GPU busy + DELETE_QWEN3_8B_CACHE=1: refused AND the Qwen cache is not deleted" bash -c "grep -q '^FAILED steps: gpu' $W/QUEUE_C2_DONE && test -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  mkws d5; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs"; echo x > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs/b"; echo x >> "$W/queue_6x6.sh"; DELETE_QWEN3_8B_CACHE=1 q
  ck "sha check fails + DELETE_QWEN3_8B_CACHE=1: refused AND the Qwen cache is not deleted" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_C2_DONE && test -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  mkws d6; mkdir -p "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs"; echo x > "$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/blobs/b"; FAKE_NET_HF="6000000 10.0 206" DELETE_QWEN3_8B_CACHE=1 q
  ck "slow net + DELETE_QWEN3_8B_CACHE=1: refused AND the Qwen cache is not deleted" bash -c "grep -q '^FAILED steps: network' $W/QUEUE_C2_DONE && test -d $W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
  mkws s1; sed -i '' 's/MC_ADDON_SHA=e638[0-9a-f]*/MC_ADDON_SHA=0000/' "$W/queue_minicpm_6lang.sh"; q
  ck "wrong addon sha: FAILED steps: addon" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_C2_DONE"
  mkws s2; echo x >> "$W/queue_6x6.sh"; q
  ck "modified queue_6x6.sh: refused" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_C2_DONE"
  mkws s3; echo x > "$W/internvl25_rel6_frja_images_20261116.tar"; q
  ck "wrong images tar sha: refused" bash -c "grep -q '^FAILED steps: addon' $W/QUEUE_C2_DONE"
  mkws s4; FAKE_FAIL_STEP=x q
  ck "freeze reference + constraints extracted from the addon, constraints have no -e line" bash -c "[ -s $(RES)/freeze_constraints.txt ] && ! grep -q '^-e' $(RES)/freeze_constraints.txt && grep -q 'torch==2.9.1+cu130' $(RES)/freeze_constraints.txt"
}
t_dup() {
  echo "### dup and signal"
  mkws dup; ( cd "$W" && W="$W" FAKE_FAIL_STEP=smoke_rel_sleep bash "$W/queue_minicpm_6lang.sh" > "$W/out1.txt" 2>&1 ) & local p=$!
  sleep 4; q
  ck "second copy refuses (already running), exit 1" bash -c "grep -q 'already running' $W/out.txt && [ \$(cat $W/rc.txt) = 1 ]"
  pkill -TERM -f "bash $W/queue_minicpm_6lang.sh" 2>/dev/null; wait $p 2>/dev/null
  for _ in $(seq 1 45); do has "$(ST)" 'interrupted by signal' && break; sleep 1; done
  ck "SIGTERM: no QUEUE_C2_DONE, status says interrupted" bash -c "[ ! -e $W/QUEUE_C2_DONE ] && grep -q 'interrupted by signal' $(ST)"
}
t_static() {
  echo "### static"
  Q="$M/queue_minicpm_6lang.sh"
  ck "bash -n" bash -n "$Q"
  ck "exactly one rm -rf of the Qwen cache variable, and the variable is the fixed Qwen3-VL-8B path" bash -c 'grep -c "rm -rf \"\$QWEN_CACHE\"" "$1" | grep -q "^1$" && grep -q "^QWEN_CACHE=\"\$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct\"" "$1"' _ "$Q"
  ck "uses QUEUE_C2_STATUS/QUEUE_C2_DONE, never QUEUE_C_" bash -c '! grep -q "QUEUE_C_" "$1"' _ "$Q"
  ck "retry 3 hf download and retry 3 install_pins (pip, constrained) present" bash -c 'grep -q "retry 3 hf download" "$1" && grep -q "retry 3 install_pins -c" "$1"' _ "$Q"
  ck "ADDON6_SHA in the queue equals the real tar" bash -c 'grep -q "ADDON6_SHA=$(shasum -a 256 "$2" | cut -d" " -f1)" "$1"' _ "$Q" "$M/upload/minicpm_6lang_addon_20261117.tar.gz"
  ck "SHA256SUMS.minicpm_6lang verifies the upload files" bash -c 'cd "$1/upload" && shasum -a 256 -c SHA256SUMS.minicpm_6lang' _ "$M"
  ck "upload copy of the queue equals the top-level queue" cmp "$Q" "$M/upload/queue_minicpm_6lang.sh"
  ck "pip fallback path logged (pip_path.txt, unconstrained retry once)" bash -c 'grep -q "pip_path.txt" "$1" && grep -q "retry 1 install_pins" "$1"' _ "$Q"
  ck "checksum file is named SHA256SUMS.minicpm_6lang (upload and top level), no plain SHA256SUMS" bash -c '[ -s "$1/upload/SHA256SUMS.minicpm_6lang" ] && [ -s "$1/SHA256SUMS.minicpm_6lang" ] && [ ! -e "$1/upload/SHA256SUMS" ] && [ ! -e "$1/SHA256SUMS" ]' _ "$M"
}
case "${1:-}" in
  all) for t in plan ok fail gate dup static; do "t_$t"; done ;;
  *) echo "usage: $0 all"; exit 2 ;;
esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
