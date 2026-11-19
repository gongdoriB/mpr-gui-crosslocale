#!/usr/bin/env bash
# Integration test of the REAL queue_6x6.sh (sha d538b0cd..., unchanged) exactly as queue_minicpm_6lang.sh calls it, in a fake /workspace that is in the "after setup_env" state
# (/workspace/mpr-minicpm = clone at the adapter-patch commit, stub venv). Fake: hf, nvidia-smi, curl, df, timeout, flock, GPU/model (Q6_TEST=1 = --mock-model). Real code paths:
# ensure_commit_and_patches (patch-id), venv_ok, ensure_data (raw data reuse OR download hook, manifest copy OR audit_release.sh), manifest sha check, check_cfg, dry_plan (79,020),
# fn_smoke + check_6x6_gate, "smoke already passed", ETA guard, full run (mock, 79,020 rows), gate, summary, package, COMPLETE, cache rule with KEEP_MODEL_CACHE=1, QUEUE6_* writes.
# NOT exercisable: real model load/GPU run; with Q6_TEST=1 the GPU wait loop and the 4090 check are skipped (a separate case runs without Q6_TEST to hit them); safetensors size check.
# usage: run_6x6_integration.sh all
set -u
S=${TMPDIR}/work
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
WS=$S/mc6/ws6
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims6x6:$PATH" EXPECT_TORCH=2.9.1 LOCAL_DATA_SRC=$S/sx/sx_qwen/01_code/data/raw/mpr_gui_bench DL_RETRY_SLEEP=1 POLL_S=1 RETRY_SLEEP=0 IGNORE_OTHER_QUEUES=1
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
has() { grep -q -- "$2" "$1" 2>/dev/null; }
mkws() {  # $1 name, $2 = rawsrc (Qwen repo with complete raw data + manifest, as on vast4090c if the 6x6 data exist there) | nodata ; stub venv pre-made unless $3 = novenv
  export W=$WS/$1 FAKE_STATE=$WS/$1/state; rm -rf "$W"; mkdir -p "$W" "$FAKE_STATE"
  cp "$M/upload/queue_6x6.sh" "$M/upload/sixbysix_addon_20261114.tar.gz" "$W/"
  git clone -q "$S/mc/repo" "$W/mpr-minicpm" && git -C "$W/mpr-minicpm" checkout -q f4f0d97 || echo "CLONE FAILED"
  if [ "${3:-}" != novenv ]; then
    mkdir -p "$W/mpr-minicpm/.venv/bin"
    printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec "%s" "$@"\n' "$W/mpr-minicpm" "$HERE/test_python.sh" > "$W/mpr-minicpm/.venv/bin/python"; chmod +x "$W/mpr-minicpm/.venv/bin/python"
    printf 'export PATH="%s/.venv/bin:$PATH"\n' "$W/mpr-minicpm" > "$W/mpr-minicpm/.venv/bin/activate"
  fi
  if [ "$2" = rawsrc ]; then
    mkdir -p "$W/mpr-gui-crosslocale/01_code/data/raw" "$W/mpr-gui-crosslocale/01_code/data/manifests"
    cp -R "$LOCAL_DATA_SRC" "$W/mpr-gui-crosslocale/01_code/data/raw/mpr_gui_bench"; cp "$S/sx/sx_qwen/01_code/data/manifests/mpr_gui_manifest.jsonl" "$W/mpr-gui-crosslocale/01_code/data/manifests/"
  else mkdir -p "$W/mpr-gui-crosslocale"; fi
  ( cd "$W" && tar -xzf sixbysix_addon_20261114.tar.gz && cd sixbysix_addon && shasum -a 256 -c SHA256SUMS >/dev/null )
}
# the calls of queue_minicpm_6lang.sh (same env)
sixfn()   { ( cd "$W" && Q6_TEST=1 Q6_TEST_PY="$HERE/test_python.sh" W="$W" bash queue_6x6.sh --fn "$@" ); }
sixsmoke(){ ( cd "$W" && Q6_TEST=1 Q6_TEST_PY="$HERE/test_python.sh" W="$W" KEY=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE=1 bash queue_6x6.sh --fn fn_smoke ); }
sixmain() { ( cd "$W" && Q6_TEST=1 Q6_TEST_PY="$HERE/test_python.sh" W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE=1 bash queue_6x6.sh ); }
dl()      { ( cd "$W" && export Q6_TEST=1 W="$W" HF_HOME="$W/mpr-minicpm/.cache/huggingface" KEY=minicpm_v45 && bash queue_6x6.sh --fn fn_download ); }   # hf shim
OUT_REL() { echo "$W/mpr-minicpm/followup_rel_4lang/results/6x6_minicpm_v45"; }

t_rawsrc() {
  echo "### A: after setup_env, 6x6 raw data reusable from the Qwen repo (symlink), manifest copied; fn_setup, download, fn_smoke, main"
  mkws a rawsrc
  sixfn fn_setup mc > "$W/setup.out" 2>&1; local rc=$?
  ck "fn_setup mc exit 0" test $rc = 0
  ck "patch-id check passed (commit/patch state ok)" has "$W/setup.out" 'commit/patch state ok'
  ck "venv ok (reused)" has "$W/setup.out" 'venv ok (reused)'
  ck "raw data reused by symlink from the Qwen repo" bash -c "[ -L $W/mpr-minicpm/01_code/data/raw/mpr_gui_bench ] && grep -q 'reusing raw data' $W/setup.out"
  ck "manifest copied and sha256 equals the 8B-run manifest (4dd919fc...)" bash -c "grep -q 4dd919fc799a653d1cc536716e953168db86a15c3c346cbe659bd41a833a805a $W/setup.out || grep -q 4dd919fc $W/setup.out"
  ck "family mc ready line" has "$W/setup.out" 'family mc ready'
  ck "QUEUE6_STATUS not needed for --fn but log dir created" test -d "$W/queue_logs6"
  dl > "$W/dl.out" 2>&1
  ck "hf download shim ran, config.json in the pinned snapshot" test -s "$W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5/snapshots/14a2004908140699317e2109e90573c1e6323c36/config.json"
  sixsmoke > "$W/smoke.out" 2>&1; rc=$?
  ck "fn_smoke (KEY=minicpm_v45, same env as the MiniCPM queue) exit 0" test $rc = 0
  ck "dry-run plan 79,020 unique, ids sha equals the 8B plan" bash -c "grep -q 'input_ids 79020 unique 79020' $W/smoke.out && grep -q 22ac60aa9b02a10ddc840e8e1a2d44c0bac4c3015678e9ff42b0a072ecd7b4c8 $W/smoke.out"
  ck "config ok line for MiniCPM (revision 14a20049...)" has "$W/smoke.out" 'config ok: openbmb/MiniCPM-V-4_5 14a2004908140699317e2109e90573c1e6323c36 family minicpmv'
  ck "smoke gate JSON passed with 216 expected, model minicpm_v45" bash -c "python3 -I -c \"import json;g=json.load(open('$(OUT_REL)/gate_6x6_smoke.json'));assert g['status']=='passed',g.get('problems');print(g.get('expected'))\""
  : > "$W/QUEUE6_STATUS"
  sixmain > "$W/main.out" 2>&1; rc=$?
  local ST="$W/QUEUE6_STATUS"
  ck "main exit 0" test $rc = 0
  ck "queue_6x6 reuses the passed smoke: 'smoke already passed'" has "$ST" 'minicpm_v45 smoke already passed'
  ck "ETA check line" has "$ST" 'minicpm_v45 ETA check'
  ck "full run mock: gate passed, 79,020 success rows" bash -c "python3 -I -c \"import json;g=json.load(open('$(OUT_REL)/gate_6x6_full.json'));assert g['status']=='passed',g.get('problems')\" && [ \$(grep -c '\"status\": \"success\"' $(OUT_REL)/full_6x6.jsonl) = 79020 ]"
  ck "summary written" test -s "$(OUT_REL)/full_6x6_summary.json"
  ck "COMPLETE marker, packaged tar + SHA256SUMS" bash -c "ls $W/queue_results/6x6_minicpm_v45/COMPLETE $W/queue_results/6x6_minicpm_v45/SHA256SUMS >/dev/null && ls $W/queue_results/6x6_minicpm_v45/*.tar.gz >/dev/null && ! ls $W/queue_results/6x6_minicpm_v45/*PARTIAL* 2>/dev/null"
  ck "KEEP_MODEL_CACHE=1: cache kept after success" bash -c "test -d $W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5 && grep -q 'model cache kept (KEEP_MODEL_CACHE=1)' $ST"
  ck "QUEUE6_DONE and net_check.txt (FORCE_SLOW_NET path) written" bash -c "test -e $W/QUEUE6_DONE && test -s $W/queue_results/net_check.txt"
  ck "no second download transfer (cached snapshot reused)" bash -c "[ \$(cat $FAKE_STATE/transfers) = 1 ]"
  ck "files created by queue_6x6.sh under W (inventory)" bash -c "cd $W && ls -d QUEUE6_STATUS QUEUE6_DONE queue_logs6 sixbysix_addon .queue6.lock queue_results/net_check.txt queue_results/disk_end6.log queue_results/QUEUE6_STATUS.txt queue_results/queue_logs6 queue_results/6x6_minicpm_v45 >/dev/null"
  ( cd "$W" && find . -maxdepth 2 -newer sixbysix_addon_20261114.tar.gz -not -path './mpr-*' -not -path './state*' | sort ) > "$W/inventory.txt"
}
t_download() {
  echo "### B: no raw data anywhere -> download hook; no venv -> stub venv created; manifest generated by the real audit_release.sh"
  mkws b nodata novenv
  sixfn fn_setup mc > "$W/setup.out" 2>&1; local rc=$?
  ck "fn_setup mc exit 0 and the stub venv was created by the Q6 hook" bash -c "test $rc = 0 && test -x $W/mpr-minicpm/.venv/bin/python"
  ck "data 'downloaded' (hook copy), manifest built by audit_release.sh, sha matches" bash -c "grep -q 'downloading MPR-GUI-Bench' $W/setup.out && [ ! -L $W/mpr-minicpm/01_code/data/raw/mpr_gui_bench ] && grep -q 'manifest sha256: 4dd919fc' $W/setup.out"
  ck "family mc ready" has "$W/setup.out" 'family mc ready'
}
t_failures() {
  echo "### C: failure paths of the real queue_6x6.sh"
  mkws c rawsrc; git -C "$W/mpr-minicpm" commit -q --allow-empty -m extra
  sixfn fn_setup mc > "$W/setup.out" 2>&1; local rc=$?
  ck "repo 2 commits after base (patch + extra): fn_setup refuses" bash -c "test $rc != 0 && grep -q 'expected 0 or 1 patch commits' $W/setup.out"
  mkws c2 rawsrc; sixfn fn_setup mc >/dev/null 2>&1
  sed -i '' 's/^+model_id/+model_idX/' /dev/null 2>/dev/null
  : > "$W/QUEUE6_STATUS"; FAKE_GPU_USED=9000 bash -c 'true'
  mkws c3 rawsrc; sixfn fn_setup mc >/dev/null 2>&1; dl >/dev/null 2>&1; sixsmoke >/dev/null 2>&1
  echo '{"status": "passed", "eta_full_seconds": 99999999}' > "$(OUT_REL)/gate_6x6_smoke.json"
  sixmain > "$W/main.out" 2>&1
  ck "ETA guard: full run not started, STOPPED_ETA, _PARTIAL package, no COMPLETE" bash -c "grep -q 'full 6x6 STOPPED, not started: ETA' $W/QUEUE6_STATUS && ls $W/queue_results/6x6_minicpm_v45/*PARTIAL*.tar.gz >/dev/null && [ ! -e $W/queue_results/6x6_minicpm_v45/COMPLETE ]"
  mkws c4 rawsrc; sixfn fn_setup mc >/dev/null 2>&1; dl >/dev/null 2>&1; sixsmoke >/dev/null 2>&1
  ( cd "$W" && Q6_TEST=1 Q6_TEST_PY="$HERE/test_python.sh" W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE=1 FAKE_FULL_LIMIT=2 bash queue_6x6.sh ) > "$W/main.out" 2>&1
  ck "full-step TIMEOUT: _PARTIAL package, no COMPLETE, cache kept" bash -c "grep -q 'full=TIMEOUT' $W/QUEUE6_STATUS && ls $W/queue_results/6x6_minicpm_v45/*PARTIAL*.tar.gz >/dev/null && [ ! -e $W/queue_results/6x6_minicpm_v45/COMPLETE ] && test -d $W/mpr-minicpm/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5"
  sixmain > "$W/main2.out" 2>&1
  ck "rerun resumes the unfinished full run and completes (COMPLETE, gate passed)" bash -c "test -e $W/queue_results/6x6_minicpm_v45/COMPLETE && grep -q '\"status\": \"passed\"' $(OUT_REL)/gate_6x6_full.json"
}
t_gpuwait() {
  echo "### D: GPU wait loop and 4090 check (real, without Q6_TEST), sleep shortened by a shim"
  mkws d rawsrc; sixfn fn_setup mc >/dev/null 2>&1
  mkdir -p "$W/shim"; printf '#!/bin/bash\nexec /bin/sleep 0.02\n' > "$W/shim/sleep"; chmod +x "$W/shim/sleep"
  ( cd "$W" && PATH="$W/shim:$PATH" FAKE_GPU_USED=9000 W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE=1 EXPECT_TORCH=2.9.1 bash queue_6x6.sh ) > "$W/main.out" 2>&1
  ck "GPU 9000 MiB busy: waits 30 loops then 'STOP: GPU still busy'" has "$W/QUEUE6_STATUS" 'STOP: GPU still busy (9000 MiB used)'
  rm -f "$W/QUEUE6_STATUS"
  ( cd "$W" && PATH="$W/shim:$PATH" W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE=1 EXPECT_TORCH=2.9.1 FAKE_NET_HF="6000000 15.0 206" bash queue_6x6.sh ) > "$W/main2.out" 2>&1
  ck "idle GPU passes the wait loop, reaches 'addon verified' (a real model run then fails: no GPU/model here; not exercised)" has "$W/QUEUE6_STATUS" 'addon verified'
  ck "FORCE_SLOW_NET=1 lets a 0.4 MB/s network check continue (the inner-call fix)" has "$W/QUEUE6_STATUS" 'FORCE_SLOW_NET=1: continuing'
  mkws d2 rawsrc; sixfn fn_setup mc >/dev/null 2>&1
  ( cd "$W" && W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 EXPECT_TORCH=2.9.1 FAKE_NET_HF="6000000 15.0 206" bash queue_6x6.sh ) > "$W/main.out" 2>&1
  ck "WITHOUT FORCE_SLOW_NET a slow check exits 1 with STOP (the problem the fix avoids)" has "$W/QUEUE6_STATUS" 'STOP: network too slow'
}
case "${1:-}" in
  all) t_rawsrc; t_download; t_failures; t_gpuwait ;;
  *) echo "usage: $0 all"; exit 2 ;;
esac
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
