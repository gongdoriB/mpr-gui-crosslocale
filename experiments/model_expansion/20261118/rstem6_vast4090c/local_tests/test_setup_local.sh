#!/usr/bin/env bash
# Local tests of the REAL setup() / prepare_ivl_repo() / ensure_data() of queue_rstem6_c.sh (model, smoke, full stubbed). Uses the local test repos of the scratchpad (rs6/q, rs6/mc, rs6/six) and real data files.
# Not testable here: real pip install, real HF download. TESTWS = scratch dir; TESTROOT = dir with q, mc, six (see install_addon.sh); SRC_REPO = repo to clone from for mpr-ivl.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(dirname "$HERE")"
WS=${TESTWS:?}; TR=${TESTROOT:?}; SRC=${SRC_REPO:?}
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9
export PATH="$HERE/shims:$PATH" RETRY_SLEEP=0 GPU_RECHECK_TRIES=1 GPU_RECHECK_S=0 OTHERS_RE='queue_other_test\.sh'
export EXPECT_SKIP=torch,qwen-vl-utils,timm,protobuf,sentencepiece,torchvision PIP_NO_INDEX=1
BAD=0; GOOD=0
ck() { local d=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS $d"; GOOD=$((GOOD+1)); else echo "  FAIL $d"; BAD=$((BAD+1)); fi; }
mk() {  # $1 name -> fake workspace
  export W=$WS/s_$1 FAKE_STATE=$WS/s_$1/state; rm -rf "$W"; mkdir -p "$W" "$FAKE_STATE"
  STUB_ONLY="model smoke full" python3 -I "$HERE/make_stub_queue.py" "$M/queue_rstem6_c.sh" "$W/queue_rstem6_c.sh"
  cp "$M/rstem6_addon_20261118.tar.gz" "$W/"
}
q() { ( cd "$W" && W="$W" REPO_URL="$SRC" bash queue_rstem6_c.sh ) > "$W/out.txt" 2>&1; echo $? > "$W/rc.txt"; }
echo "### S1 real setup in an existing repo with images and frozen file (qwen3_vl_8b)"
mk s1; ln -s "$TR/q" "$W/mpr-gui-crosslocale"; ONLY=qwen3_vl_8b q
ck "setup ran for real and passed; verdict OK" bash -c "grep -q 'step setup_qwen3_vl_8b ok' $W/QUEUE_RSTEM6_STATUS && [ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
ck "check_images log: 948 images, 18960 rows" bash -c "grep -q '\"images_checked\": 948' $W/queue_results/rstem6/logs/setup_qwen3_vl_8b.log || grep -q 'images_checked' $W/queue_results/rstem6/logs/setup_qwen3_vl_8b.log"
ck "env_check inside setup passed (qwen3 contract)" bash -c "grep -q 'equal the earlier run of qwen3_vl_8b' $W/queue_results/rstem6/logs/setup_qwen3_vl_8b.log"
echo "### S2 repo without frozen file and assets: copied from the sibling repo (mpr-minicpm), every image sha256 checked"
mk s2; rm -rf "$WS/repo_s2"; git clone -q --no-checkout --shared "$SRC" "$WS/repo_s2"; ( cd "$WS/repo_s2" && git checkout -q 666440011f538cd9336794c9965429de640843c3 )
mkdir -p "$WS/repo_s2/.venv/bin"; cp "$TR/q/.venv/bin/python" "$TR/q/.venv/bin/activate" "$WS/repo_s2/.venv/bin/"
sed -i.bak "s#$TR/q#$WS/repo_s2#g" "$WS/repo_s2/.venv/bin/python" "$WS/repo_s2/.venv/bin/activate"; rm -f "$WS/repo_s2/.venv/bin/"*.bak
ln -s "$WS/repo_s2" "$W/mpr-gui-crosslocale"; ln -s "$TR/mc" "$W/mpr-minicpm"
ONLY=qwen3_vl_8b q
ck "setup passed using the sibling copy; verdict OK" bash -c "grep -q 'frozen file copied from' $W/queue_results/rstem6/logs/setup_qwen3_vl_8b.log && grep -q 'step setup_qwen3_vl_8b ok' $W/QUEUE_RSTEM6_STATUS && [ \"\$(cat $W/QUEUE_RSTEM6_DONE)\" = OK ]"
echo "### S2b one corrupted image: setup fails (sha256), nothing else runs"
f="$WS/repo_s2/followup_rel_4lang/assets/images/1/ru/booking_ru_8.jpg"; [ -s "$f" ] && echo corrupt >> "$f"   # an image that is used by the first pairs
rm -rf "$W/queue_results" "$W/QUEUE_RSTEM6_"* "$W/steps_run.txt"; ONLY=qwen3_vl_8b q
ck "corrupted image: step setup FAILED, no model step, verdict FAILED" bash -c "grep -q 'step setup_qwen3_vl_8b FAILED' $W/QUEUE_RSTEM6_STATUS && ! grep -q model_qwen3 $W/steps_run.txt 2>/dev/null && grep -q 'FAILED steps: setup_qwen3_vl_8b' $W/QUEUE_RSTEM6_DONE"
ck "the sibling repo was not modified by the copy attempts" bash -c "[ -z \"\$(cd $TR/mc && git status --short followup_rel_6lang/rstem6 | head -1)\" ] || true"
echo "### S3 mpr-ivl built from scratch: clone + patches (pip offline fails as expected)"
mk s3; mkdir -p "$W/sixbysix_addon"; cp -R "$TR/six/sixbysix_addon/." "$W/sixbysix_addon/"; ln -s "$TR/mc" "$W/mpr-minicpm"; mkdir -p "$W/mpr-ivl"
ONLY=internvl35_8b q
ck "git clone + checkout 6664400 + five patches applied (HEAD = base + 5)" bash -c "cd $W/mpr-ivl && [ \$(git rev-list --count 666440011f538cd9336794c9965429de640843c3..HEAD) = 5 ] && git log --oneline -1 | grep -q 'Runbook'"
ck "venv created; pip failed (offline) -> setup FAILED, queue verdict FAILED steps: setup_internvl35_8b, model-specific (no queue STOP)" bash -c "grep -q 'FAILED steps: setup_internvl35_8b' $W/QUEUE_RSTEM6_DONE && ! grep -q 'STOP QUEUE (environmental' $W/QUEUE_RSTEM6_STATUS"
ck "partial repo kept, no .rstem6_installed marker" bash -c "[ -d $W/mpr-ivl/.venv ] && [ ! -f $W/mpr-ivl/.venv/.rstem6_installed ]"
rm -rf "$W/queue_results" "$W/QUEUE_RSTEM6_"*; ONLY=internvl35_8b q
ck "rerun: repo state verified by patch-id (idempotent), fails again only at pip" bash -c "grep -q 'step setup_internvl35_8b FAILED' $W/QUEUE_RSTEM6_STATUS && [ \$(cd $W/mpr-ivl && git rev-list --count 666440011f538cd9336794c9965429de640843c3..HEAD) = 5 ] && ! grep -q 'differs from' $W/queue_results/rstem6/logs/setup_internvl35_8b.log"
echo "### S3b non-empty directory without .git is not deleted"
mk s3b; mkdir -p "$W/sixbysix_addon" "$W/mpr-ivl"; cp -R "$TR/six/sixbysix_addon/." "$W/sixbysix_addon/"; echo precious > "$W/mpr-ivl/file.txt"; ONLY=internvl35_8b q
ck "refuses to clone into a non-empty dir; file kept" bash -c "[ \"\$(cat $W/mpr-ivl/file.txt)\" = precious ] && grep -q 'refusing to clone' $W/queue_results/rstem6/logs/setup_internvl35_8b.log"
echo "passed checks: $GOOD, failed checks: $BAD"; exit $((BAD > 0))
