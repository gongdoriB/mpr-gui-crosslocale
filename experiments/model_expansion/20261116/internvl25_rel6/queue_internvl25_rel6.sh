#!/usr/bin/env bash
# InternVL2.5-8B (OpenGVLab/InternVL2_5-8B), 6-language REL O/R/C/F (en, zh, th, ru, fr, ja; 52,572 condition rows -> 44,489 canonical inferences,
# batch size 1) on the Vast.ai RTX 4090, run as the LAST item of the queue chain:
#   wait for /workspace/MINICPM_DONE (written at the end of queue_minicpm_r2.sh) NEWER than WAIT_AFTER -> no other queue/overnight process -> GPU idle -> network check
#   -> checks -> setup in the EXISTING /workspace/mpr-ivl (addon unpack, fr/ja images, repo/venv/data checks) -> pinned model
#   -> smoke (--max-items 2) + gate -> REL6 full (44,489) + gate -> package -> delete this model's cache -> /workspace/QUEUE8_DONE
# Usage (inside tmux, in /workspace, with internvl25_rel6_addon_20261116.tar.gz next to this script; the fr/ja image tar is optional, see below):
#   bash queue_internvl25_rel6.sh --plan   # only print what would run; changes nothing
#   bash queue_internvl25_rel6.sh          # waits for a fresh MINICPM_DONE first
# Environment: WAIT_FOR (default $W/MINICPM_DONE), WAIT_AFTER (default 2026-11-16T02:11:00Z: a marker whose mtime is not later counts as absent), MAX_WAIT_H (default 168),
#   FORCE_SLOW_NET=1, IGNORE_ETA=1, KEEP_MODEL_CACHE=1, MIN_FREE_GB (default 30), NET_MIN_MBPS (default 20), FULL_H (full step limit in hours, default 10),
#   SUBSET=new_only (rows with fr or ja only; separate output folder; default all).
# fr/ja screenshots (729 files, 748 MB): if $W/internvl25_rel6_frja_images_20261116.tar is present it is verified (sha256) and unpacked, otherwise
#   they are downloaded from huggingface.co at the pinned dataset revision; either way every file is sha256-checked against bundle_manifest.json.
# Rerunning resumes (addon unpack is idempotent, predictions are reused). The instance is NOT stopped or destroyed.
# Progress: tail /workspace/QUEUE8_STATUS ; per-step logs: /workspace/queue_results/internvl25_rel6/logs/
# Written from queue_internvl25_expA.sh (r1 early-stop release, r2 process-name matching, r3 HF_HOME export are included).
# Only this model's cache folder is deleted; /workspace/mpr-ivl venv, other caches and all earlier results are not touched.
set -uo pipefail

W="${W:-/workspace}"
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3   # mpr-ivl = this commit + 5 InternVL3.5 patches (0001-0005); 0001 also supports internvl2_5
MODEL_ID=OpenGVLab/InternVL2_5-8B
REVISION=e9e4c0dc1db56bfab10458671519b7fa3dd29463      # HF main of 2025-03-25 (checked via HF API 2026-11-14); same as the 2026-10-05 REL run and the 2026-11-14 expA queue
MODEL_BYTES=16150800432                                 # sum of the four model-0000x-of-00004.safetensors (HF API)
ADDON=internvl25_rel6_addon_20261116.tar.gz
ADDON_SHA=a0209b50542605114117acd489c7fc5392b7ff559340072ac51f837d1bd4b100
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar     # optional
IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
SUBSET="${SUBSET:-all}"
case "$SUBSET" in all) N_FULL=44489; ROWS_FULL=52572; OUTNAME=rel6_internvl25_8b ;; new_only) N_FULL=26057; ROWS_FULL=29148; OUTNAME=rel6new_internvl25_8b ;; *) echo "SUBSET must be all or new_only"; exit 2 ;; esac
WAIT_FOR="${WAIT_FOR:-$W/MINICPM_DONE}"
WAIT_AFTER="${WAIT_AFTER:-2026-11-16T02:11:00Z}"        # start of the MiniCPM r2 run; markers of earlier (failed) runs are older and do not count
# r1 fallback (the queue that writes the marker stopped early without writing it): off unless WAIT_STATUS and WAIT_PROC are set
WAIT_STATUS="${WAIT_STATUS-}"; WAIT_PROC="${WAIT_PROC-}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"                        # model 16.2 GB x 1.3 + 8 GB for images, results (about 0.5 GB) and temp files
MAX_WAIT_H="${MAX_WAIT_H:-168}"
POLL_S="${POLL_S:-60}"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
export EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"      # only changed in local tests
T_SETUP=$((90*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_FULL=$(( ${FULL_H:-10} * 3600 ))
# Only processes whose command line STARTS with bash/sh count (a tmux server started as "tmux new-session ... bash run_all_overnight.sh" keeps that text forever).
# any queue script variant (queue_minicpm.sh, queue_minicpm_r2.sh, queue_6x6.sh, ...) except this script itself, also when it is renamed with a suffix
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_[A-Za-z0-9_]+\.sh}"   # OTHERS_RE: local test hook
SELF_RE='queue_internvl25_rel6[A-Za-z0-9_]*\.sh'
others_pids() {  # pids of other queue/overnight shells; this script (any name starting queue_internvl25_rel6) and this shell are not counted
  local p a
  for p in $(pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    a=$(ps -o args= -p "$p" 2>/dev/null) || continue
    echo "$a" | grep -q -E "$SELF_RE" && continue
    echo "$p"
  done
}
others_desc() { local p; for p in $(others_pids | head -2); do echo "$p $(ps -o args= -p "$p" 2>/dev/null | cut -c1-80);"; done; }
to_epoch() {  # ISO time (UTC) -> epoch seconds (GNU date, BSD date, else python3)
  date -u -d "$1" +%s 2>/dev/null || date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$1" +%s 2>/dev/null || python3 -c 'import sys,calendar,time;print(calendar.timegm(time.strptime(sys.argv[1],"%Y-%m-%dT%H:%M:%SZ")))' "$1"
}
mtime_of() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
AFTER_EPOCH=$(to_epoch "$WAIT_AFTER") || AFTER_EPOCH=""
marker_fresh() {  # the marker exists and its mtime is later than WAIT_AFTER
  [ -e "$WAIT_FOR" ] || return 1
  [ -n "$AFTER_EPOCH" ] || return 1
  [ "$(mtime_of "$WAIT_FOR")" -gt "$AFTER_EPOCH" ] 2>/dev/null
}

R="$W/mpr-ivl"
RES="$W/queue_results/internvl25_rel6"
LOG="$RES/logs"
ST="$W/QUEUE8_STATUS"; DONE="$W/QUEUE8_DONE"
OUTD="$R/followup_rel_6lang/results/$OUTNAME"
CACHE_DIR="$R/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
DAY=$(date -u +%Y%m%d)
FAILED=()
FINISHED=0

PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1

say() { local l; l="$(date -u +%FT%TZ) ivl25_rel6: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLANEOF
Plan (nothing is executed):
 0. wait for $WAIT_FOR with mtime later than $WAIT_AFTER (an older marker counts as absent; any content; poll ${POLL_S}s, give up after ${MAX_WAIT_H}h),
    then while any process matching [$OTHERS] (not this script) still runs,
    then until GPU memory used < 1.5 GB (max 30 min), then network check (curl ${NET_SECS}s ranged download from huggingface.co (the model's first
    shard) and download.pytorch.org (informational); STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1; MB/s go to $ST)
 1. checks: $ADDON present and sha256 = $ADDON_SHA; ${IMAGES_TAR} (optional) sha256 = $IMAGES_SHA if present; GPU is RTX 4090; free disk >= ${MIN_FREE_GB} GB;
    $R exists with .git and .venv
 2. setup (limit $((T_SETUP/60)) min): repo is a descendant of $BASE_COMMIT and carries the InternVL patches; the addon may only create files under
    01_code/src/mpr_crosslocale/{data,inference}/*_6lang.py and followup_rel_6lang/ (listing checked before unpacking), unpack it into $R, verify its SHA256SUMS;
    unpack the image tar if present; venv has torch 2.9.1+cu130, transformers 4.57.6, timm, einops, sentencepiece (else pip install with the repo's constraints file);
    config in the addon is byte-identical to 01_code/configs/models/internvl2_5_8b.yaml; 4-language images verify (fetch if missing);
    6-language data + fr/ja images verify by sha256 (fetch the missing fr/ja images from the pinned HF revision otherwise)
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION (~16.2 GB, 3 tries, partial files resume); safetensors bytes = $MODEL_BYTES;
    config.json llm_config.architectures = InternLM2ForCausalLM; tokenizer loads and A, B, C, D are single tokens (ids 290, 309, 289, 302 as in the earlier run)
 4. smoke (--max-items 2, subset=$SUBSET) (limit $((T_SMOKE/60)) min) + gate (rules: all rows ok, parse failures <= 50%, labels not all identical, A-D mass >= 0.5,
    visual tokens 768..1792, english_fixed system prompt replaced the default Chinese one in every rendered prompt)
 5. REL6 full: $N_FULL inferences / $ROWS_FULL condition rows (limit $((T_FULL/3600)) h; not started if the smoke ETA exceeds it, IGNORE_ETA=1 overrides) + gate
    (all $N_FULL rows ok, summary statuses = $ROWS_FULL success)
 6. package to $RES/internvl25_rel6_results_<date>.tar.gz; delete only $CACHE_DIR on success (kept after a failure); append to $ST;
    write $DONE (OK or FAILED steps: ...). Results go to $OUTD (never into followup_rel_4lang/results).
PLANEOF
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue8_ivl25_rel6.lock"; flock -n 9 || { echo "another queue_internvl25_rel6.sh is already running"; exit 1; }; }

retry() {  # retry <tries> <cmd...>: network steps on flaky hosts (SSL EOF / gnutls handshake errors seen on the previous server)
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "retry $i/$n failed: $*" >&2
    [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i ))
  done
  return 1
}

step() {  # step <name> <limit_s> <function>; runs the function in a subshell with a time limit
  local name=$1 limit=$2 fn=$3 rc
  say "step $name started (limit $((limit/60)) min)"
  timeout -k 60 "$limit" bash -c "set -Eeo pipefail; $fn" >> "$LOG/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "step $name ok"; return 0; fi
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then say "step $name TIMEOUT"
  elif [ $rc -eq 3 ] && [ "$name" = full_rel6 ]; then say "STOP step $name: smoke ETA longer than the step limit (see $OUTD/STATUS). IGNORE_ETA=1 overrides."; FAILED+=("${name}_eta"); return $rc
  else say "step $name FAILED (exit $rc), see $LOG/$name.log"; fi
  FAILED+=("$name"); return $rc
}

package() {
  local out="$RES/internvl25_rel6_results_${DAY}.tar.gz" stage="$RES/_stage"
  rm -rf "$stage"; mkdir -p "$stage"
  if [ -d "$R" ]; then
    cp -r "$OUTD" "$stage/" 2>/dev/null || true
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -7 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
  fi
  cp -r "$LOG" "$stage/queue_logs" 2>/dev/null || true
  cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true
  df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  tar -czf "$out" -C "$stage" . && sha256sum "$out" > "$out.sha256" && say "packaged $out ($(du -h "$out" | cut -f1))"
  rm -rf "$stage"
}

finish() {  # runs once, also on error exits
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  package || say "packaging FAILED"
  local verdict="OK"; [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  if [ -d "$CACHE_DIR" ]; then
    if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "model cache kept (KEEP_MODEL_CACHE=1)"
    elif [ "$verdict" = OK ]; then rm -rf "$CACHE_DIR"; say "model cache deleted (run finished); free disk: $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
    else say "model cache KEPT (not finished). A rerun resumes without downloading."; fi
  fi
  say "FINISHED $verdict. Results: $RES/internvl25_rel6_results_${DAY}.tar.gz"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE8_DONE written, rerun the script to resume"; FINISHED=1; exit 130' INT TERM

abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }   # abort <key> <message>; key = name in "FAILED steps:"

# ---- 0. wait for the earlier queues ------------------------------------------------------------
rm -f "$DONE"
[ -n "$AFTER_EPOCH" ] || abort wait "WAIT_AFTER '$WAIT_AFTER' is not a valid UTC time like 2026-11-16T02:11:00Z"
say "queue_internvl25_rel6 started (subset $SUBSET); waiting for $WAIT_FOR newer than $WAIT_AFTER; marker now: $(if marker_fresh; then echo "fresh"; elif [ -e "$WAIT_FOR" ]; then echo "an older marker exists and is ignored (mtime $(date -u -r "$(mtime_of "$WAIT_FOR")" +%FT%TZ 2>/dev/null || mtime_of "$WAIT_FOR"))"; else echo absent; fi)"
waited=0
while ! marker_fresh; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  if [ $((waited % 1800)) -lt "$POLL_S" ]; then
    if [ -n "$WAIT_PROC" ] && pgrep -f "^(/bin/)?(ba)?sh .*${WAIT_PROC}" >/dev/null 2>&1; then q7=running; else q7="NOT running (stopped early? check)"; fi
    say "still waiting for $(basename "$WAIT_FOR") newer than $WAIT_AFTER ($([ -e "$WAIT_FOR" ] && echo "an older marker exists and is ignored" || echo "absent"); ${waited}s); the queue that writes it: $q7; its last status: $(tail -1 "${WAIT_STATUS:-/dev/null}" 2>/dev/null | cut -c1-160)"
  fi
  # r1: a queue that STOPs early may write no DONE file. If it is not running and its last status line is a STOP or an interrupt or FINISHED, count it;
  # after 2 such checks in a row treat it as finished. The OTHERS/GPU-idle checks below still apply.
  if [ -n "$WAIT_STATUS" ] && [ -n "$WAIT_PROC" ] && ! pgrep -f "^(/bin/)?(ba)?sh .*${WAIT_PROC}" >/dev/null 2>&1 && tail -1 "$WAIT_STATUS" 2>/dev/null | grep -q -E 'STOP|interrupted|FINISHED'; then
    q7stop=$(( ${q7stop:-0} + 1 ))
    if [ "$q7stop" -ge 2 ]; then say "the queue writing $(basename "$WAIT_FOR") is not running and ended with: $(tail -1 "$WAIT_STATUS" | cut -c1-160); continuing without $(basename "$WAIT_FOR")"; break; fi
  else q7stop=0; fi
  [ "$waited" -ge "${MAX_WAIT_S:-$((MAX_WAIT_H*3600))}" ] && abort wait "$(basename "$WAIT_FOR") did not appear within ${MAX_WAIT_H}h"   # MAX_WAIT_S: local test hook
done
if marker_fresh; then say "$(basename "$WAIT_FOR") found (mtime $(date -u -r "$(mtime_of "$WAIT_FOR")" +%FT%TZ 2>/dev/null || mtime_of "$WAIT_FOR") is later than $WAIT_AFTER): $(head -c 200 "$WAIT_FOR" | tr '\n' ' ')"
elif [ -e "$WAIT_FOR" ]; then say "$(basename "$WAIT_FOR") exists but is not newer than $WAIT_AFTER; continuing only because the early-stop fallback released the wait"; fi
[ -n "$(others_pids)" ] && say "another queue/overnight process is still running ($(others_desc)); waiting for it to end"
while [ -n "$(others_pids)" ]; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  [ "$waited" -ge $((MAX_WAIT_H*3600)) ] && abort others "another queue/overnight process still running after ${MAX_WAIT_H}h"
done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
  [ -z "$used" ] && abort gpu "nvidia-smi not available"
  [ "$used" -lt 1500 ] && break
  sleep "${GPU_POLL_S:-60}"
done
[ "${used:-99999}" -lt 1500 ] || abort gpu "GPU still busy (${used} MiB used)"

# ---- 0b. network check (same method as the earlier queues): 15 s ranged download -------------------
net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}
net_check() {  # download.pytorch.org is measured and logged only; torch is not downloaded unless the venv is broken
  local hf pt slow=0
  hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$MODEL_ID/resolve/$REVISION/model-00001-of-00004.safetensors}")
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${hf} MB/s, download.pytorch.org ${pt} MB/s; minimum $NET_MIN_MBPS MB/s (huggingface.co decides)"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$hf" "$pt" > "$RES/net_check.txt"
  case "$hf" in
    ERR:4*) say "WARNING: the huggingface test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;;
    ERR*) slow=1 ;;
    *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
  esac
  if [ "$slow" = 1 ]; then
    if [ "${FORCE_SLOW_NET:-0}" = 1 ]; then say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; fi
    abort network "network too slow or unreachable (need >= $NET_MIN_MBPS MB/s from huggingface.co). Nothing was downloaded. Use another instance, or FORCE_SLOW_NET=1"
  fi
}
net_check

# ---- 1. checks --------------------------------------------------------------------------------
[ -s "$W/$ADDON" ] || abort addon "$ADDON missing in $W"
echo "$ADDON_SHA  $W/$ADDON" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$ADDON sha256 differs"
if [ -s "$W/$IMAGES_TAR" ]; then
  echo "$IMAGES_SHA  $W/$IMAGES_TAR" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$IMAGES_TAR sha256 differs (delete it to fetch the images from huggingface.co instead)"
  HAVE_IMAGES_TAR=1
else
  HAVE_IMAGES_TAR=0; say "no $IMAGES_TAR next to the script: fr/ja screenshots (748 MB) will be downloaded from huggingface.co and sha256-checked"
fi
export HAVE_IMAGES_TAR
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort gpu "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
df -h "$W" > "$LOG/disk_start.log"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || [ -s "$CACHE_DIR/snapshots/$REVISION/config.json" ] || abort disk "only ${FREE_GB} GB free in $W (need ${MIN_FREE_GB}+)"
{ [ -d "$R/.git" ] && [ -f "$R/.venv/bin/activate" ]; } || abort repo "$R (repo with .venv) does not exist; it is created by the overnight job / queue_6x6.sh (InternVL3.5). Not rebuilt here"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free"

# ---- 2. setup -------------------------------------------------------------------------------
setup() {
  cd "$R"
  git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null
  git merge-base --is-ancestor "$BASE_COMMIT" HEAD
  echo "HEAD $(git rev-parse --short HEAD), $(git rev-list --count "$BASE_COMMIT"..HEAD) commits after ${BASE_COMMIT:0:7}"
  grep -q 'INTERNVL_FAMILIES' 01_code/src/mpr_crosslocale/inference/run_cross_locale.py          # patch 0001: internvl2_5 and 3.5 share the adapter path
  grep -q 'self.model.system_message if system_prompt is None' 01_code/src/mpr_crosslocale/models/internvl25.py   # patch 0001: an explicit system prompt REPLACES the default
  # the addon may only add files of its own: refuse anything else before unpacking (tar would overwrite silently)
  tar -tzf "$W/$ADDON" > "$RES/addon_listing.txt"
  if grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | grep -q .; then
    echo "addon contains paths outside its own files:"; grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | head; exit 1
  fi
  tar -xzf "$W/$ADDON"
  ( cd followup_rel_6lang/ivl25_rel6 && sha256sum -c SHA256SUMS )
  cmp followup_rel_6lang/ivl25_rel6/internvl2_5_8b.yaml 01_code/configs/models/internvl2_5_8b.yaml
  if [ "$HAVE_IMAGES_TAR" = 1 ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" PIP_CACHE_DIR="$R/.cache/pip" PIP_NO_CACHE_DIR=1
  source .venv/bin/activate
  venv_ok() { REPO_DIR="$PWD" python - <<'PY' >/dev/null 2>&1
import os, torch, transformers, timm, einops, sentencepiece, mpr_crosslocale
assert torch.__version__ == os.environ["EXPECT_TORCH"], torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"]))
PY
  }
  if ! venv_ok; then
    echo "venv check failed; installing with the repo's pins"
    C=followup_rel_4lang/rstem_ropts_inputs/internvl35_constraints_20261109.txt
    retry 3 python -m pip install -r 01_code/requirements-qwen3-4090.txt -c "$C"
    retry 3 python -m pip install -e './01_code[models]' -c "$C"
    venv_ok
  fi
  python -c 'import torch,transformers,timm,sentencepiece;print("torch",torch.__version__,"transformers",transformers.__version__,"timm",timm.__version__,"sentencepiece",sentencepiece.__version__)'
  python -c 'import mpr_crosslocale.data.rel_followup_6lang, mpr_crosslocale.inference.run_rel_followup_6lang; print("6-language modules import ok")'
  # 4-language screenshots (en/zh/th/ru) stay where the earlier runs put them; nothing is copied
  python followup_rel_4lang/data_bundle.py verify || { python followup_rel_4lang/data_bundle.py unpack; retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2; python followup_rel_4lang/data_bundle.py verify; }
  # 6-language data + fr/ja screenshots (own) + en/zh/th/ru screenshots (legacy root) by sha256
  python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang > "$LOG/bundle6_verify_first.log" 2>&1 \
    || { echo "6-language verify failed (see bundle6_verify_first.log); fetching the fr/ja screenshots"; retry 6 python followup_rel_6lang/data_bundle_6lang.py fetch-images --data-root followup_rel_6lang --workers 2; python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang; }
}

model() {
  cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
  source .venv/bin/activate
  # 3 tries; partial files stay in the cache and hf resumes them (also after a timeout of this step)
  retry 3 hf download "$MODEL_ID" --revision "$REVISION"
  MODEL_ID="$MODEL_ID" REVISION="$REVISION" MODEL_BYTES="$MODEL_BYTES" RES="$RES" python - <<'PY'
import glob, json, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MODEL_ID"], revision=os.environ["REVISION"], local_files_only=True)
fs = glob.glob(os.path.join(snap, "model-*.safetensors"))
size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) == 4 and size == int(os.environ["MODEL_BYTES"]), size
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg["llm_config"]["architectures"]
print("llm_config.architectures", arch, "| _name_or_path", cfg["llm_config"].get("_name_or_path"), "| template", cfg.get("template"))
assert arch == ["InternLM2ForCausalLM"] and cfg.get("template") == "internvl2_5", (arch, cfg.get("template"))
open(os.path.join(os.environ["RES"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\nllm={arch} {cfg['llm_config'].get('_name_or_path')}\n")
# fail fast on a tokenizer / transformers mismatch before the GPU is used (same loader as the adapter)
from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer
tok = load_internvl_tokenizer(os.environ["MODEL_ID"], revision=os.environ["REVISION"], trust_remote_code=True)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
assert all(len(v) == 1 for v in ids.values()), ids
assert {k: v[0] for k, v in ids.items()} == {"A": 290, "B": 309, "C": 289, "D": 302}, ids   # ids recorded in the 2026-10-05 REL run
PY
  du -sh "$HF_HOME"
}

# Vast sets HF_HOME=/workspace/.hf_home in /etc/environment; the run script keeps an inherited HF_HOME, so the cache variables are exported here
# (exports are inline: steps run in a fresh bash that only gets the step function itself)
smoke_rel6() { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; SUBSET="$SUBSET" bash followup_rel_6lang/ivl25_rel6/run_rel6_ivl25.sh smoke; }
full_rel6()  { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; SUBSET="$SUBSET" REL_LIMIT_S="$T_FULL" bash followup_rel_6lang/ivl25_rel6/run_rel6_ivl25.sh full; }
export -f retry setup model smoke_rel6 full_rel6
export W R RES LOG BASE_COMMIT ADDON IMAGES_TAR MODEL_ID REVISION MODEL_BYTES T_FULL SUBSET

smoke_report() {  # accuracy, speed, ETA and warnings from the smoke gate
  python3 - "$OUTD/smoke/gate_rel_smoke.json" <<'PY' 2>/dev/null
import json, sys
try:
    g = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
line = f"rel6: acc {s.get('accuracy')} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')}"
if g.get("warnings"):
    line += " WARN " + "; ".join(g["warnings"])
if g.get("problems"):
    line += " PROBLEMS " + "; ".join(g["problems"])
print(line, end=" | ")
PY
}

if step setup "$T_SETUP" setup && step model "$T_MODEL" model; then
  if step smoke_rel6 "$T_SMOKE" smoke_rel6; then
    say "smoke gate: $(smoke_report)"
    step full_rel6 "$T_FULL" full_rel6 || true
  else
    say "smoke failed: $(smoke_report)see $LOG/smoke_rel6.log"
  fi
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
