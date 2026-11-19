#!/usr/bin/env bash
# vast4090b (ONE RTX 4090): complete 6-language results for the two supplementary 4B models. Starts only AFTER the B2 queue (queue_b2_frja.sh) has finished.
# Order (env ONLY="internvl35_4b" selects a subset, order kept; env ONLY_6X6=1 runs only part c, SKIP_6X6=1 only parts a+b):
#   model 1 qwen3_vl_4b   [repo /workspace/mpr-gui-crosslocale]:  a) REL O/R/C/F new_only (26,057 inferences / 29,148 rows)   b) Exp A new pairs, main 2,844 x O/T/R = 8,532 + T_retry 148 x 3 = 444
#   model 2 internvl35_4b [repo /workspace/mpr-ivl]:               a) and b) as above
#   then c) the 6x6 full benchmark (79,020 unique inferences) for model 1, then for model 2 = the UNCHANGED queue_6x6.sh with MODELS=<key> (same call as the earlier 8B / sizes runs)
# Every model uses the conditions of its earlier 4-language run (2026-11-14, sizes_uitars queue): batch 1, strict_batch_size true, its config yaml, system prompt english_fixed, code hashes, package versions
# (checked against the earlier run_contract.json files shipped in the addon; nothing is installed). Results: <repo>/followup_rel_6lang/results/{rel6new_<KEY>,expA6_<KEY>} and <repo>/followup_rel_4lang/results/6x6_<KEY> (new folders only).
# Usage (tmux, in /workspace, with the 4 upload files next to this script):  bash queue_b3_4b.sh --plan | bash queue_b3_4b.sh
# Waits (no changes, no downloads) until /workspace/QUEUE_B2_DONE exists AND no other queue script runs AND no GPU compute process AND GPU memory < 1,500 MiB, twice in a row (env POLL_S 60, WAIT_MAX_H 96).
# Env: ONLY, ONLY_6X6, SKIP_6X6, KEEP_MODEL_CACHE=1, FORCE_SLOW_NET=1, IGNORE_ETA=1, NET_MIN_MBPS (20), REL_H / EXPA_H (full-step limits in hours; defaults per model below), S6_H (limit of one queue_6x6.sh call),
#   SKIP_6X6_DATA_CHECK=1 (do not stop when the existing 6x6 raw data differs from the pinned revision), NO_WAIT=1 (do not wait for QUEUE_B2_DONE; the GPU/other-queue checks still apply).
# FAILURE RULE (as queue_b2_frja.sh): a failing step stops THAT model's remaining steps (and its 6x6); the queue goes on with the next model when the failure is model-specific. It STOPS the whole queue (verdict FAILED, rest not started)
#   when the failure is environmental: GPU busy / not a 4090, another queue running, network too slow, addon sha mismatch, repo or venv missing, setup failure (venv / code hash / config identity), disk too low for the next download,
#   a CUDA fatal error text in a step log, or a failed GPU/disk re-check after a failed model.
# Status /workspace/QUEUE_B3_STATUS, done marker /workspace/QUEUE_B3_DONE (OK | FAILED steps: ...), packages in /workspace/queue_results/b3_4b/<KEY>/ (after EACH major step, with sha256 + per-file sha list).
# Rerun = resume (finished steps are skipped, predictions resume).
set -uo pipefail
W="${W:-/workspace}"
ADDON=b3_4b_addon_20261117.tar.gz
ADDON_SHA=20155e4a801c0e92597b853c3eabda2bfc9c73b10be676b34868ca25eb9f51fd
SIX_ADDON=sixbysix_addon_20261114.tar.gz;   SIX_ADDON_SHA=ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a
SIX_QUEUE=queue_6x6.sh;                     SIX_QUEUE_SHA=d538b0cd799a653d1cc536716e953168db86a15c3c346cbe659bd41a833a805a
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar     # optional (the fr/ja images are normally already unpacked in /workspace/mpr-ivl)
IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3
ALL_MODELS="qwen3_vl_4b internvl35_4b"
ORDER=""; for k in $ALL_MODELS; do case " ${ONLY:-$ALL_MODELS} " in *" $k "*) ORDER="$ORDER $k" ;; esac; done; ORDER="${ORDER# }"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"; MIN_DISK_ALWAYS_GB="${MIN_DISK_ALWAYS_GB:-5}"; KEEP_MARGIN_GB="${KEEP_MARGIN_GB:-10}"
POLL_S="${POLL_S:-60}"; WAIT_MAX_H="${WAIT_MAX_H:-96}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
T_SETUP=$((60*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_RETRY=$((60*60)); T_SIXSETUP=$((3*3600))
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_[A-Za-z0-9_]+\.sh}"; SELF_RE='queue_b3_[A-Za-z0-9_]*\.sh'
RES="$W/queue_results/b3_4b"; LOG="$RES/logs"
ST="$W/QUEUE_B3_STATUS"; DONE="$W/QUEUE_B3_DONE"; B2DONE="${B2DONE:-$W/QUEUE_B2_DONE}"
DAY=$(date -u +%Y%m%d)
FAILED=(); NOTRUN=(); FINISHED=0
setm() { eval "MSTAT_$1=\"\$2\""; }; getm() { eval "echo \"\${MSTAT_$1:-}\""; }   # per-model status (no associative arrays: bash 3.2 on the test Mac)

minfo() {  # minfo KEY -> sets the per-model facts. Estimates (s/inference) = medians of inference_runtime_ms of the earlier 4090 runs of 2026-11-14 on this server (estimates, not measured on this run).
  KEY=$1
  case "$KEY" in
    qwen3_vl_4b)   MID=Qwen/Qwen3-VL-4B-Instruct; REV=ebb281ec70b05090aa6165b016eac8ec08e71b17; REPO_NAME=mpr-gui-crosslocale; FAM=qwen; MINFREE=20; MIN_SHARDS=2; BYTES_MIN=8875719344; BYTES_MAX=8875719344
                   T_REL=$(( ${REL_H:-6} * 3600 )); T_EXPA=$(( ${EXPA_H:-3} * 3600 )); S6_LIMIT=$(( ${S6_H:-40} * 3600 )); LABELS="32,33,34,35"; TOKLOADER=auto; ARCH_CHECK=qwen3vl; EST_REL="0.448"; EST_EXPA="0.455" ;;
    internvl35_4b) MID=OpenGVLab/InternVL3_5-4B; REV=481f6e32467eab4e922ccd7fd6cf420441a62331; REPO_NAME=mpr-ivl; FAM=ivl; MINFREE=20; MIN_SHARDS=1; BYTES_MIN=8000000000; BYTES_MAX=10500000000
                   T_REL=$(( ${REL_H:-8} * 3600 )); T_EXPA=$(( ${EXPA_H:-3} * 3600 )); S6_LIMIT=$(( ${S6_H:-42} * 3600 )); LABELS="32,33,34,35"; TOKLOADER=internvl; ARCH_CHECK=none; EST_REL="0.669"; EST_EXPA="0.645" ;;
    *) echo "unknown model $KEY"; exit 2 ;;
  esac
  HAS_REL=1; SCOPE=new; N_REL=26057; N_EXPA=8532; N_RETRY=444; EXPA_BATCH_V=1
  R="$W/$REPO_NAME"; CACHE_DIR="$R/.cache/huggingface/hub/models--${MID//\//--}"
  OUT_REL="$R/followup_rel_6lang/results/rel6new_$KEY"; OUT_EXA="$R/followup_rel_6lang/results/expA6_$KEY"; OUT_6X6="$R/followup_rel_4lang/results/6x6_$KEY"
  return 0
}

for k in $ORDER; do ( minfo "$k" ) || exit 2; done
PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1
say() { local l; l="$(date -u +%FT%TZ) b3_4b: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLAN
Plan (nothing is executed, nothing is changed). Models in this order: ${ORDER}   (ONLY_6X6=${ONLY_6X6:-0}, SKIP_6X6=${SKIP_6X6:-0})
 0. upload files: sha256 of $ADDON ($ADDON_SHA), $SIX_ADDON, $SIX_QUEUE (and the optional $IMAGES_TAR). Then flock against a second copy.
    WAIT (no changes, no downloads): until $B2DONE exists AND no other queue/overnight script [$OTHERS] runs AND no GPU compute process AND GPU memory used < ${GPU_BUSY_MIB} MiB, twice in a row, polling every ${POLL_S} s, at most ${WAIT_MAX_H} h.
    Then the same checks once more as a refusal (nothing is started when they fail), RTX 4090 check, network check (${NET_SECS}s ranged download from huggingface.co; STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1).
 per model, parts a and b:
   setup (limit $((T_SETUP/60)) min): addon files are copied into the repo only where no file exists yet (an existing file must be byte-identical, else STOP; nothing is overwritten); addon SHA256SUMS; repo is a descendant of $BASE_COMMIT;
     venv packages, inference-code hashes, config yaml and system prompt equal the earlier 4B run contracts (nothing is installed); 4-/6-language data + images verify by sha256 (fr/ja images: already in the repo, else copied from /workspace/mpr-ivl, else the image tar, else fetched from the pinned HF revision)
   disk check before the download (need MINFREE GB, skipped when cached; the cache of an earlier model that is kept for its 6x6 is removed first when free disk < MINFREE + ${KEEP_MARGIN_GB} GB)
   model (limit $((T_MODEL/60)) min): hf download @ pinned revision (3 tries); safetensors bytes; config architecture; tokenizer A-D ids 32..35 (as in the earlier runs' label_token_ids)
   smoke + gate for REL and for Exp A (limit $((T_SMOKE/60)) min each), ETA check against the step limit, full runs with --resume + completion gates; package after REL (01_rel) and after Exp A + T_retry (02_expA)
 part c (per model, unchanged queue_6x6.sh): addon unpack + sha, repo state guard (qwen repo HEAD must be exactly ${BASE_COMMIT:0:7}; InternVL repo = ${BASE_COMMIT:0:7} + 5 patches; venv imports; nothing is installed or checked out by us),
   6x6 raw data verified when present (HF revision c1edb808, 3,685 files, GitHub e4f1cfcd, manifest sha 4dd919fc, every manifest image exists) and reused, else downloaded (2.6 GB, rate-limit retries) by queue_6x6.sh;
   then MODELS=<key> bash $SIX_QUEUE: pinned download (skipped when cached), dry-run plan 79,020 + ID-list sha, smoke 216 + gate, full 79,020 (batch 1, --score-labels, english_fixed, seed 42, dimensions wf wi au ap ael rel) + gate + summary, package, cache deleted on success. package 03_6x6.
 final package 00_final (all logs, status, sha list of every package), then $DONE.
 (1) qwen3_vl_4b    REL new_only 26,057 (strict true) + Exp A 8,532 + T_retry 444 (batch 1, strict true)           6x6 79,020
 (2) internvl35_4b  REL new_only 26,057 (strict true) + Exp A 8,532 + T_retry 444 (batch 1, strict true)           6x6 79,020
 Estimated times (estimates from the medians of the earlier 4090 runs of 2026-11-14, NOT measured here; the 6x6 per-inference time is ASSUMED equal to the REL median):
   qwen3_vl_4b    REL 26,057 x 0.448 s = 3.2 h; Exp A 8,532 x 0.455 = 1.1 h (+ 444 = 3 min); 6x6 79,020 x 0.448 = 9.8 h
   internvl35_4b  REL 26,057 x 0.669 s = 4.8 h; Exp A 8,532 x 0.645 = 1.5 h (+ 444 = 5 min); 6x6 79,020 x 0.669 = 14.7 h
   plus 2 x (setup + 9 GB download + load + smokes) and 2 x 6x6 setup/smoke: about 2 h. Total about 36-38 h (a+b of both models about 11 h, then 6x6 about 25 h).
 Step limits (hours): REL ${REL_H:-qwen 6 / internvl 8}, Exp A ${EXPA_H:-3}, one queue_6x6.sh call ${S6_H:-qwen 40 / internvl 42}. A step whose smoke ETA exceeds its limit is not started.
 Disk (34-50 GB free at the start): model caches 8.9 GB (qwen3_vl_4b, exact 8,875,719,344 bytes) and about 9.5 GB (internvl35_4b); the first cache is kept for its 6x6 while free disk allows; results+packages about 0.5-1.5 GB per model for a+b, 6x6 about 0.4 GB per model; 6x6 raw data 2.6 GB only when not present.
 Failure rule: see the header of this script (model-specific failure: continue with the next model; environmental failure: stop the queue).
PLAN
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue_b3.lock"; flock -n 9 || { echo "another queue_b3_4b.sh is already running"; exit 1; }; }
retry() { local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n failed: $*" >&2; [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i )); done; return 1; }
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc 0-9; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -i 'no running' | sed '/^$/d'; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
others_running() {  # prints "pid:cmd ..." of other queue/overnight scripts (not this one)
  local p a o=""
  for p in $(pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" 2>/dev/null); do [ "$p" = "$$" ] && continue; a=$(ps -o args= -p "$p" 2>/dev/null) || continue; echo "$a" | grep -q -E "$SELF_RE" && continue; o="$o $p:$(echo "$a" | cut -c1-60)"; done
  echo "$o"
}

CUDA_FATAL='CUDA error|device-side assert|Xid|NVML|GPU is lost|unable to determine the device handle|CUBLAS_STATUS|illegal memory access'
step_failed=0; ENV_FAIL=0; ENV_REASON=""
step() {  # step <name> <limit_s> <function> ; sets step_failed=1 on failure; sets ENV_FAIL=1 when the failure is environmental
  local name=$1 limit=$2 fn=$3 rc
  say "step ${name}_$KEY started (limit $((limit/60)) min)"
  timeout -k 60 "$limit" bash -c "set -Eeo pipefail; $fn" >> "$LOG/${name}_$KEY.log" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then say "step ${name}_$KEY ok"; return 0; fi
  step_failed=1
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then say "step ${name}_$KEY TIMEOUT"; FAILED+=("${name}_${KEY}_timeout")
  elif [ $rc -eq 3 ] && [[ "$name" == full_* ]]; then say "STOP step ${name}_$KEY: smoke ETA longer than the step limit (IGNORE_ETA=1 overrides)."; FAILED+=("${name}_${KEY}_eta")
  else say "step ${name}_$KEY FAILED (exit $rc), see $LOG/${name}_$KEY.log"; FAILED+=("${name}_$KEY"); fi
  if [ "$name" = setup ]; then ENV_FAIL=1; ENV_REASON="setup of $KEY failed (repo/venv/code/config/data identity)"; fi
  if tail -n 200 "$LOG/${name}_$KEY.log" 2>/dev/null | grep -q -E "$CUDA_FATAL"; then ENV_FAIL=1; ENV_REASON="CUDA fatal error text in $LOG/${name}_$KEY.log"; fi
  return $rc
}

package() {  # package <KEY> <label> <suffix> <result dir names under followup_rel_6lang/results ...> : tar.gz + .sha256 + per-file sha list (inside and next to it); sets PKG_OK
  local k=$1 label=$2 suf=$3; shift 3
  local dest="$RES/$k" stage="$RES/_stage_${k}_$label" dirs=() d out
  minfo "$k"; mkdir -p "$dest"
  out="$dest/b3_${k}_${label}_${DAY}${suf}.tar.gz"
  rm -rf "$stage"; mkdir -p "$stage"
  for d in "$@"; do [ -d "$R/followup_rel_6lang/results/$d" ] && dirs+=("followup_rel_6lang/results/$d"); done
  if [ -d "$R" ]; then
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    [ ${#dirs[@]} -gt 0 ] && (cd "$R" && find "${dirs[@]}" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true
  fi
  mkdir -p "$stage/queue_logs"; cp "$LOG/"*"_$k.log" "$stage/queue_logs/" 2>/dev/null || true; cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  cp "$RES"/model_size_"$k".txt "$RES"/net_check.txt "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  if [ ${#dirs[@]} -gt 0 ]; then tar -czf "$out" -C "$R" "${dirs[@]}" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && tar -tzf "$out" > /dev/null && (cd "$dest" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" && sha256sum "$(basename "$out")" >> "$RES/PACKAGES_SHA256.txt") \
    && { [ -s "$stage/results_files.sha256" ] && cp "$stage/results_files.sha256" "${out%.tar.gz}.files.sha256"; true; } \
    && say "$k packaged ($label): $out ($(du -h "$out" | cut -f1)); sha256 $(cut -d' ' -f1 "$out.sha256")"
  local rc=$?; rm -rf "$stage"; PKG_OK=$((rc == 0)); [ $rc -eq 0 ] || { say "$k packaging $label FAILED"; FAILED+=("package_${label}_$k"); }
  return $rc
}

finish() {
  [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  local verdict="OK" k line=""
  [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  [ ${#FAILED[@]} -gt 0 ] && [ ${#NOTRUN[@]} -gt 0 ] && verdict="$verdict | not run: ${NOTRUN[*]}"
  for k in $ORDER; do v=$(getm "$k"); line="$line $k=${v:-not_started}"; done
  say "models:$line"
  # final package: every log, the status files and the sha list of all packages
  if [ -d "$RES" ]; then
    local fs="$RES/_stage_final" out="$RES/b3_4b_00_final_${DAY}.tar.gz"
    rm -rf "$fs"; mkdir -p "$fs/queue_logs"; cp -R "$LOG/." "$fs/queue_logs/" 2>/dev/null || true; cp "$RES/STATUS" "$RES/PACKAGES_SHA256.txt" "$RES"/net_check.txt "$fs/" 2>/dev/null || true
    cp "$ST" "$fs/QUEUE_B3_STATUS.txt" 2>/dev/null || true; cp "$W/QUEUE6_STATUS" "$fs/QUEUE6_STATUS.txt" 2>/dev/null || true
    nvidia-smi > "$fs/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$fs/disk_end.txt" 2>&1 || true; echo "$verdict" > "$fs/verdict.txt"
    tar -czf "$out" -C "$fs" . 2>/dev/null && (cd "$RES" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256") && say "final package: $out"
    rm -rf "$fs"
  fi
  say "FINISHED $verdict. Packages: $RES/<KEY>/ (sha list: $RES/PACKAGES_SHA256.txt). The instance is NOT stopped."
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_B3_DONE written, rerun to resume"; FINISHED=1; exit 130' INT TERM
abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }

rm -f "$DONE"
say "queue_b3_4b started; models: $ORDER"
# ---- 0. upload files (cheap, before the long wait) ----------------------------------------------------------------------
for pair in "$ADDON:$ADDON_SHA" "$SIX_ADDON:$SIX_ADDON_SHA" "$SIX_QUEUE:$SIX_QUEUE_SHA"; do
  f=${pair%%:*}; s=${pair##*:}
  [ -s "$W/$f" ] || abort addon "$f missing in $W"
  echo "$s  $W/$f" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$f sha256 differs"
done
HAVE_IMAGES_TAR=0
if [ -s "$W/$IMAGES_TAR" ]; then
  echo "$IMAGES_SHA  $W/$IMAGES_TAR" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$IMAGES_TAR sha256 differs"
  HAVE_IMAGES_TAR=1
fi
export HAVE_IMAGES_TAR
say "upload files verified (sha256); image tar: $HAVE_IMAGES_TAR"
# ---- 0b. wait for the B2 queue and an idle GPU --------------------------------------------------------------------------
if [ "${NO_WAIT:-0}" = 1 ]; then say "NO_WAIT=1: not waiting for $B2DONE"
else
  say "waiting for $B2DONE, no other queue script, no GPU process and GPU memory < ${GPU_BUSY_MIB} MiB (poll ${POLL_S}s, max ${WAIT_MAX_H} h)"
  waited=0; okc=0; lastmsg=""
  while :; do
    why=""; [ -e "$B2DONE" ] || why="$B2DONE does not exist yet"
    if [ -z "$why" ]; then o=$(others_running); [ -n "$(echo "$o" | tr -d ' ')" ] && why="other queue script running:$o"; fi
    if [ -z "$why" ]; then u=$(gpu_used); a=$(gpu_apps); if [ -z "$u" ]; then why="nvidia-smi not available"; elif [ -n "$a" ] || [ "$u" -ge "$GPU_BUSY_MIB" ]; then why="GPU in use (${u} MiB)"; fi; fi
    if [ -z "$why" ]; then okc=$((okc + 1)); [ "$okc" -ge 2 ] && break; else okc=0; fi
    if [ "$why" != "$lastmsg" ] || [ $((waited % 1800)) -lt "$POLL_S" ]; then say "still waiting (${waited}s): ${why:-condition met once, checking again}"; lastmsg="$why"; fi
    [ "$waited" -ge $(( WAIT_MAX_H * 3600 )) ] && abort wait "$B2DONE / idle GPU not reached within ${WAIT_MAX_H} h"
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  say "wait over after ${waited}s: $B2DONE = '$(head -c 200 "$B2DONE" 2>/dev/null | tr '\n' ' ')'; GPU idle (${u:-?} MiB)"
fi
# ---- 0c. refusal checks (nothing else may run) --------------------------------------------------------------------------
o=$(others_running); [ -n "$(echo "$o" | tr -d ' ')" ] && abort others "another queue/overnight script is running ($o). Nothing was started."
apps=$(gpu_apps); used=$(gpu_used)
[ -z "$used" ] && abort gpu "nvidia-smi not available"
if [ -n "$apps" ] || [ "$used" -ge "$GPU_BUSY_MIB" ]; then abort gpu "the GPU is in use (${used} MiB; processes: $(echo "$apps" | head -3 | tr '\n' ';')). Nothing was started."; fi
say "GPU idle (${used} MiB, no compute process)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort gpu "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
net_speed() { local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'; }
hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/model-00001-of-00002.safetensors}")
say "network check (${NET_SECS}s): huggingface.co ${hf} MB/s; minimum $NET_MIN_MBPS MB/s"; slow=0
printf 'huggingface.co %s MB/s\n' "$hf" > "$RES/net_check.txt"
case "$hf" in ERR:4*) say "WARNING: test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;; *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
if [ "$slow" = 1 ] && [ "${FORCE_SLOW_NET:-0}" != 1 ]; then abort network "network too slow or unreachable. Nothing was downloaded. FORCE_SLOW_NET=1 overrides"; fi
say "checks ok: $(cat "$LOG/gpu.log"); $(free_gb) GB free; image tar: $HAVE_IMAGES_TAR"

# ---- per-model functions (run inside `step`; variables come from the exported environment) -----------------------------------------------------------------------
repo_env() { export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; }
setup() {
  cd "$R"
  if [ -d .git ]; then git cat-file -e "$BASE_COMMIT^{commit}"; git merge-base --is-ancestor "$BASE_COMMIT" HEAD; else echo "WARNING: no .git in $R; the code-hash check decides"; fi
  tar -tzf "$W/$ADDON" > "$RES/addon_listing.txt"
  if grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | grep -q .; then
    echo "addon contains paths outside its own files:"; grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | head; exit 1
  fi
  # copy only where nothing exists; an existing file must be byte-identical (e.g. the files the B2 queue / the InternVL2.5 REL6 queue unpacked); nothing is ever overwritten
  local stage="$RES/_addon_stage" f
  rm -rf "${stage:?}"; mkdir -p "$stage"; tar -xzf "$W/$ADDON" -C "$stage"
  while IFS= read -r f; do
    if [ -e "$R/$f" ]; then cmp -s "$stage/$f" "$R/$f" || { echo "addon file differs from the existing $R/$f; refusing to overwrite"; exit 1; }
    else mkdir -p "$(dirname "$R/$f")"; cp "$stage/$f" "$R/$f"; fi
  done < <(cd "$stage" && find . -type f | sed 's#^\./##' | sort)
  rm -rf "${stage:?}"
  ( cd followup_rel_6lang/b3_4b && sha256sum -c SHA256SUMS )
  if [ "$HAVE_IMAGES_TAR" = 1 ] && [ ! -d followup_rel_6lang/assets ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  repo_env; export PIP_CACHE_DIR="$R/.cache/pip" PIP_NO_CACHE_DIR=1
  source .venv/bin/activate
  case "$KEY" in
    qwen3_vl_4b)   CFGN=qwen3_vl_4b_4090_channels_last.yaml ;;
    internvl35_4b) CFGN=internvl3_5_4b_4090.yaml ;;
  esac
  B=followup_rel_6lang/b3_4b
  KEY="$KEY" CONFIG="$B/models/$CFGN" EXPA_REF="$B/contracts/ref_${KEY}_expA_20261114.json" REL_REF="$B/contracts/ref_${KEY}_rel_20261114.json" CODE_REF="$B/contracts/ref_${KEY}_rel_20261114.json" B2_PRE_MODEL=1 python "$B/env_check.py"
  python -c 'import mpr_crosslocale.data.rel_followup_6lang, mpr_crosslocale.inference.run_rel_followup_6lang; print("6-language modules import ok")'
  python followup_rel_4lang/data_bundle.py verify || { python followup_rel_4lang/data_bundle.py unpack; retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2; python followup_rel_4lang/data_bundle.py verify; }
  if ! python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang > "$LOG/bundle6_verify_$KEY.log" 2>&1; then
    echo "6-language verify failed (see bundle6_verify_$KEY.log)"
    if [ -d "$W/mpr-ivl/followup_rel_6lang/assets" ] && [ "$R" != "$W/mpr-ivl" ]; then echo "copying the fr/ja images from $W/mpr-ivl (same server)"; mkdir -p followup_rel_6lang/assets; cp -R -n "$W/mpr-ivl/followup_rel_6lang/assets/." followup_rel_6lang/assets/; fi
    python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang > "$LOG/bundle6_verify2_$KEY.log" 2>&1 \
      || { echo "fetching the missing fr/ja screenshots from the pinned HF revision"; retry 6 python followup_rel_6lang/data_bundle_6lang.py fetch-images --data-root followup_rel_6lang --workers 2; python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang; }
  fi
}
model() {
  cd "$R"; repo_env; source .venv/bin/activate
  retry 3 hf download "$MID" --revision "$REV"
  MID="$MID" REV="$REV" RES="$RES" python - <<'PY'
import glob, json, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MID"], revision=os.environ["REV"], local_files_only=True)
fs = glob.glob(os.path.join(snap, "*.safetensors")); size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) >= int(os.environ["MIN_SHARDS"]) and int(os.environ["BYTES_MIN"]) <= size <= int(os.environ["BYTES_MAX"]), (len(fs), size)
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg.get("architectures"); llm = (cfg.get("llm_config") or {}).get("architectures")
print("architectures", arch, "| llm", llm, "| template", cfg.get("template"))
if os.environ["ARCH_CHECK"] == "qwen3vl": assert arch == ["Qwen3VLForConditionalGeneration"], arch
open(os.path.join(os.environ["RES"], "model_size_" + os.environ["KEY"] + ".txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\narchitectures={arch} llm={llm}\n")
if os.environ["TOKLOADER"] == "internvl":
    from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer
    tok = load_internvl_tokenizer(os.environ["MID"], revision=os.environ["REV"], trust_remote_code=True)
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
want = [int(x) for x in os.environ["LABELS"].split(",")]
assert ids == {l: [w] for l, w in zip("ABCD", want)}, ids       # ids recorded in the earlier runs' label_token_ids
PY
  du -sh "$HF_HOME"
}
smoke_rel()  { cd "$R"; repo_env; bash followup_rel_6lang/b3_4b/run_rel6_b3.sh "$KEY" smoke; }
full_rel()   { cd "$R"; repo_env; REL_LIMIT_S="$T_REL" bash followup_rel_6lang/b3_4b/run_rel6_b3.sh "$KEY" full; }
smoke_expa() { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" bash followup_rel_6lang/b3_4b/run_expA6_b3.sh "$KEY" smoke; }
full_expa()  { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" EXPA_LIMIT_S="$T_EXPA" bash followup_rel_6lang/b3_4b/run_expA6_b3.sh "$KEY" full; }
retry_expa() { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" bash followup_rel_6lang/b3_4b/run_expA6_b3.sh "$KEY" retry; }
six_setup() {  # part c, per model: read-only guards, then the setup function of the unchanged queue_6x6.sh (it only reuses what is right; ours stops before it could modify the repo)
  cd "$W"
  tar -xzf "$W/$SIX_ADDON" -C "$W"; ( cd "$W/sixbysix_addon" && sha256sum -c SHA256SUMS )
  cd "$R"
  if [ -d .git ]; then
    if [ "$FAM" = qwen ]; then
      [ "$(git rev-parse HEAD)" = "$BASE_COMMIT" ] || { echo "6x6: HEAD of $R is $(git rev-parse --short HEAD), not ${BASE_COMMIT:0:7}; queue_6x6.sh would run git checkout -f. Not touching the repo."; exit 1; }
    else
      git merge-base --is-ancestor "$BASE_COMMIT" HEAD || { echo "6x6: HEAD of $R is not a descendant of ${BASE_COMMIT:0:7}"; exit 1; }
      [ "$(git rev-list --count "$BASE_COMMIT"..HEAD)" = 5 ] || { echo "6x6: $R is $(git rev-list --count "$BASE_COMMIT"..HEAD) commits after ${BASE_COMMIT:0:7}, expected the 5 InternVL patches. Not touching the repo."; exit 1; }
    fi
  elif [ "${Q6_TEST:-0}" != 1 ]; then echo "6x6: $R has no .git"; exit 1; fi
  if [ "${Q6_TEST:-0}" != 1 ]; then   # the venv must already be right: queue_6x6.sh would run pip install otherwise
    EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}" REPO_DIR="$PWD" FAMX="$FAM" .venv/bin/python - <<'PY'
import os
import torch, transformers, PIL, yaml, accelerate, mpr_crosslocale
assert torch.__version__ == os.environ["EXPECT_TORCH"], torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"])), mpr_crosslocale.__file__
if os.environ["FAMX"] == "ivl":
    import timm, einops
else:
    import qwen_vl_utils
print("6x6 venv imports ok (queue_6x6.sh will not run pip)")
PY
  fi
  local rc=0 V="$RES/tools/followup_rel_6lang/b3_4b/verify_6x6_data.py"   # taken from the addon tar, so that this step does not depend on part a/b having unpacked it into the repo
  mkdir -p "$RES/tools"; tar -xzf "$W/$ADDON" -C "$RES/tools" followup_rel_6lang/b3_4b/verify_6x6_data.py
  python3 -I "$V" --repo "$R" --allow-no-manifest || rc=$?
  if [ $rc -eq 0 ]; then echo "6x6 raw data present and verified in $R: reused"
  elif [ $rc -eq 10 ]; then
    # not here: another repo of this server may hold it (queue_6x6.sh links it), else it downloads 2.6 GB
    local other found=""
    for other in "$W/mpr-gui-crosslocale" "$W/mpr-ivl"; do
      [ "$other" = "$R" ] && continue
      if python3 -I "$V" --repo "$other" --allow-no-manifest > "$LOG/6x6_data_$(basename "$other").log" 2>&1; then found="$other"; break; fi
    done
    if [ -n "$found" ]; then echo "6x6 raw data verified in $found: queue_6x6.sh links it"
    else [ "$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)" -ge 8 ] || { echo "6x6: no data on this server and less than 8 GB free for the 2.6 GB download"; exit 1; }
      echo "6x6: no raw data on this server; queue_6x6.sh downloads it (2.6 GB, rate-limit retries)"; fi
  elif [ $rc -ne 1 ]; then echo "6x6: data check could not run (exit $rc)"; exit 1
  elif [ "${SKIP_6X6_DATA_CHECK:-0}" = 1 ]; then echo "6x6 raw data differs from the pinned revision but SKIP_6X6_DATA_CHECK=1: continuing"
  else echo "6x6 raw data in $R differs from the pinned revision (see above). Not continuing; SKIP_6X6_DATA_CHECK=1 overrides."; exit 1; fi
  W="$W" bash "$W/$SIX_QUEUE" --fn fn_setup "$FAM"
  python3 -I "$V" --repo "$R" || [ "${SKIP_6X6_DATA_CHECK:-0}" = 1 ]
}
six_run() {  # part c, per model: the unchanged queue_6x6.sh for this one key
  cd "$W"
  local rc=0
  W="$W" MODELS="$KEY" IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 bash "$W/$SIX_QUEUE" || rc=$?   # 2026-11-17 review fix: outer queue already checked the network
  for f in "$W"/queue_logs6/${KEY}_*.log; do [ -f "$f" ] && { echo "== tail of $f"; tail -n 60 "$f"; }; done   # so that the CUDA-text check of step() sees the model logs
  [ -f "$W/queue_results/6x6_$KEY/COMPLETE" ]
}
export -f retry setup model repo_env smoke_rel full_rel smoke_expa full_expa retry_expa six_setup six_run
export W RES LOG BASE_COMMIT ADDON IMAGES_TAR SIX_ADDON SIX_QUEUE

smoke_report() {  # $1 gate json, $2 label
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
try: g = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
l = f"{sys.argv[2]}: acc {s.get('accuracy')} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')} prompt tokens {s.get('prompt_token_count')}"
if g.get("warnings"): l += " WARN " + "; ".join(g["warnings"])
if g.get("problems"): l += " PROBLEMS " + "; ".join(g["problems"])
print(l, end=" | ")
PY
}
env_recheck() {  # after a failed model: the GPU must be idle (our processes gone), a 4090, and disk must be left
  local i u a
  for i in $(seq 1 "${GPU_RECHECK_TRIES:-12}"); do u=$(gpu_used); a=$(gpu_apps); [ -n "$u" ] && [ -z "$a" ] && [ "$u" -lt "$GPU_BUSY_MIB" ] && break; sleep "${GPU_RECHECK_S:-10}"; done
  [ -n "$u" ] || { ENV_REASON="nvidia-smi lost after the failure"; return 1; }
  if [ -n "$a" ] || [ "$u" -ge "$GPU_BUSY_MIB" ]; then ENV_REASON="GPU not released after the failure (${u} MiB used)"; return 1; fi
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q 4090 || { ENV_REASON="GPU is no longer an RTX 4090"; return 1; }
  [ "$(free_gb)" -ge "$MIN_DISK_ALWAYS_GB" ] || { ENV_REASON="only $(free_gb) GB free"; return 1; }
  return 0
}
free_prior_caches() {  # make room for the next download: remove the kept cache(s) of EARLIER models of this queue (kept for their 6x6, or after a failure); a later 6x6 of such a model downloads it again
  local k
  for k in $ORDER; do
    [ "$k" = "$KEY" ] && break
    ( minfo "$k"; if [ -d "$CACHE_DIR" ]; then rm -rf "${CACHE_DIR:?}"; echo "$(date -u +%FT%TZ) b3_4b: removed the kept cache of $k to make room (free now $(free_gb) GB); its later step downloads it again" | tee -a "$ST" "$RES/STATUS"; fi )
  done
}
gate_ok() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }

run_model() {  # parts a and b of one model
  minfo "$1"; step_failed=0; ENV_FAIL=0; ENV_REASON=""
  export KEY MID REV R W RES LOG BYTES_MIN BYTES_MAX MIN_SHARDS LABELS TOKLOADER ARCH_CHECK SCOPE T_REL T_EXPA EXPA_BATCH_V BASE_COMMIT ADDON FAM
  if [ -f "$RES/$KEY/COMPLETE" ]; then say "$KEY parts a+b already complete (package exists), skipped"; setm "$KEY" "ab_complete_earlier"; return 0; fi
  if ! { [ -d "$R/01_code" ] && [ -f "$R/.venv/bin/activate" ]; }; then say "STOP: $R (repo with 01_code and .venv) does not exist"; FAILED+=("repo_$KEY"); setm "$KEY" "STOPPED_repo"; ENV_FAIL=1; ENV_REASON="repo missing"; return 1; fi
  say "=== $KEY: $MID @ $REV (repo $REPO_NAME; REL new_only $N_REL, Exp A $N_EXPA + T_retry $N_RETRY, batch 1, strict true; estimate $EST_REL s/inference REL, $EST_EXPA Exp A) ==="
  local rel_state=pending expa_state=pending
  if gate_ok "$OUT_REL/full/gate_rel_complete.json"; then rel_state=done_earlier; fi
  if gate_ok "$OUT_EXA/full/gate_complete.json" && gate_ok "$OUT_EXA/retry/gate_complete.json"; then expa_state=done_earlier; fi
  if [ "$rel_state" = pending ] || [ "$expa_state" = pending ]; then
    if ! step setup "$T_SETUP" setup; then setm "$KEY" "FAILED_setup"; return 1; fi
    local f; f=$(free_gb)
    if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt $(( MINFREE + KEEP_MARGIN_GB )) ]; then free_prior_caches; f=$(free_gb); fi
    if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then say "STOP: only ${f} GB free (need $MINFREE GB for $KEY)"; FAILED+=("disk_$KEY"); setm "$KEY" "STOPPED_disk"; ENV_FAIL=1; ENV_REASON="disk"; return 1; fi
    if ! step model "$T_MODEL" model; then setm "$KEY" "FAILED_model"; return 1; fi
    local rel_ok=0 expa_ok=0
    if [ "$rel_state" = pending ]; then
      if step smoke_rel "$T_SMOKE" smoke_rel; then rel_ok=1; say "smoke gate: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "$KEY rel fr/ja")"
      else say "$KEY REL smoke failed: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "$KEY rel")see $LOG/smoke_rel_$KEY.log; REL full skipped"; rel_state=failed; fi
    else say "$KEY REL fr/ja already complete (gate passed earlier); REL steps skipped"; fi
    if [ "$expa_state" = pending ]; then
      if step smoke_expa "$T_SMOKE" smoke_expa; then expa_ok=1; say "smoke gate: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "$KEY expA $SCOPE")"
      else say "$KEY Exp A smoke failed: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "$KEY expA")see $LOG/smoke_expa_$KEY.log; Exp A full skipped"; expa_state=failed; fi
    else say "$KEY Exp A fr/ja already complete (gates passed earlier); Exp A steps skipped"; fi
    [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
    if [ "$rel_ok" = 1 ]; then
      if step full_rel "$T_REL" full_rel; then rel_state=done; else rel_state=failed; fi
    fi
    if [ "$rel_state" = done ] || [ "$rel_state" = failed ] || [ "$rel_state" = done_earlier ]; then
      local s=""; [ "$rel_state" = failed ] && s="_PARTIAL"
      package "$KEY" 01_rel "$s" "rel6new_$KEY" || true
    fi
    [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
    if [ "$expa_ok" = 1 ]; then
      if step full_expa "$T_EXPA" full_expa && step retry_expa "$T_RETRY" retry_expa; then expa_state=done; else expa_state=failed; fi
    fi
    if [ "$expa_state" = done ] || [ "$expa_state" = failed ] || [ "$expa_state" = done_earlier ]; then
      local s=""; [ "$expa_state" = failed ] && s="_PARTIAL"
      package "$KEY" 02_expA "$s" "expA6_$KEY" || true
    fi
    [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
  else
    say "$KEY REL and Exp A gates already passed on disk; packaging only"
    package "$KEY" 01_rel "" "rel6new_$KEY" || true; package "$KEY" 02_expA "" "expA6_$KEY" || true
  fi
  case "$rel_state:$expa_state" in
    done:done|done:done_earlier|done_earlier:done|done_earlier:done_earlier) touch "$RES/$KEY/COMPLETE"; setm "$KEY" "ab_ok"; return 0 ;;
    *) setm "$KEY" "ab_FAILED(rel=$rel_state,expA=$expa_state)"; return 1 ;;
  esac
}

run_six() {  # part c for one model
  minfo "$1"; step_failed=0; ENV_FAIL=0; ENV_REASON=""
  export KEY MID REV R W RES LOG BASE_COMMIT FAM SIX_ADDON SIX_QUEUE REPO_NAME
  if [ -f "$W/queue_results/6x6_$KEY/COMPLETE" ]; then say "$KEY 6x6 already complete, skipped"; setm "$KEY" "$(getm "$KEY") 6x6_complete_earlier"; return 0; fi
  if ! { [ -d "$R/01_code" ] && [ -f "$R/.venv/bin/activate" ]; }; then say "STOP: $R (repo with 01_code and .venv) does not exist; no 6x6 for $KEY"; FAILED+=("repo6x6_$KEY"); setm "$KEY" "$(getm "$KEY") 6x6_STOPPED_repo"; ENV_FAIL=1; ENV_REASON="repo missing"; return 1; fi
  say "=== $KEY 6x6: 79,020 unique inferences (queue_6x6.sh MODELS=$KEY; estimate $EST_REL s/inference assumed = REL median) ==="
  if ! step six_setup "$T_SIXSETUP" six_setup; then setm "$KEY" "$(getm "$KEY") 6x6_FAILED_setup"; return 1; fi
  # a failed six_setup is model-specific here (repo state/data differ); it never installed or checked out anything
  step 6x6 "$S6_LIMIT" six_run
  local rc=$?
  say "6x6 $KEY: $(grep -h "6x6: $KEY finished\|6x6: $KEY full 6x6\|6x6: $KEY smoke\|6x6: $KEY SKIPPED\|6x6: $KEY DOWNLOAD\|6x6: $KEY SETUP" "$W/QUEUE6_STATUS" 2>/dev/null | tail -4 | cut -c1-300 | tr '\n' '|')"
  # package: link/copy the tar of queue_6x6.sh and write the per-file sha list of the result folder
  local d6="$W/queue_results/6x6_$KEY" dest="$RES/$KEY" sfx=""; mkdir -p "$dest"
  [ $rc -eq 0 ] || sfx="_PARTIAL"
  local f; for f in "$d6"/*.tar.gz; do [ -e "$f" ] || continue; ln -f "$f" "$dest/" 2>/dev/null || cp "$f" "$dest/"; done
  [ -e "$d6/SHA256SUMS" ] && cp "$d6/SHA256SUMS" "$dest/6x6_SHA256SUMS"
  [ -d "$OUT_6X6" ] && (cd "$(dirname "$OUT_6X6")" && find "$(basename "$OUT_6X6")" -type f -exec sha256sum {} + > "$dest/b3_${KEY}_03_6x6_${DAY}${sfx}.files.sha256") 2>/dev/null
  (cd "$dest" && sha256sum ./*6x6*.tar.gz 2>/dev/null >> "$RES/PACKAGES_SHA256.txt")
  say "$KEY 6x6 package(s) in $dest ($(ls "$dest" | grep -c 6x6) files); per-file sha list b3_${KEY}_03_6x6_${DAY}${sfx}.files.sha256"
  if [ $rc -eq 0 ]; then setm "$KEY" "$(getm "$KEY") 6x6_ok"; else setm "$KEY" "$(getm "$KEY") 6x6_NOT_COMPLETE"; fi
  return $rc
}

after_failure_check() {  # $1 key; returns 1 (and stops the loop) on an environmental failure
  if [ "$ENV_FAIL" = 1 ]; then say "STOP QUEUE (environmental failure: ${ENV_REASON:-see step log}); remaining steps are not started"; return 1; fi
  if ! env_recheck; then say "STOP QUEUE (environment re-check after the failed $1: $ENV_REASON); remaining steps are not started"; FAILED+=("env_after_$1"); return 1; fi
  say "$1 failed with a model-specific problem; environment re-check ok; continuing"; return 0
}

STOPPED=0
if [ "${ONLY_6X6:-0}" != 1 ]; then
  for K in $ORDER; do
    run_model "$K"; rc=$?
    if [ $rc -eq 0 ]; then
      minfo "$K"
      [ "${SKIP_6X6:-0}" = 1 ] && [ "${KEEP_MODEL_CACHE:-0}" != 1 ] && [ -d "$CACHE_DIR" ] && { rm -rf "${CACHE_DIR:?}"; say "$K cache deleted (parts a+b finished, SKIP_6X6=1); free disk: $(free_gb) GB"; }
    else
      minfo "$K"; [ -d "$CACHE_DIR" ] && say "$K cache KEPT (not finished; free $(free_gb) GB); it is removed automatically if the next download needs the room"
      if ! after_failure_check "$K"; then STOPPED=1; break; fi
    fi
  done
fi
if [ "$STOPPED" = 0 ] && [ "${SKIP_6X6:-0}" != 1 ]; then
  for K in $ORDER; do
    run_six "$K"; rc=$?
    if [ $rc -ne 0 ]; then
      if ! after_failure_check "6x6_$K"; then STOPPED=1; break; fi
    fi
  done
fi
if [ "$STOPPED" = 1 ]; then for K in $ORDER; do [ -z "$(getm "$K")" ] && NOTRUN+=("$K"); done; fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
