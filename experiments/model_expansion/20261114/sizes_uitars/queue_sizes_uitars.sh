#!/usr/bin/env bash
# GPU queue on the Vast.ai RTX 4090 (after the Qwen3/InternVL3.5 Exp A overnight job):
#   Qwen3-VL-2B, Qwen3-VL-4B, InternVL3.5-1B, -2B, -4B, UI-TARS-1.5-7B; batch size 1 for all.
# Order: early checks (addon, sha256, timeout, GPU name) -> wait for FINISHED in /workspace/OVERNIGHT_STATUS (bounded) -> GPU idle
#   -> network check -> repo checks (per family) -> unpack addon -> batch-1 check of Qwen3-VL-8B (only if overnight Qwen3 was ok)
#   -> delete the 8B caches (only if overnight reports qwen3=ok internvl35=ok) -> models.
# Per model: download pinned revision -> smoke + gate -> REL O/R/C/F full (18,432) -> Exp A O/T/R full (5,688)
#            -> package to /workspace/queue_results/<key>/ -> cache rule (below) -> next model (also after a failure).
# Cache rule: a model that finished everything loses its cache. After a failure or timeout the cache is KEPT (a rerun resumes without
#   downloading; hf download resumes partial files) as long as the free disk still covers the largest MIN_FREE_GB of the models that come
#   next (the last model: HANDOFF_FREE_GB, default 40, for the MiniCPM queue); otherwise it is deleted. KEEP_FAILED_CACHE=0 = always delete.
# Usage (inside tmux, in /workspace, with the addon tar.gz next to this script):
#   bash queue_sizes_uitars.sh            # waits for FINISHED in /workspace/OVERNIGHT_STATUS (max OVERNIGHT_MAX_H, default 14 h)
#   SKIP_WAIT=1 bash queue_sizes_uitars.sh   # only if the earlier job is NOT running (the GPU idle check still applies)
#   ONLY="qwen3_vl_2b internvl35_1b" bash queue_sizes_uitars.sh   # subset, same order
#   FORCE_SLOW_NET=1                      # start although the network is below 20 MB/s
# Progress: cat /workspace/QUEUE_STATUS   Logs: /workspace/queue_logs/
# /workspace/QUEUE_DONE is written when the script ends for any reason except Ctrl-C/kill: content "OK" (queue ran through; look at the
#   model lines for what completed) or "STOPPED rc=<n>" (a STOP). The next queue (MiniCPM) starts on either.
# Rerunning resumes (finished stages are skipped). The instance is NOT stopped or destroyed.
set -uo pipefail
W="${WORKSPACE:-/workspace}"   # only changed for local tests
LOG="$W/queue_logs"; RES="$W/queue_results"; ST="$W/QUEUE_STATUS"
mkdir -p "$LOG" "$RES"
ADDON="$W/model_expansion_sizes_uitars_20261114.tar.gz"
ADDON_SHA=79367d4108b2ba4b4d8049286158753c68e14a65291936feaacedf85497ffa7b
QWEN="$W/mpr-gui-crosslocale"; IVL="$W/mpr-ivl"
ORDER="qwen3_vl_2b qwen3_vl_4b internvl35_1b internvl35_2b internvl35_4b uitars_15_7b"
ORDER="${ONLY:-$ORDER}"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"
HANDOFF_FREE_GB="${HANDOFF_FREE_GB:-40}"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc 0-9; }
retry() {  # retry <tries> <cmd...>: network steps on flaky hosts (SSL EOF, gnutls handshake seen on the previous server)
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "retry $i/$n failed: $*" >&2
    [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i ))
  done
  return 1
}
dur_s() {  # "90m", "4h" -> seconds (GNU timeout takes ONE unit: "4h30m" is invalid, use "270m")
  local d=$1 s=0
  [[ $d =~ ([0-9]+)h ]] && s=$((s + ${BASH_REMATCH[1]} * 3600))
  [[ $d =~ ([0-9]+)m ]] && s=$((s + ${BASH_REMATCH[1]} * 60))
  echo "$s"
}

# key -> REPO MODEL_ID REVISION MIN_FREE_GB T_SMOKE T_REL T_EXPA (GNU timeout durations)
# MIN_FREE_GB = model files x about 1.3 + 8 GB for results and pip/HF temp files.
model_info() {
  case "$1" in
    qwen3_vl_2b)   REPO=$QWEN; MID=Qwen/Qwen3-VL-2B-Instruct;       REV=89644892e4d85e24eaac8bacfd4f463576704203; MINFREE=14; T_SMOKE=45m; T_REL=4h; T_EXPA=90m ;;
    qwen3_vl_4b)   REPO=$QWEN; MID=Qwen/Qwen3-VL-4B-Instruct;       REV=ebb281ec70b05090aa6165b016eac8ec08e71b17; MINFREE=20; T_SMOKE=45m; T_REL=5h; T_EXPA=2h ;;
    internvl35_1b) REPO=$IVL;  MID=OpenGVLab/InternVL3_5-1B;        REV=2f71cf52542334823e48a46ffba0e2bc9add3446; MINFREE=12; T_SMOKE=45m; T_REL=4h; T_EXPA=90m ;;
    internvl35_2b) REPO=$IVL;  MID=OpenGVLab/InternVL3_5-2B;        REV=7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0; MINFREE=14; T_SMOKE=45m; T_REL=270m; T_EXPA=105m ;;
    internvl35_4b) REPO=$IVL;  MID=OpenGVLab/InternVL3_5-4B;        REV=481f6e32467eab4e922ccd7fd6cf420441a62331; MINFREE=20; T_SMOKE=45m; T_REL=330m; T_EXPA=2h ;;
    uitars_15_7b)  REPO=$QWEN; MID=ByteDance-Seed/UI-TARS-1.5-7B;   REV=683d002dd99d8f95104d31e70391a39348857f4e; MINFREE=52; T_SMOKE=1h; T_REL=8h; T_EXPA=3h ;;
    *) return 1 ;;
  esac
  CACHE_DIR="$REPO/.cache/huggingface/hub/models--${MID//\//--}"
}

repo_env() {  # $1 repo; exports the same cache layout as run_all_overnight.sh
  export XDG_CACHE_HOME="$1/.cache" HF_HOME="$1/.cache/huggingface" TORCH_HOME="$1/.cache/torch" PIP_CACHE_DIR="$1/.cache/pip"
  export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
}

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }

# ----------------------------------------------------------------------------------------------------------------
# Lock, QUEUE_DONE trap
# ----------------------------------------------------------------------------------------------------------------
command -v flock >/dev/null && { exec 9> "$W/.queue_sizes.lock"; flock -n 9 || { echo "another queue_sizes_uitars.sh is already running"; exit 1; }; }
rm -f "$W/QUEUE_DONE"
QUEUE_OK=0; NO_DONE=0
on_exit() {
  local rc=$?
  [ "$NO_DONE" = 1 ] && return 0
  if [ "$QUEUE_OK" = 1 ]; then echo "OK" > "$W/QUEUE_DONE"; else echo "STOPPED rc=$rc" > "$W/QUEUE_DONE"; say "QUEUE_DONE written: STOPPED rc=$rc"; fi
}
trap on_exit EXIT
trap 'say "interrupted by signal; no QUEUE_DONE written. Rerun the same command to resume."; NO_DONE=1; exit 130' INT TERM

# ----------------------------------------------------------------------------------------------------------------
# 1. Early checks: only what does not depend on the overnight job
# ----------------------------------------------------------------------------------------------------------------
early_checks() {
  [ -s "$ADDON" ] || { say "STOP: $ADDON is missing"; exit 1; }
  echo "$ADDON_SHA  $ADDON" | sha256sum -c - > "$LOG/addon_sha.log" 2>&1 || { say "STOP: addon sha256 differs"; exit 1; }
  command -v timeout >/dev/null || { say "STOP: GNU timeout missing"; exit 1; }
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 4090 || { say "STOP: GPU is not an RTX 4090"; exit 1; }
}

# ----------------------------------------------------------------------------------------------------------------
# 2. Wait for the overnight job (bounded), then an idle GPU
# ----------------------------------------------------------------------------------------------------------------
overnight_running() { pgrep -f 'run_all_overnight\.sh' > /dev/null 2>&1; }

wait_for_overnight() {
  if [ "${SKIP_WAIT:-0}" = 1 ]; then say "SKIP_WAIT=1: not waiting for OVERNIGHT_STATUS"; return; fi
  local poll=${POLL_S:-60} max=${OVERNIGHT_MAX_S:-$(( ${OVERNIGHT_MAX_H:-14} * 3600 ))} grace=${OVERNIGHT_GRACE_S:-$(( ${OVERNIGHT_GRACE_MIN:-10} * 60 ))} waited=0 absent=0
  say "waiting for FINISHED in $W/OVERNIGHT_STATUS (max ${OVERNIGHT_MAX_H:-14} h; STOP if run_all_overnight.sh is gone for ${OVERNIGHT_GRACE_MIN:-10} min without FINISHED)"
  until grep -q FINISHED "$W/OVERNIGHT_STATUS" 2>/dev/null; do
    if overnight_running; then absent=0; else absent=$((absent + poll)); fi
    if [ "$absent" -ge "$grace" ]; then
      say "STOP: no run_all_overnight.sh process for $((absent / 60)) min and no FINISHED line in OVERNIGHT_STATUS (last line: $(tail -1 "$W/OVERNIGHT_STATUS" 2>/dev/null || echo none)). The overnight job probably stopped itself. Check it; rerun with SKIP_WAIT=1 only if the GPU is free."
      exit 1
    fi
    if [ "$waited" -ge "$max" ]; then say "STOP: no FINISHED line within ${OVERNIGHT_MAX_H:-14} h"; exit 1; fi
    sleep "$poll"; waited=$((waited + poll))
    [ $((waited % 1800)) -lt "$poll" ] && say "still waiting for overnight FINISHED (${waited}s)"
  done
  say "overnight job reports: $(grep FINISHED "$W/OVERNIGHT_STATUS" | tail -1)"
}

wait_gpu_idle() {  # memory.used < 1500 MiB, also with SKIP_WAIT=1; up to 30 min
  local n=0 used
  while :; do
    used=$(gpu_used_mib)
    [ -z "$used" ] && { say "STOP: nvidia-smi gives no memory.used"; exit 1; }
    [ "$used" -lt 1500 ] && return 0
    [ "$n" -ge "${GPU_WAIT_MIN:-30}" ] && { say "STOP: GPU still busy (${used} MiB used) after ${GPU_WAIT_MIN:-30} min. Check nvidia-smi"; exit 1; }
    sleep "${GPU_POLL_S:-60}"; n=$((n + 1))
  done
}

# ----------------------------------------------------------------------------------------------------------------
# 3. Network check (same method as queue_6x6.sh): 15 s ranged download
# ----------------------------------------------------------------------------------------------------------------
net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}

net_check() {  # download.pytorch.org is measured and logged; it only matters (STOP) when torch must be downloaded, which this queue does not do
  local hf pt slow=0
  hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/main/model-00001-of-00004.safetensors}")
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${hf} MB/s, download.pytorch.org ${pt} MB/s; minimum $NET_MIN_MBPS MB/s (huggingface.co decides)"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$hf" "$pt" > "$RES/net_check_sizes.txt"
  case "$hf" in
    ERR:4*) say "WARNING: the test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;;
    ERR*) slow=1 ;;
    *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
  esac
  if [ "$slow" = 1 ]; then
    if [ "${FORCE_SLOW_NET:-0}" = 1 ]; then say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; fi
    say "STOP: network too slow or unreachable (need >= $NET_MIN_MBPS MB/s from huggingface.co). Nothing was downloaded. Use another instance, or FORCE_SLOW_NET=1."
    exit 1
  fi
}

# ----------------------------------------------------------------------------------------------------------------
# 4. Repo checks after the overnight wait; a family whose repo is missing is skipped, the other still runs
# ----------------------------------------------------------------------------------------------------------------
QWEN_OK=0; IVL_OK=0
repo_ready() {  # $1 repo
  [ -f "$1/.venv/bin/activate" ] && [ -f "$1/followup_rel_4lang/run_inference.py" ] \
    && [ -f "$1/followup_rel_4lang/expA/expA_inputs/EXPA_SHA256SUMS" ]
}
repo_checks() {
  if repo_ready "$QWEN"; then QWEN_OK=1; else say "WARNING: $QWEN is not set up as in run_all_overnight.sh (venv, run_inference.py, expA_inputs); Qwen3-VL and UI-TARS models are skipped"; fi
  if repo_ready "$IVL"; then IVL_OK=1; else say "WARNING: $IVL is not set up as in run_all_overnight.sh; InternVL3.5 models are skipped"; fi
  if [ "$QWEN_OK" = 0 ] && [ "$IVL_OK" = 0 ]; then say "STOP: neither repo is set up"; exit 1; fi
}

unpack_addon() {
  local r
  for r in "$QWEN" "$IVL"; do
    [ "$r" = "$QWEN" ] && [ "$QWEN_OK" = 0 ] && continue
    [ "$r" = "$IVL" ] && [ "$IVL_OK" = 0 ] && continue
    if ( cd "$r" && tar -xzf "$ADDON" && sha256sum -c followup_rel_4lang/model_expansion/SHA256SUMS ) >> "$LOG/addon_unpack.log" 2>&1; then
      say "addon unpacked and verified in $r"
    else
      say "WARNING: addon unpack/verify failed in $r (see $LOG/addon_unpack.log); that family is skipped"
      [ "$r" = "$QWEN" ] && QWEN_OK=0; [ "$r" = "$IVL" ] && IVL_OK=0
    fi
  done
  if [ "$QWEN_OK" = 0 ] && [ "$IVL_OK" = 0 ]; then say "STOP: addon could not be unpacked in any repo"; exit 1; fi
}

# ----------------------------------------------------------------------------------------------------------------
# 5. Batch-size check of the Qwen3-VL-8B (before its cache is deleted); a failure here never stops the queue
# ----------------------------------------------------------------------------------------------------------------
OVN_QWEN_OK=0; OVN_ALL_OK=0
read_overnight_result() {
  local l; l=$(grep FINISHED "$W/OVERNIGHT_STATUS" 2>/dev/null | tail -1)
  case "$l" in *"qwen3=ok"*) OVN_QWEN_OK=1 ;; esac
  case "$l" in *"FINISHED qwen3=ok internvl35=ok"*) OVN_ALL_OK=1 ;; esac
}

batch1_check() {
  local out="$RES/batch1_check_qwen3_8b" c8="$QWEN/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct" rc
  if [ "$OVN_QWEN_OK" != 1 ]; then say "batch-1 check skipped: overnight did not report qwen3=ok"; return 0; fi
  if [ "$QWEN_OK" != 1 ]; then say "batch-1 check skipped: Qwen repo not usable"; return 0; fi
  if [ ! -d "$c8" ]; then say "batch-1 check skipped: Qwen3-VL-8B cache not found"; return 0; fi
  if [ -s "$out/batch1_vs_batch4.json" ]; then say "batch-1 check already done: $(tail -1 "$out/STATUS" 2>/dev/null)"; return 0; fi
  mkdir -p "$out"
  say "batch-1 check: Qwen3-VL-8B at batch size 1 on ${BATCH1_ITEMS:-28} items (about 1,000 inferences), limit ${T_BATCH1:-1h}"
  ( repo_env "$QWEN"; export BATCH1_OUT="$out"; cd "$QWEN" && timeout -k 60 "${T_BATCH1:-1h}" bash followup_rel_4lang/model_expansion/batch1_check_qwen3_8b.sh ) >> "$LOG/batch1_check.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then say "batch-1 check done: $(grep '^BATCH1' "$LOG/batch1_check.log" | tail -1) (details $out/batch1_vs_batch4.json)"
  else say "batch-1 check FAILED (exit $rc; see $LOG/batch1_check.log). The queue continues."; fi
  return 0
}

delete_8b_caches() {
  # Only these two model folders (HF hub cache), and only if the overnight job reports both models ok. Results and venvs stay.
  if [ "$OVN_ALL_OK" != 1 ]; then
    say "8B caches kept: overnight did not report 'FINISHED qwen3=ok internvl35=ok' (the per-model free-disk checks decide what still fits). free disk: $(free_gb) GB"
    return 0
  fi
  for d in "$QWEN/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct" \
           "$IVL/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B"; do
    if [ -d "$d" ]; then rm -rf "$d"; say "deleted $d"; else say "not found (already gone): $d"; fi
  done
  say "free disk: $(free_gb) GB"
}

# ----------------------------------------------------------------------------------------------------------------
# 6. Models
# ----------------------------------------------------------------------------------------------------------------
download() {  # model_info already called. Partial files stay in the cache; hf resumes them.
  ( repo_env "$REPO"; cd "$REPO" && source .venv/bin/activate
    retry 3 timeout "${T_DOWNLOAD:-2h}" hf download "$MID" --revision "$REV" ) > "$LOG/${KEY}_download.log" 2>&1 || return 1
  [ -s "$CACHE_DIR/snapshots/$REV/config.json" ] || return 1
  du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 > "$LOG/${KEY}_size_on_disk.txt"
}

stage() {  # $1 stage name, $2 timeout
  ( repo_env "$REPO"; export REL_LIMIT_S EXPA_LIMIT_S   # FORCE_*_LIMIT_S: local test hooks only
    REL_LIMIT_S=${FORCE_REL_LIMIT_S:-$(dur_s "$T_REL")}; EXPA_LIMIT_S=${FORCE_EXPA_LIMIT_S:-$(dur_s "$T_EXPA")}
    cd "$REPO" && timeout -k 60 "$2" bash followup_rel_4lang/model_expansion/run_model.sh "$KEY" "$1" ) >> "$LOG/${KEY}_$1.log" 2>&1
}

gate_report() {  # one line per smoke gate: accuracy, speed, ETA, warnings, problems
  python3 - "$REPO/followup_rel_4lang/results/sizes_$KEY" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for sub, name in (("rel_smoke", "rel"), ("expa_smoke", "expa")):
    p = base / sub / f"gate_{name}_smoke.json"
    if not p.exists():
        continue
    g = json.loads(p.read_text())
    t, s = g.get("timing", {}), g.get("sanity", {})
    line = f"{name}: acc {s.get('accuracy')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h"
    if g.get("warnings"):
        line += " WARN " + "; ".join(g["warnings"])
    if g.get("problems"):
        line += " PROBLEMS " + "; ".join(g["problems"])
    notes = [f"{k}: {v}" for k, v in s.items() if isinstance(v, str) and ("not evaluated" in v or "not recorded" in v)]
    if notes:
        line += " NOT CHECKED (field missing) " + "; ".join(notes)
    print(line, end=" | ")
PY
}

next_need_gb() {  # largest MIN_FREE_GB of the models after $KEY in ORDER that are not complete; HANDOFF_FREE_GB after the last one
  local k seen=0 need=0 m
  for k in $ORDER; do
    if [ "$seen" = 1 ] && [ ! -f "$RES/$k/COMPLETE" ]; then m=$( model_info "$k" && echo "$MINFREE" ); [ -n "$m" ] && [ "$m" -gt "$need" ] && need=$m; fi
    [ "$k" = "$KEY" ] && seen=1
  done
  [ "$need" = 0 ] && need=$HANDOFF_FREE_GB
  echo "$need"
}

cache_rule() {  # $1 = success | notdone
  local f need; f=$(free_gb); need=$(next_need_gb)
  if [ "$1" = success ]; then rm -rf "$CACHE_DIR"; say "$KEY cache deleted (model finished); free $(free_gb) GB"; return; fi
  if [ "${KEEP_FAILED_CACHE:-1}" = 1 ] && [ "$f" -ge "$need" ]; then
    say "$KEY cache KEPT (not finished; free $f GB >= $need GB needed next). A rerun resumes without downloading."
  else
    rm -rf "$CACHE_DIR"; say "$KEY cache deleted (not finished; free $f GB < $need GB needed next, or KEEP_FAILED_CACHE=0); free $(free_gb) GB"
  fi
}

package() {  # $1 suffix
  local out="$REPO/followup_rel_4lang/results/sizes_${KEY}" dest="$RES/$KEY"
  mkdir -p "$dest" "$out/queue_logs"
  cp "$LOG/${KEY}"_*.log "$out/queue_logs/" 2>/dev/null
  cp "$ST" "$out/queue_logs/QUEUE_STATUS_at_packaging.txt" 2>/dev/null
  local tarf="$dest/sizes_${KEY}_$(date -u +%Y%m%d_%H%M)$1.tar.gz"
  tar -czf "$tarf" -C "$(dirname "$out")" "$(basename "$out")" && tar -tzf "$tarf" > /dev/null \
    && (cd "$dest" && sha256sum "$(basename "$tarf")" >> SHA256SUMS) && echo "$tarf"
}

run_one() {
  KEY=$1
  model_info "$KEY" || { say "$KEY: unknown key, skipped"; return 1; }
  if [ -f "$RES/$KEY/COMPLETE" ]; then say "$KEY already complete, skipped"; return 0; fi
  if { [ "$REPO" = "$QWEN" ] && [ "$QWEN_OK" = 0 ]; } || { [ "$REPO" = "$IVL" ] && [ "$IVL_OK" = 0 ]; }; then
    say "$KEY SKIPPED: its repo $REPO is not set up (only this family is skipped)"; return 1
  fi
  local f; f=$(free_gb)
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then say "$KEY SKIPPED: only ${f} GB free (need $MINFREE GB)"; return 1; fi
  say "$KEY start: $MID@$REV, free ${f} GB"
  if ! download; then say "$KEY DOWNLOAD FAILED (see $LOG/${KEY}_download.log)"; cache_rule notdone; return 1; fi
  say "$KEY downloaded, size on disk $(cat "$LOG/${KEY}_size_on_disk.txt")"
  local smoke=FAIL rel=skipped expa=skipped t0=$SECONDS rc
  if stage smoke "$T_SMOKE"; then
    smoke=ok
    say "$KEY smoke gates: $(gate_report)"
    stage rel "$T_REL"; rc=$?
    case $rc in 0) rel=ok ;; 3) rel="STOPPED_ETA" ;; 124|137) rel="TIMEOUT" ;; *) rel="FAILED(exit $rc)" ;; esac
    stage expa "$T_EXPA"; rc=$?
    case $rc in 0) expa=ok ;; 3) expa="STOPPED_ETA" ;; 124|137) expa="TIMEOUT" ;; *) expa="FAILED(exit $rc)" ;; esac
  else
    say "$KEY smoke/gate FAILED (exit $?); full runs skipped. $(gate_report)See $LOG/${KEY}_smoke.log"
  fi
  local suffix=""; { [ "$smoke" = ok ] && [ "$rel" = ok ] && [ "$expa" = ok ]; } || suffix="_PARTIAL"
  local tarf; if tarf=$(package "$suffix"); then say "$KEY packaged: $tarf"; else say "$KEY PACKAGING FAILED"; tarf=""; fi
  [ -z "$suffix" ] && [ -n "$tarf" ] && touch "$RES/$KEY/COMPLETE"
  if [ -z "$suffix" ]; then cache_rule success; else cache_rule notdone; fi
  say "$KEY finished in $(( (SECONDS - t0) / 60 )) min: smoke=$smoke rel=$rel expa=$expa; free $(free_gb) GB"
  [ -z "$suffix" ]
}

# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
say "queue started: $ORDER"
early_checks
wait_for_overnight
wait_gpu_idle
net_check
repo_checks
unpack_addon
read_overnight_result
batch1_check
delete_8b_caches
declare -a SUMMARY=()
for k in $ORDER; do
  if run_one "$k"; then SUMMARY+=("$k=ok"); else SUMMARY+=("$k=NOT_COMPLETE"); fi
done
say "QUEUE FINISHED: ${SUMMARY[*]}. Download $RES/ and $ST, then DESTROY the instance (Stop keeps charging disk)."
df -h "$W" > "$RES/disk_end.log"; cp "$ST" "$RES/QUEUE_STATUS.txt"; cp -r "$LOG" "$RES/queue_logs" 2>/dev/null
QUEUE_OK=1
exit 0
