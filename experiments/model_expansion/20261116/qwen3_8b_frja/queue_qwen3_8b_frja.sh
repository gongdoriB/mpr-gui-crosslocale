#!/usr/bin/env bash
# Qwen/Qwen3-VL-8B-Instruct on the NEW Vast.ai RTX 4090 "vast4090c" (Ubuntu 24.04, driver 580, CUDA 13.0, 65 GB disk, nothing else runs there):
#   (1) consistency check: 600 OLD 4-language REL inferences re-run and compared with the earlier 4090 run (report only)
#   (2) REL O/R/C/F rows that involve fr or ja (new_only: 29,148 condition rows -> 26,057 canonical inferences)
#   (3) Experiment A (O/T/R) for the new directed pairs that involve fr or ja (3,060 pairs x 3 = 9,180 inferences) + T_retry sensitivity (159 pairs x 3 = 477)
# Chain: GPU must be free (refuses to start otherwise) -> network check -> checks -> setup in the EXISTING /workspace/mpr-gui-crosslocale (addon unpack, fr/ja images,
#   repo/venv/data checks) -> pinned model -> smoke REL + gate -> smoke Exp A + gate -> consistency check -> REL full + gate -> Exp A full + gate -> T_retry + gate
#   -> package -> /workspace/QUEUE_C_DONE. The model cache is KEPT at the end (DELETE_MODEL_CACHE=1 deletes it).
# Usage (inside tmux, in /workspace, with qwen3_8b_frja_addon_20261116.tar.gz next to this script; the fr/ja image tar is optional, see below):
#   bash queue_qwen3_8b_frja.sh --plan   # only print what would run; changes nothing
#   bash queue_qwen3_8b_frja.sh          # starts at once if the GPU is idle
# Environment: REL_BATCH (default 1), EXPA_BATCH (default 3), CC_BATCH (default = REL_BATCH), EXPA_SPLIT (all|main|pilot, default all),
#   REL_H (REL full step limit in hours, default 8), EXPA_H (default 3), CC_H (default 1), IGNORE_ETA=1, FORCE_SLOW_NET=1, MIN_FREE_GB (default 30), NET_MIN_MBPS (default 20),
#   DELETE_MODEL_CACHE=1, SKIP_CONSISTENCY=1, MODEL_BYTES (exact byte sum of the four safetensors if known; otherwise a 16.5-18.5 GB range is checked).
# fr/ja screenshots (729 files, 748 MB): if $W/internvl25_rel6_frja_images_20261116.tar is present it is verified (sha256) and unpacked (same file as for the InternVL2.5 queue),
#   otherwise they are downloaded from huggingface.co at the pinned dataset revision; either way every file is sha256-checked against bundle_manifest.json.
# Rerunning resumes (addon unpack is idempotent, predictions are reused). The instance is NOT stopped or destroyed.
# Progress: tail /workspace/QUEUE_C_STATUS ; per-step logs: /workspace/queue_results/qwen3_8b_frja/logs/
# Written from queue_internvl25_rel6.sh (reviewed). Differences: no waiting for other queues (a busy GPU stops it), Qwen model checks, three result areas, model cache kept.
# Nothing outside /workspace/mpr-gui-crosslocale/followup_rel_6lang/ and the files named below is created; followup_rel_4lang/results is never written.
set -uo pipefail

W="${W:-/workspace}"
BASE_COMMIT="${BASE_COMMIT:-666440011f538cd9336794c9965429de640843c3}"   # the repository state of the earlier Qwen3 runs (the earlier run contracts record this commit)
MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b      # same as the earlier REL (2026-11-07) and Exp A (2026-11-14) runs
MODEL_BYTES="${MODEL_BYTES:-}"                         # exact size not known locally (확인 필요); empty -> 16.5-18.5 GB range check
ADDON=qwen3_8b_frja_addon_20261116.tar.gz
ADDON_SHA=661d65dc694bb75b89f11f547e39cecc7d9f9d0a734fdae1b0f4fc5d3fe15fed
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar     # optional
IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
export REL_BATCH="${REL_BATCH:-1}" EXPA_BATCH="${EXPA_BATCH:-3}" EXPA_SPLIT="${EXPA_SPLIT:-all}"
export CC_BATCH="${CC_BATCH:-$REL_BATCH}"
case "$EXPA_SPLIT" in all) EXPA_N=9180; EXPA_RN=477 ;; main) EXPA_N=8532; EXPA_RN=444 ;; pilot) EXPA_N=648; EXPA_RN=33 ;; *) echo "EXPA_SPLIT must be all, main or pilot"; exit 2 ;; esac
case "$REL_BATCH" in 1|2|3|4) ;; *) echo "REL_BATCH must be 1..4"; exit 2 ;; esac
case "$EXPA_BATCH" in 1|3) ;; *) echo "EXPA_BATCH must be 1 or 3"; exit 2 ;; esac
MIN_FREE_GB="${MIN_FREE_GB:-30}"                        # model ~17.5 GB x 1.3 + images 0.75 GB x 2 + results ~1 GB + temp
POLL_S="${POLL_S:-60}"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
export EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"      # only changed in local tests
T_SETUP=$((90*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_CC=$(( ${CC_H:-1} * 3600 )); T_REL=$(( ${REL_H:-8} * 3600 )); T_EXPA=$(( ${EXPA_H:-3} * 3600 )); T_RETRY=$((60*60))

R="$W/mpr-gui-crosslocale"
RES="$W/queue_results/qwen3_8b_frja"
LOG="$RES/logs"
ST="$W/QUEUE_C_STATUS"; DONE="$W/QUEUE_C_DONE"
RDIR="$R/followup_rel_6lang/results"
OUT_REL="$RDIR/rel6new_qwen3_vl_8b"; OUT_CC="$RDIR/consistency_qwen3_vl_8b"; OUT_EXA="$RDIR/expA6_qwen3_vl_8b"
CACHE_DIR="$R/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
DAY=$(date -u +%Y%m%d)
FAILED=(); NONFATAL=()
FINISHED=0

PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1

say() { local l; l="$(date -u +%FT%TZ) qwen3_frja: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLANEOF
Plan (nothing is executed):
 0. refuse to start if a process uses the GPU (nvidia-smi compute apps, or memory used >= ${GPU_BUSY_MIB} MiB): nothing else is expected on this server. No waiting.
    Then network check (curl ${NET_SECS}s ranged download from huggingface.co (the model's first shard) and download.pytorch.org (informational); STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1)
 1. checks: $ADDON present and sha256 = $ADDON_SHA; ${IMAGES_TAR} (optional) sha256 = $IMAGES_SHA if present; GPU is RTX 4090; free disk >= ${MIN_FREE_GB} GB;
    $R exists with 01_code and .venv (.git is checked when present)
 2. setup (limit $((T_SETUP/60)) min): the addon may only create files under 01_code/src/mpr_crosslocale/{data/rel_followup_6lang.py,inference/run_rel_followup_6lang.py} and
    followup_rel_6lang/ (listing checked before unpacking); unpack, verify its SHA256SUMS; unpack the image tar if present; venv must hold torch 2.9.1+cu130, transformers 4.57.6,
    accelerate 1.15.0, qwen-vl-utils 0.0.14, torchvision 0.24.1+cu130 (otherwise STOP; nothing is pip-installed); the 12 inference-code files of the repo must have the sha256 values of
    the 2026-11-07 Qwen3-VL-8B REL run contract; 4-language images verify (fetch if missing); 6-language data + fr/ja images verify by sha256 (fetch the missing ones from the pinned HF revision)
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION (~17.5 GB, 3 tries, partial files resume); 4 safetensors shards, byte sum ${MODEL_BYTES:-in 16.5-18.5 GB (exact size 확인 필요)};
    config.json architecture Qwen3VLForConditionalGeneration; tokenizer: A, B, C, D are single tokens with ids 32, 33, 34, 35 (as in the earlier run)
 4. smoke REL (--max-items 2, new_only, batch $REL_BATCH; limit $((T_SMOKE/60)) min) + gate; smoke Exp A (--max-items 3: 162 inferences, batch $EXPA_BATCH; limit $((T_SMOKE/60)) min) + gate.
    A failed smoke skips only its own full run.
 5. consistency check (limit $((T_CC/3600)) h): 600 old 4-language REL inferences (seed 20261116, 52 condition x language-pair cells), batch $CC_BATCH, results in $OUT_CC;
    compares with the earlier run -> consistency_agreement.json; its failure never stops the queue (recorded as non-fatal)
 6. REL fr/ja full: 26,057 inferences / 29,148 condition rows, batch $REL_BATCH (limit $((T_REL/3600)) h; not started if the smoke ETA exceeds it, IGNORE_ETA=1 overrides) + gate
    -> $OUT_REL/full
 7. Exp A fr/ja full: $EXPA_N inferences (split $EXPA_SPLIT), batch $EXPA_BATCH (limit $((T_EXPA/3600)) h, same ETA rule) + gate (O/R answers compared with the REL run) -> $OUT_EXA/full;
    then T_retry sensitivity: $EXPA_RN inferences (limit $((T_RETRY/60)) min) -> $OUT_EXA/retry
 8. package to $RES/qwen3_8b_frja_results_<date>.tar.gz (+ .sha256, + per-file sha256 list); model cache KEPT unless DELETE_MODEL_CACHE=1; append to $ST;
    write $DONE (OK, "OK (non-fatal: ...)" or FAILED steps: ...). Nothing is written into followup_rel_4lang/results.
 Estimated time (estimate, not measured on this server): earlier Qwen3-VL-8B medians on a 4090: REL batch 4 0.587 s/inference, REL batch 1 0.700 s, Exp A batch 3 0.613 s.
 REL 26,057 x 0.70 = 5.1 h (batch 1) or x 0.587 = 4.2 h (batch 4); Exp A 9,180 x 0.613 = 1.6 h; consistency 600 x 0.70 = 7 min; T_retry 477 x 0.61 = 5 min; model download + 4 model loads + checks 0.5-1 h.
 Disk (65 GB disk, repo and venv already there): model 17.5 GB + fr/ja images 0.75 GB (+0.75 GB image tar if uploaded) + results ~1 GB + package ~0.6 GB = about 21 GB new data.
PLANEOF
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue_c_qwen3.lock"; flock -n 9 || { echo "another queue_qwen3_8b_frja.sh is already running"; exit 1; }; }

retry() {  # retry <tries> <cmd...>: network steps on flaky hosts (SSL EOF / gnutls handshake errors seen on the previous server)
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "retry $i/$n failed: $*" >&2
    [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i ))
  done
  return 1
}

step() {  # step <name> <limit_s> <function> [nonfatal]; runs the function in a subshell with a time limit
  local name=$1 limit=$2 fn=$3 nonfatal=${4:-} rc
  say "step $name started (limit $((limit/60)) min)"
  timeout -k 60 "$limit" bash -c "set -Eeo pipefail; $fn" >> "$LOG/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "step $name ok"; return 0; fi
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then say "step $name TIMEOUT"
  elif [ $rc -eq 3 ] && { [ "$name" = full_rel ] || [ "$name" = full_expa ]; }; then
    say "STOP step $name: smoke ETA longer than the step limit. IGNORE_ETA=1 overrides."; FAILED+=("${name}_eta"); return $rc
  else say "step $name FAILED (exit $rc), see $LOG/$name.log"; fi
  if [ "$nonfatal" = nonfatal ]; then NONFATAL+=("$name"); else FAILED+=("$name"); fi
  return $rc
}

package() {
  local out="$RES/qwen3_8b_frja_results_${DAY}.tar.gz" stage="$RES/_stage" dirs=() d
  rm -rf "$stage"; mkdir -p "$stage"
  for d in rel6new_qwen3_vl_8b consistency_qwen3_vl_8b expA6_qwen3_vl_8b; do [ -d "$RDIR/$d" ] && dirs+=("followup_rel_6lang/results/$d"); done
  if [ -d "$R" ]; then
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -7 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    [ ${#dirs[@]} -gt 0 ] && (cd "$R" && find "${dirs[@]}" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true
  fi
  cp -r "$LOG" "$stage/queue_logs" 2>/dev/null || true
  cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true
  df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  if [ ${#dirs[@]} -gt 0 ]; then tar -czf "$out" -C "$R" "${dirs[@]}" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && sha256sum "$out" > "$out.sha256" && say "packaged $out ($(du -h "$out" | cut -f1))"
  rm -rf "$stage"
}

finish() {  # runs once, also on error exits
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  package || say "packaging FAILED"
  local verdict="OK"
  [ ${#NONFATAL[@]} -gt 0 ] && verdict="OK (non-fatal: ${NONFATAL[*]} failed)"
  [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  if [ -d "$CACHE_DIR" ]; then
    if [ "${DELETE_MODEL_CACHE:-0}" = 1 ] && [ ${#FAILED[@]} -eq 0 ]; then rm -rf "$CACHE_DIR"; say "model cache deleted (DELETE_MODEL_CACHE=1); free disk: $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
    else say "model cache KEPT at $CACHE_DIR (DELETE_MODEL_CACHE=1 deletes it after a successful run)"; fi
  fi
  say "FINISHED $verdict. Results: $RES/qwen3_8b_frja_results_${DAY}.tar.gz"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_C_DONE written, rerun the script to resume"; FINISHED=1; exit 130' INT TERM

abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }   # abort <key> <message>; key = name in "FAILED steps:"

# ---- 0. the GPU must be free; no waiting ---------------------------------------------------------
rm -f "$DONE"
say "queue_qwen3_8b_frja started (REL batch $REL_BATCH, Exp A batch $EXPA_BATCH, consistency batch $CC_BATCH, Exp A split $EXPA_SPLIT)"
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -i 'no running' | sed '/^$/d')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
[ -z "$used" ] && abort gpu "nvidia-smi not available"
if [ -n "$apps" ] || [ "$used" -ge "$GPU_BUSY_MIB" ]; then
  abort gpu "the GPU is in use (${used} MiB used; processes: $(echo "$apps" | head -3 | tr '\n' ';')). Nothing else should run on this server; stop that process first. Nothing was started."
fi
say "GPU idle (${used} MiB used, no compute process)"

# ---- 0b. network check (same method as the earlier queues): 15 s ranged download -------------------
net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}
net_check() {  # download.pytorch.org is measured and logged only; torch is never downloaded by this queue
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
{ [ -d "$R/01_code" ] && [ -f "$R/.venv/bin/activate" ]; } || abort repo "$R (repo copy with 01_code and .venv) does not exist; it is copied/installed by hand before this queue. Not rebuilt here"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free"

# ---- 2. setup -------------------------------------------------------------------------------
setup() {
  cd "$R"
  if [ -d .git ]; then
    git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null
    git merge-base --is-ancestor "$BASE_COMMIT" HEAD
    echo "HEAD $(git rev-parse --short HEAD), $(git rev-list --count "$BASE_COMMIT"..HEAD) commits after ${BASE_COMMIT:0:7}"
  else
    echo "WARNING: no .git in $R; the commit cannot be checked, the code identity check below decides"
  fi
  # the addon may only add files of its own: refuse anything else before unpacking (tar would overwrite silently)
  tar -tzf "$W/$ADDON" > "$RES/addon_listing.txt"
  if grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | grep -q .; then
    echo "addon contains paths outside its own files:"; grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon_listing.txt" | head; exit 1
  fi
  tar -xzf "$W/$ADDON"
  ( cd followup_rel_6lang/q3_frja && sha256sum -c SHA256SUMS )
  if [ "$HAVE_IMAGES_TAR" = 1 ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" PIP_CACHE_DIR="$R/.cache/pip" PIP_NO_CACHE_DIR=1
  source .venv/bin/activate
  # venv: the exact package set of the earlier Qwen3 runs; nothing is installed here
  REPO_DIR="$PWD" python - <<'PY'
import importlib.metadata as md, os
import mpr_crosslocale, transformers, torch
want = {"torch": os.environ["EXPECT_TORCH"], "transformers": "4.57.6", "accelerate": "1.15.0", "qwen-vl-utils": "0.0.14", "torchvision": os.environ.get("EXPECT_TV", "0.24.1+cu130")}
for k, v in want.items():
    if k in os.environ.get("EXPECT_SKIP", "").split(","):   # local test hook only (the Mac test venv has no qwen-vl-utils)
        continue
    have = md.version(k)
    assert have == v, f"{k} {have} != {v} (the venv must be the exact pip freeze of the earlier Qwen3 runs; this queue installs nothing)"
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"])), mpr_crosslocale.__file__
print("venv ok:", ", ".join(f"{k} {md.version(k)}" for k in want if k not in os.environ.get("EXPECT_SKIP", "").split(",")))
PY
  python -c 'import mpr_crosslocale.data.rel_followup_6lang, mpr_crosslocale.inference.run_rel_followup_6lang; print("6-language modules import ok")'
  # inference code of the repo = the code of the earlier runs (before any large download)
  REFERENCE=followup_rel_6lang/q3_frja/reference_contract_qwen3_rel_20261107.json CONFIG=followup_rel_6lang/q3_frja/qwen3_vl_8b_4090_channels_last.yaml python - <<'PY'
import json, os
from pathlib import Path
import yaml
from mpr_crosslocale.inference.run_rel_followup import code_identity
ref = json.loads(Path(os.environ["REFERENCE"]).read_text())
now = code_identity()
diff = sorted(k for k in set(now) | set(ref["code_sha256"]) if k != "run_rel_T.py" and ref["code_sha256"].get(k) != now.get(k))
assert not diff, f"inference code of the repository differs from the 2026-11-07 Qwen3-VL-8B run: {diff}"
assert yaml.safe_load(Path(os.environ["CONFIG"]).read_text()) == ref["model_config"]
print("12 inference-code hashes and the model config equal the 2026-11-07 contract")
PY
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
assert len(fs) == 4, len(fs)
if os.environ.get("MODEL_BYTES"):
    assert size == int(os.environ["MODEL_BYTES"]), size
else:
    assert 16.5e9 <= size <= 18.5e9, size
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg["architectures"]
print("architectures", arch, "| model_type", cfg.get("model_type"))
assert arch == ["Qwen3VLForConditionalGeneration"] and cfg.get("model_type") == "qwen3_vl", (arch, cfg.get("model_type"))
open(os.path.join(os.environ["RES"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\narchitectures={arch}\n")
# fail fast on a tokenizer / transformers mismatch before the GPU is used
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
assert ids == {"A": [32], "B": [33], "C": [34], "D": [35]}, ids   # ids recorded in the 2026-11-07 REL run (label_token_ids)
PY
  du -sh "$HF_HOME"
}

# Vast sets HF_HOME=/workspace/.hf_home in /etc/environment; the run scripts keep an inherited HF_HOME, so the cache variables are exported here
# (exports are inline: steps run in a fresh bash that only gets the step function itself)
smoke_rel()    { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh smoke; }
smoke_expa()   { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; bash followup_rel_6lang/q3_frja/run_expA6_qwen3.sh smoke; }
consistency()  { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; bash followup_rel_6lang/q3_frja/run_consistency_qwen3.sh run; }
full_rel()     { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; REL_LIMIT_S="$T_REL" bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh full; }
full_expa()    { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; EXPA_LIMIT_S="$T_EXPA" bash followup_rel_6lang/q3_frja/run_expA6_qwen3.sh full; }
retry_expa()   { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; bash followup_rel_6lang/q3_frja/run_expA6_qwen3.sh retry; }
export -f retry setup model smoke_rel smoke_expa consistency full_rel full_expa retry_expa
export W R RES LOG BASE_COMMIT ADDON IMAGES_TAR MODEL_ID REVISION MODEL_BYTES T_REL T_EXPA

smoke_report() {  # $1 gate json, $2 label: accuracy, speed, ETA and warnings
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
try:
    g = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
line = f"{sys.argv[2]}: acc {s.get('accuracy')} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h prompt tokens {s.get('prompt_token_count')}"
if g.get("warnings"):
    line += " WARN " + "; ".join(g["warnings"])
if g.get("problems"):
    line += " PROBLEMS " + "; ".join(g["problems"])
print(line, end=" | ")
PY
}

rel_ok=0; expa_ok=0
if step setup "$T_SETUP" setup && step model "$T_MODEL" model; then
  if step smoke_rel "$T_SMOKE" smoke_rel; then rel_ok=1; say "smoke gate: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "rel fr/ja")"
  else say "REL smoke failed: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "rel fr/ja")see $LOG/smoke_rel.log; the REL full run and the consistency check are skipped"; fi
  if step smoke_expa "$T_SMOKE" smoke_expa; then expa_ok=1; say "smoke gate: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "expA fr/ja")"
  else say "Exp A smoke failed: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "expA fr/ja")see $LOG/smoke_expa.log; the Exp A full run is skipped"; fi
  if [ "$rel_ok" = 1 ]; then
    if [ "${SKIP_CONSISTENCY:-0}" = 1 ]; then say "consistency check skipped (SKIP_CONSISTENCY=1)"
    else step consistency "$T_CC" consistency nonfatal && say "consistency: $(python3 -c "import json;d=json.load(open('$OUT_CC/consistency_agreement.json'))['overall'];print('label agreement',d['label_agreement'],'accuracy old',d['old_accuracy'],'new',d['new_accuracy'])" 2>/dev/null)"; fi
    step full_rel "$T_REL" full_rel || true
  fi
  if [ "$expa_ok" = 1 ]; then
    step full_expa "$T_EXPA" full_expa && step retry_expa "$T_RETRY" retry_expa || true
  fi
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
