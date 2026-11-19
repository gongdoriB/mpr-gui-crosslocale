#!/usr/bin/env bash
# vast4090b (ONE RTX 4090) fr/ja queue, three models in this order (env ONLY="internvl35_8b uitars_15_7b" selects a subset, order kept):
#   (1) internvl25_8b : Exp A for ALL main pairs of the 6 languages (4,740 pairs = 1,896 earlier 4-language + 2,844 new; x O/T/R = 14,220 inferences) + T_retry (262 pairs = 786)   [mpr-ivl]
#   (2) internvl35_8b : REL O/R/C/F new_only (29,148 rows = 26,057 canonical inferences; skipped when step 1 (queue_b2_step1_ivl35_rel.sh) already passed its gate) + Exp A new fr/ja pairs, main
#                       (2,844 pairs = 8,532) + T_retry (148 pairs = 444)                                                                                                       [mpr-ivl]
#   (3) uitars_15_7b  : REL new_only (26,057) + Exp A new pairs, main (8,532) + T_retry (444)                                                                                    [mpr-gui-crosslocale]
# Every model uses the conditions of its earlier 4-language run (batch 1, config, strict flag, system prompt, code hashes, package versions; checked against the earlier run contracts).
# Usage (tmux, in /workspace, with b2_frja_addon_20261116.tar.gz next to this script):  bash queue_b2_frja.sh --plan | bash queue_b2_frja.sh
# Env: ONLY, KEEP_MODEL_CACHE=1 (default: delete a model's cache after its package was written and every step passed), KEEP_FAILED_CACHE (default 1), FORCE_SLOW_NET=1, IGNORE_ETA=1, NET_MIN_MBPS (20),
#   EXPA_BATCH_<KEY> (uitars_15_7b only: 3), REL_H / EXPA_H (full-step limits in hours, defaults per model below), SKIP_REL_internvl35_8b=1.
# FAILURE RULE: a failing step stops THAT model; the queue goes on with the next model only when the failure is model-specific (download/model check, smoke or gate failure, run error, timeout, ETA stop,
#   packaging). It STOPS the whole queue (verdict FAILED, remaining models not started) when the failure is environmental: GPU busy / not a 4090 / nvidia-smi lost, another queue running, network too slow,
#   addon or image-tar sha mismatch, repo or venv missing, setup (venv / code-hash / config identity) failure, disk too low for the next download, a CUDA fatal error text in the step log
#   (CUDA error, device-side assert, Xid, NVML, GPU lost) or a failed GPU/disk re-check after a failed model.
# Status /workspace/QUEUE_B2_STATUS, done marker /workspace/QUEUE_B2_DONE (OK | FAILED steps: ...), per-model packages in /workspace/queue_results/b2_frja/<KEY>/. Rerun = resume (COMPLETE models are skipped).
# Results go to <repo>/followup_rel_6lang/results/{rel6new_<KEY>,expA6_<KEY>}: new folders, never followup_rel_4lang/results.
set -uo pipefail
W="${W:-/workspace}"
ADDON=b2_frja_addon_20261116.tar.gz
ADDON_SHA=32965ed5fac9a2d5b982a33d657fb280fd683f67dbccbf0fc4ba8b56912748b5
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar     # optional
IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3
ALL_MODELS="internvl25_8b internvl35_8b uitars_15_7b"
ORDER=""; for k in $ALL_MODELS; do case " ${ONLY:-$ALL_MODELS} " in *" $k "*) ORDER="$ORDER $k" ;; esac; done; ORDER="${ORDER# }"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"; MIN_DISK_ALWAYS_GB="${MIN_DISK_ALWAYS_GB:-5}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
T_SETUP=$((60*60)); T_MODEL=$((150*60)); T_SMOKE=$((40*60)); T_RETRY=$((60*60))
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_[A-Za-z0-9_]+\.sh}"; SELF_RE='queue_b2_[A-Za-z0-9_]*\.sh'
RES="$W/queue_results/b2_frja"; LOG="$RES/logs"
ST="$W/QUEUE_B2_STATUS"; DONE="$W/QUEUE_B2_DONE"
DAY=$(date -u +%Y%m%d)
FAILED=(); FINISHED=0
setm() { eval "MSTAT_$1=\"\$2\""; }; getm() { eval "echo \"\${MSTAT_$1:-}\""; }   # per-model status (no associative arrays: bash 3.2 on the test Mac)

minfo() {  # minfo KEY -> sets the per-model facts. Estimates (s/inference) = medians of the earlier 4090 runs (estimates, not measured on this run).
  KEY=$1
  case "$KEY" in
    internvl25_8b) MID=OpenGVLab/InternVL2_5-8B; REV=e9e4c0dc1db56bfab10458671519b7fa3dd29463; REPO_NAME=mpr-ivl; HAS_REL=0; SCOPE=all6; MINFREE=30; BYTES_MIN=16150800432; BYTES_MAX=16150800432
                   N_EXPA=14220; N_RETRY=786; T_REL=0; T_EXPA=$(( ${EXPA_H:-3} * 3600 )); EST="0.29"; LABELS="290,309,289,302"; TOKLOADER=internvl; ARCH_CHECK=internlm2 ;;
    internvl35_8b) MID=OpenGVLab/InternVL3_5-8B; REV=9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79; REPO_NAME=mpr-ivl; HAS_REL=1; SCOPE=new; MINFREE=30; BYTES_MIN=16000000000; BYTES_MAX=18000000000
                   N_EXPA=8532; N_RETRY=444; T_REL=$(( ${REL_H:-9} * 3600 )); T_EXPA=$(( ${EXPA_H:-4} * 3600 )); EST="0.75"; LABELS="32,33,34,35"; TOKLOADER=internvl; ARCH_CHECK=none ;;
    uitars_15_7b)  MID=ByteDance-Seed/UI-TARS-1.5-7B; REV=683d002dd99d8f95104d31e70391a39348857f4e; REPO_NAME=mpr-gui-crosslocale; HAS_REL=1; SCOPE=new; MINFREE=48; BYTES_MIN=32500000000; BYTES_MAX=34000000000
                   N_EXPA=8532; N_RETRY=444; T_REL=$(( ${REL_H:-11} * 3600 )); T_EXPA=$(( ${EXPA_H:-4} * 3600 )); EST="1.0"; LABELS="32,33,34,35"; TOKLOADER=auto; ARCH_CHECK=qwen25vl ;;
    *) echo "unknown model $KEY"; exit 2 ;;
  esac
  R="$W/$REPO_NAME"; CACHE_DIR="$R/.cache/huggingface/hub/models--${MID//\//--}"
  OUT_REL="$R/followup_rel_6lang/results/rel6new_$KEY"; OUT_EXA="$R/followup_rel_6lang/results/expA6_$KEY"
  ebv="EXPA_BATCH_$KEY"; EXPA_BATCH_V="${!ebv:-1}"
  case "$KEY:$EXPA_BATCH_V" in uitars_15_7b:1|uitars_15_7b:3|internvl25_8b:1|internvl35_8b:1) ;; *) echo "EXPA_BATCH_$KEY=$EXPA_BATCH_V not allowed (earlier runs: batch 1; uitars_15_7b also 3)"; exit 2 ;; esac
  rs="SKIP_REL_$KEY"; [ "${!rs:-0}" = 1 ] && HAS_REL=0
  return 0
}

for k in $ORDER; do ( minfo "$k" ) || exit 2; done   # validates per-model env (EXPA_BATCH_<KEY>) before anything starts
PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1
say() { local l; l="$(date -u +%FT%TZ) b2_frja: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLAN
Plan (nothing is executed). Models in this order: ${ORDER}
 0. refuse to start (no waiting) when another queue/overnight script [$OTHERS] runs, a compute process uses the GPU, or GPU memory used >= ${GPU_BUSY_MIB} MiB; flock against a second copy;
    network check (${NET_SECS}s ranged download from huggingface.co; STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1); RTX 4090 check; $ADDON sha256 = $ADDON_SHA (image tar optional, sha256 = $IMAGES_SHA)
 per model:
   setup (limit $((T_SETUP/60)) min): addon files are copied into the repo only where no file exists yet (an existing file must be byte-identical, else STOP); addon SHA256SUMS; repo is a descendant of $BASE_COMMIT;
     venv = the earlier runs' package versions, inference-code hashes, config, system prompt equal the earlier 4-language contracts (nothing is installed); 4-/6-language data + images verify by sha256
     (fr/ja images: image tar if present, else copied from /workspace/mpr-ivl, else fetched from the pinned HF revision)
   disk check before the download (need MINFREE GB, skipped when cached): internvl25_8b 30, internvl35_8b 30, uitars_15_7b 48 (weights 33.2 GB, fp32)
   model (limit $((T_MODEL/60)) min): hf download @ pinned revision (3 tries); safetensors byte sum; tokenizer A-D ids (InternVL2.5: 290,309,289,302; others 32,33,34,35)
   smoke + gate for each run, ETA check against the step limit, full runs with --resume + completion gates, packaging with sha256, then cache deletion (default) when everything passed
 (1) internvl25_8b  Exp A scope all6: 4,740 pairs x O/T/R = 14,220 inferences (batch 1, strict true, as its earlier Exp A run) + T_retry 262 pairs = 786; anchored on its REL6 results when present
 (2) internvl35_8b  REL new_only 26,057 inferences (batch 1, strict false; skipped if rel6new_internvl35_8b/full already passed) + Exp A new pairs 8,532 + T_retry 444 (batch 1, strict false)
 (3) uitars_15_7b   REL new_only 26,057 (batch 1, --strict-batch-size) + Exp A new pairs 8,532 + T_retry 444 (batch 1, strict true)
 Estimated times (estimates from the earlier 4090 medians, NOT measured here): internvl25_8b 14,220 x 0.29 s = 1.1 h (+786 x 0.29 = 4 min); internvl35_8b REL 26,057 x 0.75 = 5.4 h, Exp A 8,532 x 0.75 = 1.8 h (+ 444 = 6 min);
   uitars_15_7b REL 26,057 x 1.01 = 7.3 h, Exp A 8,532 x 0.98 = 2.3 h (+ 444 = 7 min); model downloads 16.2 / ~17 / 33.2 GB and 3 model loads about 1 h in total. Total about 19 h (about 13.6 h if step 1 already finished the InternVL3.5 REL).
 Disk (about 67 GB free at the start): one model cache at a time (16.2, ~17, 33.2 GB), deleted after each model succeeded; results/packages < 1.5 GB per model; peak about 36 GB for UI-TARS.
 Failure rule: see the header of this script (model-specific failure: continue with the next model; environmental failure: stop the queue).
PLAN
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue_b2.lock"; flock -n 9 || { echo "another queue_b2 script is already running"; exit 1; }; }
retry() { local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n failed: $*" >&2; [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i )); done; return 1; }
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc 0-9; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -i 'no running' | sed '/^$/d'; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }

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

package() {  # package <KEY> <suffix>
  local k=$1 suf=$2 dest="$RES/$1" stage="$RES/_stage_$1" dirs=() d out
  minfo "$k"; mkdir -p "$dest"
  out="$dest/b2_${k}_results_${DAY}${suf}.tar.gz"
  rm -rf "$stage"; mkdir -p "$stage"
  for d in rel6new_$k expA6_$k; do [ -d "$R/followup_rel_6lang/results/$d" ] && dirs+=("followup_rel_6lang/results/$d"); done
  if [ -d "$R" ]; then
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    [ ${#dirs[@]} -gt 0 ] && (cd "$R" && find "${dirs[@]}" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true
  fi
  mkdir -p "$stage/queue_logs"; cp "$LOG/"*"_$k.log" "$stage/queue_logs/" 2>/dev/null || true; cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  if [ ${#dirs[@]} -gt 0 ]; then tar -czf "$out" -C "$R" "${dirs[@]}" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && tar -tzf "$out" > /dev/null && (cd "$dest" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256") \
    && say "$k packaged: $out ($(du -h "$out" | cut -f1)); sha256 $(cut -d' ' -f1 "$out.sha256")"
  local rc=$?; rm -rf "$stage"; return $rc
}

finish() {
  [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  local verdict="OK" k line=""
  [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  for k in $ORDER; do v=$(getm "$k"); line="$line $k=${v:-not_started}"; done
  say "models:$line"
  say "FINISHED $verdict. Packages: $RES/<KEY>/"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_B2_DONE written, rerun to resume"; FINISHED=1; exit 130' INT TERM
abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }

rm -f "$DONE"
say "queue_b2_frja started; models: $ORDER"
# ---- 0. nothing else may run ------------------------------------------------------------------
others=""; for p in $(pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" 2>/dev/null); do [ "$p" = "$$" ] && continue; a=$(ps -o args= -p "$p" 2>/dev/null) || continue; echo "$a" | grep -q -E "$SELF_RE" && continue; others="$others $p:$(echo "$a" | cut -c1-60)"; done
[ -n "$others" ] && abort others "another queue/overnight script is running ($others). Nothing was started."
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
hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/OpenGVLab/InternVL2_5-8B/resolve/e9e4c0dc1db56bfab10458671519b7fa3dd29463/model-00001-of-00004.safetensors}")
say "network check (${NET_SECS}s): huggingface.co ${hf} MB/s; minimum $NET_MIN_MBPS MB/s"; slow=0
case "$hf" in ERR:4*) say "WARNING: test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;; *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
if [ "$slow" = 1 ] && [ "${FORCE_SLOW_NET:-0}" != 1 ]; then abort network "network too slow or unreachable. Nothing was downloaded. FORCE_SLOW_NET=1 overrides"; fi
[ -s "$W/$ADDON" ] || abort addon "$ADDON missing in $W"
echo "$ADDON_SHA  $W/$ADDON" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$ADDON sha256 differs"
HAVE_IMAGES_TAR=0
if [ -s "$W/$IMAGES_TAR" ]; then
  echo "$IMAGES_SHA  $W/$IMAGES_TAR" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$IMAGES_TAR sha256 differs"
  HAVE_IMAGES_TAR=1
fi
export HAVE_IMAGES_TAR
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
  # copy only where nothing exists; an existing file must be byte-identical (e.g. the files the InternVL2.5 REL6 queue unpacked); nothing is ever overwritten
  local stage="$RES/_addon_stage" f
  rm -rf "${stage:?}"; mkdir -p "$stage"; tar -xzf "$W/$ADDON" -C "$stage"
  while IFS= read -r f; do
    if [ -e "$R/$f" ]; then cmp -s "$stage/$f" "$R/$f" || { echo "addon file differs from the existing $R/$f; refusing to overwrite"; exit 1; }
    else mkdir -p "$(dirname "$R/$f")"; cp "$stage/$f" "$R/$f"; fi
  done < <(cd "$stage" && find . -type f | sed 's#^\./##' | sort)
  rm -rf "${stage:?}"
  ( cd followup_rel_6lang/b2_frja && sha256sum -c SHA256SUMS )
  if [ "$HAVE_IMAGES_TAR" = 1 ] && [ ! -d followup_rel_6lang/assets ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  repo_env; export PIP_CACHE_DIR="$R/.cache/pip" PIP_NO_CACHE_DIR=1
  source .venv/bin/activate
  case "$KEY" in
    internvl25_8b) CFGN=internvl2_5_8b.yaml; E_EXPA=contracts/ref_internvl25_expA_20261114.json; E_REL=""; E_CODE=$E_EXPA ;;
    internvl35_8b) CFGN=internvl3_5_8b_4090.yaml; E_EXPA=contracts/ref_internvl35_expA_20261114.json; E_REL=contracts/ref_internvl35_rel_20261109.json; E_CODE=$E_REL ;;
    uitars_15_7b)  CFGN=uitars_1_5_7b_4090.yaml; E_EXPA=contracts/ref_uitars_expA_20261114.json; E_REL=contracts/ref_uitars_rel_20261114.json; E_CODE=$E_REL ;;
  esac
  B=followup_rel_6lang/b2_frja
  KEY="$KEY" CONFIG="$B/models/$CFGN" EXPA_REF="$B/$E_EXPA" REL_REF="${E_REL:+$B/$E_REL}" CODE_REF="$B/$E_CODE" B2_PRE_MODEL=1 python "$B/env_check.py"
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
fs = glob.glob(os.path.join(snap, "model-*.safetensors")); size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) >= 4 and int(os.environ["BYTES_MIN"]) <= size <= int(os.environ["BYTES_MAX"]), (len(fs), size)
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg.get("architectures"); llm = (cfg.get("llm_config") or {}).get("architectures")
print("architectures", arch, "| llm", llm, "| template", cfg.get("template"))
if os.environ["ARCH_CHECK"] == "internlm2": assert llm == ["InternLM2ForCausalLM"] and cfg.get("template") == "internvl2_5", (llm, cfg.get("template"))
if os.environ["ARCH_CHECK"] == "qwen25vl": assert arch == ["Qwen2_5_VLForConditionalGeneration"], arch
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
smoke_rel()  { cd "$R"; repo_env; bash followup_rel_6lang/b2_frja/run_rel6_b2.sh "$KEY" smoke; }
full_rel()   { cd "$R"; repo_env; REL_LIMIT_S="$T_REL" bash followup_rel_6lang/b2_frja/run_rel6_b2.sh "$KEY" full; }
smoke_expa() { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" bash followup_rel_6lang/b2_frja/run_expA6_b2.sh "$KEY" smoke; }
full_expa()  { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" EXPA_LIMIT_S="$T_EXPA" bash followup_rel_6lang/b2_frja/run_expA6_b2.sh "$KEY" full; }
retry_expa() { cd "$R"; repo_env; SCOPE="$SCOPE" EXPA_BATCH="$EXPA_BATCH_V" bash followup_rel_6lang/b2_frja/run_expA6_b2.sh "$KEY" retry; }
export -f retry setup model repo_env smoke_rel full_rel smoke_expa full_expa retry_expa
export W RES LOG BASE_COMMIT ADDON IMAGES_TAR

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
free_failed_caches() {  # make room for the next download by removing the kept cache(s) of earlier failed models of this queue
  local k
  for k in $ORDER; do
    [ "$k" = "$KEY" ] && break
    [ -f "$RES/$k/COMPLETE" ] && continue
    ( minfo "$k"; if [ -d "$CACHE_DIR" ]; then rm -rf "${CACHE_DIR:?}"; echo "$(date -u +%FT%TZ) b2_frja: removed the kept cache of $k to make room (free now $(free_gb) GB); a rerun of $k downloads it again" | tee -a "$ST" "$RES/STATUS"; fi )
  done
}

run_model() {
  minfo "$1"; step_failed=0; ENV_FAIL=0; ENV_REASON=""
  export KEY MID REV R W RES LOG BYTES_MIN BYTES_MAX LABELS TOKLOADER ARCH_CHECK SCOPE T_REL T_EXPA EXPA_BATCH_V BASE_COMMIT ADDON
  if [ -f "$RES/$KEY/COMPLETE" ]; then say "$KEY already complete (package exists), skipped"; setm "$KEY" "complete_earlier"; return 0; fi
  if ! { [ -d "$R/01_code" ] && [ -f "$R/.venv/bin/activate" ]; }; then say "STOP: $R (repo with 01_code and .venv) does not exist"; FAILED+=("repo_$KEY"); setm "$KEY" "STOPPED_repo"; ENV_FAIL=1; return 1; fi
  say "=== $KEY: $MID @ $REV (repo $REPO_NAME, Exp A scope $SCOPE, batch $EXPA_BATCH_V) ==="
  if ! step setup "$T_SETUP" setup; then setm "$KEY" "FAILED_setup"; return 1; fi
  local f; f=$(free_gb)
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then free_failed_caches; f=$(free_gb); fi
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then say "STOP: only ${f} GB free (need $MINFREE GB for $KEY)"; FAILED+=("disk_$KEY"); setm "$KEY" "STOPPED_disk"; ENV_FAIL=1; ENV_REASON="disk"; return 1; fi
  if ! step model "$T_MODEL" model; then setm "$KEY" "FAILED_model"; return 1; fi
  local rel_ok=0 expa_ok=0
  if [ "$HAS_REL" = 1 ]; then
    if grep -q '"status": "passed"' "$OUT_REL/full/gate_rel_complete.json" 2>/dev/null; then say "$KEY REL fr/ja already complete (gate passed earlier); REL steps skipped"
    elif step smoke_rel "$T_SMOKE" smoke_rel; then rel_ok=1; say "smoke gate: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "$KEY rel fr/ja")"
    else say "$KEY REL smoke failed: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "$KEY rel")see $LOG/smoke_rel_$KEY.log; REL full skipped"; fi
  fi
  if step smoke_expa "$T_SMOKE" smoke_expa; then expa_ok=1; say "smoke gate: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "$KEY expA $SCOPE")"
  else say "$KEY Exp A smoke failed: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "$KEY expA")see $LOG/smoke_expa_$KEY.log; Exp A full skipped"; fi
  [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
  [ "$rel_ok" = 1 ] && step full_rel "$T_REL" full_rel
  [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
  if [ "$expa_ok" = 1 ]; then step full_expa "$T_EXPA" full_expa && step retry_expa "$T_RETRY" retry_expa; fi
  [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
  return $step_failed
}

for K in $ORDER; do
  run_model "$K"; rc=$?
  minfo "$K"
  suf=""; [ "$step_failed" = 1 ] || [ $rc -ne 0 ] && suf="_PARTIAL"
  if package "$K" "$suf"; then pk=1; else pk=0; say "$K packaging FAILED"; FAILED+=("package_$K"); fi
  if [ -z "$suf" ] && [ "$pk" = 1 ]; then
    touch "$RES/$K/COMPLETE"; setm "$K" "ok"
    if [ -d "$CACHE_DIR" ]; then
      if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "$K cache kept (KEEP_MODEL_CACHE=1)"; else rm -rf "${CACHE_DIR:?}"; say "$K cache deleted (model finished); free disk: $(free_gb) GB"; fi
    fi
  else
    [ -z "$(getm "$K")" ] && setm "$K" "FAILED_steps"
    [ -d "$CACHE_DIR" ] && say "$K cache KEPT (not finished; free $(free_gb) GB); it is removed automatically if the next download needs the room"
  fi
  if [ "$ENV_FAIL" = 1 ]; then say "STOP QUEUE (environmental failure: ${ENV_REASON:-see step log}); remaining models are not started"; break; fi
  if [ $rc -ne 0 ] || [ -n "$suf" ]; then
    if ! env_recheck; then say "STOP QUEUE (environment re-check after the failed $K: $ENV_REASON); remaining models are not started"; FAILED+=("env_after_$K"); break; fi
    say "$K failed with a model-specific problem; environment re-check ok; continuing with the next model"
  fi
done
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
