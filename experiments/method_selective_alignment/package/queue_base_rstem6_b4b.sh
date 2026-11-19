#!/usr/bin/env bash
# 2026-11-19 VARIANT queue_rstem6_b4b.sh for vast4090b: the two supplementary 4B models (qwen3_vl_4b in mpr-gui-crosslocale, internvl35_4b in mpr-ivl), batch 1 strict,
# contracts = their Exp A 6-language runs on this server (2026-11-17; the REL runs used the separate 6lang runner, so their code hashes cover other files). Status /workspace/QUEUE_RSTEM6B_STATUS, done /workspace/QUEUE_RSTEM6B_DONE. Everything else as queue_rstem6_c.sh 71110d18.
# R_stem / R_opts 6-language run (O, R_stem, R_opts, R: 4,740 pairs x 4 = 18,960 inferences per model) on vast4090c (ONE RTX 4090), five models in this order
# (env ONLY="minicpm_v45 uitars_15_7b" selects a subset, order kept):
#   (1) qwen3_vl_8b   Qwen/Qwen3-VL-8B-Instruct                   repo /workspace/mpr-gui-crosslocale (exists)   batch 4, strict   (as the 2026-11-09 R_stem/R_opts run)
#   (2) minicpm_v45   openbmb/MiniCPM-V-4_5                       repo /workspace/mpr-minicpm (exists, model cache present)  batch 1
#   (3) uitars_15_7b  ByteDance-Seed/UI-TARS-1.5-7B               repo /workspace/mpr-gui-crosslocale (exists)   batch 1, strict
#   (4) internvl35_8b OpenGVLab/InternVL3_5-8B                    repo /workspace/mpr-ivl (NEW: clone + patches 0001-0005 + venv)  batch 1
#   (5) internvl25_8b OpenGVLab/InternVL2_5-8B                    repo /workspace/mpr-ivl                         batch 1, strict
# Batch size, strict flag, model config, system prompt, inference-code hashes and package versions are those of the earlier run of the same model (contracts/ref_*.json in the addon;
# checked before every model; a difference stops that model before any GPU work).
# Per model: setup (addon into the repo, data/images/frozen file checks) -> pinned download (exact byte sum, tokenizer A-D ids) -> smoke (240 inferences) + gate -> ETA check -> full 18,960 with --resume
#   -> completion gate -> package with per-file sha256 -> model cache deleted (default).
# Usage (tmux, in /workspace):  bash queue_rstem6_c.sh --plan | bash queue_rstem6_c.sh
# Files needed in /workspace (sha256 checked at start): this script, rstem6_addon_20261118.tar.gz; already on the server: sixbysix_addon/ (ivl patches + constraints, only when mpr-ivl must be built),
#   internvl25_rel6_frja_images_20261116.tar (fr/ja images for mpr-ivl; optional), minicpm_6lang_addon_20261117.tar.gz (only as a fallback source of the frozen conditions file).
# Env: ONLY, KEEP_MODEL_CACHE=1 (never delete caches), KEEP_FAILED_CACHE (default 1), FORCE_SLOW_NET=1, IGNORE_ETA=1, NET_MIN_MBPS (20), FULL_H_<KEY> (full-step limit in hours),
#   MIN_FREE_GB_<KEY>, OR_SOURCE=<OR_gemma_conditions.jsonl> (optional extra O/R equality check).
# FAILURE RULE (as queue_b2_frja.sh): a model-specific failure (download/model check, smoke or gate failure, run error, timeout, ETA stop, packaging) stops THAT model; the queue goes on with the next
#   model after the GPU/disk re-check. A setup failure (identity check of config / code hashes / packages, images, pip) is also model-specific: the models do not share a setup except qwen3_vl_8b +
#   uitars_15_7b (repo mpr-gui-crosslocale) and the two InternVL models (mpr-ivl). An environmental failure (GPU busy / not a 4090, another queue running, network too slow, addon sha mismatch,
#   a missing repo of an existing family, disk too low, CUDA fatal error text in a step log, GPU/disk re-check failing after a failed model) stops the whole queue (verdict FAILED, remaining models not started).
# Status /workspace/QUEUE_RSTEM6_STATUS, done marker /workspace/QUEUE_RSTEM6_DONE (OK | FAILED steps: ...), packages in /workspace/queue_results/rstem6/<KEY>/. Rerun = resume (COMPLETE models skipped).
# Results go to <repo>/followup_rel_6lang/results/rstem6_<KEY>/ (new folders). The server's AGENTS.md / CLAUDE.md are not read. The instance is NOT stopped.
set -uo pipefail
W="${W:-/workspace}"
ADDON=rstem6_addon_20261119_4b.tar.gz
ADDON_SHA=__ADDON_SHA__
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar; IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
SIX_ADDON=sixbysix_addon_20261114.tar.gz; SIX_SHA=ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a; AD="$W/sixbysix_addon"
MC6_ADDON=minicpm_6lang_addon_20261117.tar.gz; MC6_SHA=dbf726db2f9165e5a91d22058959453ddbab1dc2e9dde17fd09b4ad9b53bef05
FROZEN_REL=followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl; FROZEN_SHA=63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed
REPO_URL="${REPO_URL:-https://github.com/dltmdwoOS/mpr-gui-crosslocale.git}"; BRANCH=followup-rel-4lang-inference   # REPO_URL: only a test hook
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3
CONSTRAINTS_SHA=a3d98e62728f07ad583335ae0b5eae76dd794abd9d8bb48c57fb25d7c8275c48     # sixbysix_addon/internvl35_constraints_20261109.txt
ALL_MODELS="qwen3_vl_4b internvl35_4b"   # 2026-11-18: UI-TARS moved before the InternVL pair (it needs 36 GB free; before the new mpr-ivl repo is built there is ~10 GB more room)
ORDER=""; for k in $ALL_MODELS; do case " ${ONLY:-$ALL_MODELS} " in *" $k "*) ORDER="$ORDER $k" ;; esac; done; ORDER="${ORDER# }"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"; MIN_DISK_ALWAYS_GB="${MIN_DISK_ALWAYS_GB:-5}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
T_SETUP=$((90*60)); T_MODEL=$((150*60)); T_SMOKE=$((40*60))
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_[A-Za-z0-9_]+\.sh}"; SELF_RE='queue_rstem6_[A-Za-z0-9_]*\.sh'
RES="$W/queue_results/rstem6"; LOG="$RES/logs"
ST="$W/QUEUE_RSTEM6B_STATUS"; DONE="$W/QUEUE_RSTEM6B_DONE"
DAY=$(date -u +%Y%m%d)
FAILED=(); FINISHED=0
setm() { eval "MSTAT_$1=\"\$2\""; }; getm() { eval "echo \"\${MSTAT_$1:-}\""; }

minfo() {  # minfo KEY -> per-model facts. EST = s/inference (median / mean) of the earlier 4090 runs: estimates, not measured on this server.
  KEY=$1
  case "$KEY" in
    qwen3_vl_8b)   MID=Qwen/Qwen3-VL-8B-Instruct;        REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; REPO_NAME=mpr-gui-crosslocale; FAM=qwen; BATCH=4; BYTES=17534339512; NFILES=4; LABELS="32,33,34,35"; TOKLOADER=auto;    ARCH_CHECK=qwen3vl;   SETUPGB=0;  EST="0.615 / 0.773 (11-09 R_stem/R_opts run, batch 4)"; EST_H="3.2-4.1"; FULL_H_DEF=8 ;;
    minicpm_v45)   MID=openbmb/MiniCPM-V-4_5;            REV=14a2004908140699317e2109e90573c1e6323c36; REPO_NAME=mpr-minicpm;        FAM=mc;   BATCH=1; BYTES=17391890184; NFILES=4; LABELS="32,33,34,35"; TOKLOADER=minicpm; ARCH_CHECK=none;      SETUPGB=0;  EST="0.512 / 0.539 (Exp A 6-language run)"; EST_H="2.7-2.8"; FULL_H_DEF=6 ;;
    internvl35_8b) MID=OpenGVLab/InternVL3_5-8B;         REV=9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79; REPO_NAME=mpr-ivl;            FAM=ivl;  BATCH=1; BYTES=17056729432; NFILES=4; LABELS="32,33,34,35"; TOKLOADER=internvl; ARCH_CHECK=none;      SETUPGB=10; EST="0.751 / 0.732 (11-10 R_stem/R_opts run), 0.762 / 0.743 (Exp A 6-language run)"; EST_H="3.9-4.0"; FULL_H_DEF=8 ;;
    uitars_15_7b)  MID=ByteDance-Seed/UI-TARS-1.5-7B;    REV=683d002dd99d8f95104d31e70391a39348857f4e; REPO_NAME=mpr-gui-crosslocale; FAM=qwen; BATCH=1; BYTES=33168747240; NFILES=7; LABELS="32,33,34,35"; TOKLOADER=auto;    ARCH_CHECK=qwen25vl;  SETUPGB=0;  EST="1.004 / 1.039 (Exp A 6-language run)"; EST_H="5.3-5.5"; FULL_H_DEF=10 ;;
    internvl25_8b) MID=OpenGVLab/InternVL2_5-8B;         REV=e9e4c0dc1db56bfab10458671519b7fa3dd29463; REPO_NAME=mpr-ivl;            FAM=ivl;  BATCH=1; BYTES=16150800432; NFILES=4; LABELS="290,309,289,302"; TOKLOADER=internvl; ARCH_CHECK=internlm2; SETUPGB=10; EST="0.290 / 0.378 (Exp A 6-language run)"; EST_H="1.5-2.0"; FULL_H_DEF=5 ;;
    qwen3_vl_4b)   MID=Qwen/Qwen3-VL-4B-Instruct;        REV=ebb281ec70b05090aa6165b016eac8ec08e71b17; REPO_NAME=mpr-gui-crosslocale; FAM=qwen; BATCH=1; BYTES=8875719344;  NFILES=2; LABELS="32,33,34,35"; TOKLOADER=auto;    ARCH_CHECK=qwen3vl;   SETUPGB=0;  EST="0.45-0.65 (6x6 queue assumption; not measured for this input)"; EST_H="2.4-3.4"; FULL_H_DEF=8 ;;
    internvl35_4b) MID=OpenGVLab/InternVL3_5-4B;         REV=481f6e32467eab4e922ccd7fd6cf420441a62331; REPO_NAME=mpr-ivl;            FAM=ivl;  BATCH=1; BYTES=9465070800;  NFILES=2; LABELS="32,33,34,35"; TOKLOADER=internvl; ARCH_CHECK=none;      SETUPGB=10; EST="0.45-0.70 (6x6 queue assumption; not measured for this input)"; EST_H="2.4-3.7"; FULL_H_DEF=8 ;;
    *) echo "unknown model $KEY"; exit 2 ;;
  esac
  R="$W/$REPO_NAME"; CACHE_DIR="$R/.cache/huggingface/hub/models--${MID//\//--}"
  OUT="$R/followup_rel_6lang/results/rstem6_$KEY"
  fh="FULL_H_$KEY"; T_FULL=$(( ${!fh:-$FULL_H_DEF} * 3600 ))
  mf="MIN_FREE_GB_$KEY"; MINFREE="${!mf:-$(( BYTES / 1000000000 + 3 ))}"     # model bytes + 3 GB; setup extras are added separately (SETUPGB) when the repo does not exist
  return 0
}
for k in $ORDER; do ( minfo "$k" ) || exit 2; done
PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1
say() { local l; l="$(date -u +%FT%TZ) rstem6: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLAN
Plan (nothing is executed, nothing is changed). Models in this order: ${ORDER}
 0. refuse to start (no waiting) when another queue/overnight script [$OTHERS] runs, a compute process uses the GPU, or GPU memory used >= ${GPU_BUSY_MIB} MiB; flock against a second copy;
    network check (${NET_SECS}s ranged download from huggingface.co, and from download.pytorch.org when mpr-ivl must get a venv; STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1); RTX 4090 check;
    $ADDON sha256 = $ADDON_SHA (image tar, when present: $IMAGES_SHA)
 per model:
   setup (limit $((T_SETUP/60)) min): addon (only followup_rel_6lang/rstem6/ is allowed in the tar listing) unpacked into the repo + its SHA256SUMS; repo state (HEAD is $BASE_COMMIT or a descendant);
     mpr-ivl only (does not exist yet): git clone $REPO_URL @ $BRANCH -> checkout ${BASE_COMMIT:0:7} -> git am sixbysix_addon/ivl_patches/0001-0005 (patch-id checked) -> venv + pip install with
     requirements-qwen3-4090.txt and the constraints file internvl35_constraints_20261109.txt (sha ${CONSTRAINTS_SHA:0:8}); frozen conditions_6lang.jsonl (sha ${FROZEN_SHA:0:8}) and all images (4-lang assets, fr/ja assets)
     present, sha256 of ALL 948 image files checked against the rows (run_rstem6.py --check-images); env_check_rstem6.py: model config, system prompt, 12-13 inference-code hashes, package versions == earlier run contract
   model (limit $((T_MODEL/60)) min): disk check, hf download @ pinned revision (3 tries), exact safetensors byte sum, tokenizer A-D ids
   smoke: --max-items 2 = 240 inferences + gate (all rows ok, parse failures <= 3, batch shape, A-D mass, token ranges; label agreement >= 0.90 with the earlier REL O/R answers of the same model, and with the 4-language R_stem/R_opts run for qwen3_vl_8b / internvl35_8b)
   ETA check: smoke median x 18,960 must fit the full-step limit (IGNORE_ETA=1 overrides); full 18,960 inferences with --resume; completion gate; package (tar.gz + sha256 + per-file sha256); cache deleted when everything passed
$(for k in $ORDER; do ( minfo "$k"; echo " ($k) $MID @ ${REV:0:8} in $REPO_NAME, batch $BATCH, $(( BYTES / 1000000 )) MB, free disk needed ${MINFREE} GB (+${SETUPGB} GB for a new repo), full limit $((T_FULL/3600)) h, est. $EST s/inference -> $EST_H h" ); done)
 Estimates are from the earlier 4090 runs (median / mean s per inference), NOT measured on this server: inference about 18 h in total plus setup, downloads and model loads about 2 h.
 Failure rule: see the header of this script.
PLAN
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue_rstem6.lock"; flock -n 9 || { echo "another queue_rstem6 script is already running"; exit 1; }; }
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
  elif [ $rc -eq 3 ] && [[ "$name" == full ]]; then say "STOP step ${name}_$KEY: smoke ETA longer than the step limit (IGNORE_ETA=1 overrides)."; FAILED+=("${name}_${KEY}_eta")
  else say "step ${name}_$KEY FAILED (exit $rc), see $LOG/${name}_$KEY.log"; FAILED+=("${name}_$KEY"); fi
  if tail -n 200 "$LOG/${name}_$KEY.log" 2>/dev/null | grep -q -E "$CUDA_FATAL"; then ENV_FAIL=1; ENV_REASON="CUDA fatal error text in $LOG/${name}_$KEY.log"; fi
  return $rc
}

package() {  # package <KEY> <suffix>
  local k=$1 suf=$2 dest="$RES/$1" stage="$RES/_stage_$1" out
  minfo "$k"; mkdir -p "$dest"
  out="$dest/rstem6_${k}_results_${DAY}${suf}.tar.gz"
  rm -rf "$stage"; mkdir -p "$stage"
  if [ -d "$R" ]; then
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    [ -d "$OUT" ] && (cd "$R" && find "followup_rel_6lang/results/rstem6_$k" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true
  fi
  mkdir -p "$stage/queue_logs"; cp "$LOG/"*"_$k.log" "$stage/queue_logs/" 2>/dev/null || true; cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  if [ -d "$OUT" ]; then tar -czf "$out" -C "$R" "followup_rel_6lang/results/rstem6_$k" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && tar -tzf "$out" > /dev/null && (cd "$dest" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" && cp "$stage/results_files.sha256" "$dest/rstem6_${k}_${DAY}${suf}.files.sha256" 2>/dev/null; sha256sum "$(basename "$out")" >> "$RES/PACKAGES_SHA256.txt") \
    && say "$k packaged: $out ($(du -h "$out" | cut -f1)); sha256 $(cut -d' ' -f1 "$out.sha256")"
  local rc=$?; rm -rf "$stage"; return $rc
}

delete_cache() {  # exactly the HF cache folder of this model inside its repo; nothing else
  local d="$CACHE_DIR"
  case "$d" in "$W"/mpr-*/.cache/huggingface/hub/models--*) ;; *) say "refusing to delete unexpected path $d"; return 1 ;; esac
  [ -d "$d" ] && [ ! -L "$d" ] || return 0
  rm -rf "${d:?}"; say "$KEY model cache deleted: $d (free disk $(free_gb) GB)"
}

finish() {
  [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  local verdict="OK" k line="" v
  [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  for k in $ORDER; do v=$(getm "$k"); line="$line $k=${v:-not_started}"; done
  say "models:$line"
  ( cd "$RES" && tar -czf "queue_logs_${DAY}.tar.gz" logs STATUS 2>/dev/null && sha256sum "queue_logs_${DAY}.tar.gz" >> PACKAGES_SHA256.txt ) || true
  say "FINISHED $verdict. Packages: $RES/<KEY>/ and $RES/PACKAGES_SHA256.txt"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_RSTEM6B_DONE written, rerun to resume"; FINISHED=1; exit 130' INT TERM
abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }

rm -f "$DONE"
say "queue_rstem6_c started; models: $ORDER"
# ---- 0. nothing else may run --------------------------------------------------------------------
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
first=${ORDER%% *}; minfo "$first"; FIRST_MID=$MID; FIRST_REV=$REV
slow=0
hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$FIRST_MID/resolve/$FIRST_REV/model-00001-of-0000${NFILES}.safetensors}")
say "network check (${NET_SECS}s): huggingface.co ${hf} MB/s; minimum $NET_MIN_MBPS MB/s"
case "$hf" in ERR:4*) say "WARNING: test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;; *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
if echo " $ORDER " | grep -q -E ' (internvl35_8b|internvl25_8b) ' && [ ! -x "$W/mpr-ivl/.venv/bin/python" ]; then
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (mpr-ivl needs a new venv): download.pytorch.org ${pt} MB/s"
  case "$pt" in ERR:4*) say "WARNING: pytorch test URL answered HTTP ${pt#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;; *) awk -v v="$pt" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
fi
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

# ---- per-model functions (run inside `step`; variables come from the exported environment) ---------------------------------------------------------------------------
repo_env() { export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" PIP_CACHE_DIR="$R/.cache/pip" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PIP_NO_CACHE_DIR=1; }
check_images() { python followup_rel_6lang/rstem6/run_rstem6.py --model "$KEY" --model-config "followup_rel_6lang/rstem6/models/$CFGN" --check-images; }
prepare_ivl_repo() {   # mpr-ivl = BASE_COMMIT + patches 0001-0005 of the earlier InternVL runs (same procedure as queue_6x6.sh fn_setup ivl, without the 6x6 data)
  cd "$W"
  [ -d "$AD/ivl_patches" ] || { [ -s "$W/$SIX_ADDON" ] && echo "$SIX_SHA  $W/$SIX_ADDON" | sha256sum -c - && tar -xzf "$W/$SIX_ADDON" -C "$W"; }
  ( cd "$AD" && sha256sum -c SHA256SUMS )
  echo "$CONSTRAINTS_SHA  $AD/internvl35_constraints_20261109.txt" | sha256sum -c -
  local fresh=0 np n i c p1 p2; local -a P
  if [ ! -d "$R/.git" ]; then
    clone_repo() { [ -z "$(ls -A "$R" 2>/dev/null)" ] || { echo "refusing to clone: $R exists, is not empty and is not a git repository"; return 1; }; GIT_TERMINAL_PROMPT=0 git clone --branch "$BRANCH" "$REPO_URL" "$R"; }
    retry 5 clone_repo; fresh=1
  fi
  cd "$R"
  P=($(ls "$AD"/ivl_patches/000*.patch)); np=${#P[@]}; [ "$np" = 5 ]
  git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null || retry 3 git fetch origin "$BRANCH"
  [ "$fresh" = 1 ] && git checkout -q -f "$BASE_COMMIT"
  git merge-base --is-ancestor "$BASE_COMMIT" HEAD
  n=$(git rev-list --count "$BASE_COMMIT"..HEAD)
  if [ "$n" = 0 ]; then git am --abort 2>/dev/null || true; git -c user.name=local -c user.email=local@local am "${P[@]}"
  elif [ "$n" = "$np" ]; then
    i=0; for c in $(git rev-list --reverse "$BASE_COMMIT"..HEAD); do
      p1=$(git show "$c" | git patch-id --stable | cut -d' ' -f1); p2=$(git patch-id --stable < "${P[$i]}" | cut -d' ' -f1)
      [ "$p1" = "$p2" ] || { echo "commit $c differs from ${P[$i]} (patch-id)"; exit 1; }; i=$((i+1)); done
  else echo "repo is $n commits after $BASE_COMMIT, expected 0 or $np patch commits"; exit 1; fi
  if [ ! -x .venv/bin/python ]; then
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    [ -f .venv/bin/activate ] || { rm -rf .venv; python3 -m venv .venv || { apt-get update && apt-get install -y python3.12-venv && python3 -m venv .venv; }; }
  fi
  source .venv/bin/activate
  if [ ! -f .venv/.rstem6_installed ]; then
    local C="$AD/internvl35_constraints_20261109.txt"
    retry 3 python -m pip install -r 01_code/requirements-qwen3-4090.txt -c "$C"
    retry 3 python -m pip install -e './01_code[models]' -c "$C"
    python -m pip check || true
    touch .venv/.rstem6_installed
  fi
}
ensure_data() {   # frozen REL file + images (copies from sibling repos on this server first, then the earlier tools); the final check hashes every image
  local sib sub F="$FROZEN_REL"
  if [ ! -s "$F" ] || [ "$(sha256sum "$F" | cut -d' ' -f1)" != "$FROZEN_SHA" ]; then
    mkdir -p "$(dirname "$F")"
    for sib in "$W/mpr-minicpm" "$W/mpr-gui-crosslocale"; do
      [ "$sib" = "$R" ] && continue
      [ -s "$sib/$F" ] && [ "$(sha256sum "$sib/$F" | cut -d' ' -f1)" = "$FROZEN_SHA" ] && { cp "$sib/$F" "$F"; echo "frozen file copied from $sib"; break; }
    done
    if [ "$(sha256sum "$F" 2>/dev/null | cut -d' ' -f1)" != "$FROZEN_SHA" ] && [ -s "$W/$MC6_ADDON" ]; then
      echo "$MC6_SHA  $W/$MC6_ADDON" | sha256sum -c - && tar -xzf "$W/$MC6_ADDON" -O "$F" > "$F.tmp" && mv "$F.tmp" "$F"
    fi
    [ "$(sha256sum "$F" | cut -d' ' -f1)" = "$FROZEN_SHA" ]
  fi
  check_images > "$LOG/check_images_$KEY.log" 2>&1 && { grep -E '"(rows|images_checked|status)"' "$LOG/check_images_$KEY.log"; return 0; }
  echo "image check failed ($(tail -1 "$LOG/check_images_$KEY.log" | cut -c1-150)); copying missing images from sibling repos / tar"
  for sib in "$W/mpr-minicpm" "$W/mpr-gui-crosslocale"; do
    [ "$sib" = "$R" ] && continue
    for sub in followup_rel_4lang/assets followup_rel_6lang/assets; do [ -d "$sib/$sub" ] && { mkdir -p "$sub"; cp -R -n "$sib/$sub/." "$sub/"; }; done
  done
  if [ "$HAVE_IMAGES_TAR" = 1 ] && [ ! -d followup_rel_6lang/assets/images ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  check_images > "$LOG/check_images2_$KEY.log" 2>&1 && { grep -E '"(rows|images_checked|status)"' "$LOG/check_images2_$KEY.log"; return 0; }
  echo "still failing; fetching the 4-language images with the repository's data_bundle.py"
  python followup_rel_4lang/data_bundle.py unpack
  retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2
  check_images
}
setup() {
  if [ "$FAM" = ivl ]; then prepare_ivl_repo; fi      # no "|| exit": set -e must stay active inside the function
  cd "$R"; repo_env
  [ -d .git ] && git merge-base --is-ancestor "$BASE_COMMIT" HEAD
  tar -tzf "$W/$ADDON" > "$RES/addon_listing.txt"
  if grep -v -E '^followup_rel_6lang(/rstem6(/.*)?)?/?$' "$RES/addon_listing.txt" | grep -q .; then echo "addon contains paths outside followup_rel_6lang/rstem6:"; grep -v -E '^followup_rel_6lang(/rstem6(/.*)?)?/?$' "$RES/addon_listing.txt" | head; exit 1; fi
  tar -xzf "$W/$ADDON" -C "$R"
  ( cd followup_rel_6lang/rstem6 && sha256sum -c SHA256SUMS )
  source .venv/bin/activate
  ensure_data
  KEY="$KEY" CONFIG="followup_rel_6lang/rstem6/models/$CFGN" CONTRACT="followup_rel_6lang/rstem6/contracts/$CONTRACTN" PRE_MODEL=1 python followup_rel_6lang/rstem6/env_check_rstem6.py
  [ ! -f "$REPO_CFGP" ] || cmp "followup_rel_6lang/rstem6/models/$CFGN" "$REPO_CFGP"
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
assert len(fs) == int(os.environ["NFILES"]) and size == int(os.environ["BYTES"]), (len(fs), size)
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg.get("architectures"); llm = (cfg.get("llm_config") or {}).get("architectures")
print("architectures", arch, "| llm", llm, "| template", cfg.get("template"))
ac = os.environ["ARCH_CHECK"]
if ac == "internlm2": assert llm == ["InternLM2ForCausalLM"] and cfg.get("template") == "internvl2_5", (llm, cfg.get("template"))
if ac == "qwen25vl": assert arch == ["Qwen2_5_VLForConditionalGeneration"], arch
if ac == "qwen3vl": assert arch == ["Qwen3VLForConditionalGeneration"], arch
open(os.path.join(os.environ["RES"], "model_size_" + os.environ["KEY"] + ".txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\narchitectures={arch} llm={llm}\n")
tl = os.environ["TOKLOADER"]
if tl == "internvl":
    from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer
    tok = load_internvl_tokenizer(os.environ["MID"], revision=os.environ["REV"], trust_remote_code=True)
elif tl == "minicpm":
    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    AutoProcessor.from_pretrained(snap, trust_remote_code=True)
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
smoke()  { cd "$R"; repo_env; bash followup_rel_6lang/rstem6/run_rstem6.sh "$KEY" smoke; }
full()   { cd "$R"; repo_env; FULL_LIMIT_S="$T_FULL" bash followup_rel_6lang/rstem6/run_rstem6.sh "$KEY" full; }
export -f retry repo_env check_images prepare_ivl_repo ensure_data setup model smoke full
export W RES LOG BASE_COMMIT ADDON IMAGES_TAR SIX_ADDON SIX_SHA AD CONSTRAINTS_SHA REPO_URL BRANCH FROZEN_REL FROZEN_SHA MC6_ADDON MC6_SHA

smoke_report() {  # $1 gate json, $2 label
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
try: g = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
s, t, refs = g.get("sanity", {}), g.get("timing", {}), g.get("references") or {}
acc = {c: round(v[1] / v[0], 3) for c, v in (g.get("by_condition(n,correct)") or {}).items() if v[0]}
l = f"{sys.argv[2]}: acc by condition {acc} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')} prompt tokens {s.get('prompt_token_count')}"
for n, r in refs.items(): l += f" | reference {n}: label agreement {r.get('label_agreement')} on {r.get('common_rows')} rows, acc now {r.get('accuracy_now')} earlier {r.get('accuracy_earlier_run')}"
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
free_failed_caches() {  # make room for the next download by removing the kept cache(s) of earlier failed models of this queue (their own model cache folders only)
  local k
  for k in $ORDER; do
    [ "$k" = "$KEY" ] && break
    [ -f "$RES/$k/COMPLETE" ] && continue
    ( minfo "$k"; if [ -d "$CACHE_DIR" ]; then delete_cache; echo "a rerun of $k downloads it again" | tee -a "$ST" "$RES/STATUS"; fi )
  done
}

run_model() {
  minfo "$1"; step_failed=0; ENV_FAIL=0; ENV_REASON=""
  case "$KEY" in
    qwen3_vl_8b)   CFGN=qwen3_vl_8b_4090_channels_last.yaml; CONTRACTN=ref_qwen3_rel_20261107.json ;;
    minicpm_v45)   CFGN=minicpm_v4_5_8b_4090.yaml;           CONTRACTN=ref_minicpm_rel_20261116.json ;;
    internvl35_8b) CFGN=internvl3_5_8b_4090.yaml;            CONTRACTN=ref_internvl35_rel_20261109.json ;;
    uitars_15_7b)  CFGN=uitars_1_5_7b_4090.yaml;             CONTRACTN=ref_uitars_rel_20261114.json ;;
    internvl25_8b) CFGN=internvl2_5_8b.yaml;                 CONTRACTN=ref_internvl25_expA_20261114.json ;;
    qwen3_vl_4b)   CFGN=qwen3_vl_4b_4090_channels_last.yaml; CONTRACTN=ref_qwen3_4b_expA_20261117.json ;;
    internvl35_4b) CFGN=internvl3_5_4b_4090.yaml;            CONTRACTN=ref_internvl35_4b_expA_20261117.json ;;
  esac
  REPO_CFGP="01_code/configs/models/${CFGN}"
  export KEY MID REV R W RES LOG BYTES NFILES LABELS TOKLOADER ARCH_CHECK T_FULL FAM CFGN CONTRACTN REPO_CFGP BATCH HAVE_IMAGES_TAR
  if [ -f "$RES/$KEY/COMPLETE" ]; then say "$KEY already complete (package exists), skipped"; setm "$KEY" "complete_earlier"; return 0; fi
  if [ "$FAM" != ivl ] && ! { [ -d "$R/01_code" ] && [ -f "$R/.venv/bin/activate" ]; }; then say "STOP: $R (repo with 01_code and .venv) does not exist"; FAILED+=("repo_$KEY"); setm "$KEY" "STOPPED_repo"; ENV_FAIL=1; return 1; fi
  say "=== $KEY: $MID @ $REV (repo $REPO_NAME, batch $BATCH) ==="
  local f need; f=$(free_gb); need=$MINFREE; [ -d "$CACHE_DIR/snapshots/$REV" ] && [ -s "$CACHE_DIR/snapshots/$REV/config.json" ] && need=5
  [ ! -d "$R/.git" ] && need=$(( need + SETUPGB ))
  if [ "$f" -lt "$need" ]; then free_failed_caches; f=$(free_gb); fi
  if [ "$f" -lt "$need" ]; then say "STOP: only ${f} GB free (need $need GB for $KEY)"; FAILED+=("disk_$KEY"); setm "$KEY" "STOPPED_disk"; ENV_FAIL=1; ENV_REASON="disk"; return 1; fi
  if ! step setup "$T_SETUP" setup; then setm "$KEY" "FAILED_setup"; return 1; fi
  f=$(free_gb)
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then free_failed_caches; f=$(free_gb); fi
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ] && [ "$f" -lt "$MINFREE" ]; then say "STOP: only ${f} GB free (need $MINFREE GB for the $KEY download)"; FAILED+=("disk_$KEY"); setm "$KEY" "STOPPED_disk"; ENV_FAIL=1; ENV_REASON="disk"; return 1; fi
  if ! step model "$T_MODEL" model; then setm "$KEY" "FAILED_model"; return 1; fi
  if step smoke "$T_SMOKE" smoke; then say "smoke gate: $(smoke_report "$OUT/smoke/gate_smoke.json" "$KEY rstem6 smoke")"
  else say "$KEY smoke failed: $(smoke_report "$OUT/smoke/gate_smoke.json" "$KEY rstem6 smoke")see $LOG/smoke_$KEY.log; full run skipped"; setm "$KEY" "FAILED_smoke"; [ "$ENV_FAIL" = 1 ] && setm "$KEY" "STOPPED_env"; return 1; fi
  step full "$T_FULL" full
  say "complete gate: $(smoke_report "$OUT/full/gate_complete.json" "$KEY rstem6 full")"
  [ "$ENV_FAIL" = 1 ] && { setm "$KEY" "STOPPED_env"; return 1; }
  return $step_failed
}

for K in $ORDER; do
  run_model "$K"; rc=$?
  minfo "$K"
  [ "$(getm "$K")" = complete_earlier ] && continue
  suf=""; [ "$step_failed" = 1 ] || [ $rc -ne 0 ] && suf="_PARTIAL"
  if package "$K" "$suf"; then pk=1; else pk=0; say "$K packaging FAILED"; FAILED+=("package_$K"); fi
  if [ -z "$suf" ] && [ "$pk" = 1 ] && [ -z "$(getm "$K")" ]; then
    touch "$RES/$K/COMPLETE"; setm "$K" "ok"
    if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "$K cache kept (KEEP_MODEL_CACHE=1)"; else delete_cache; fi
  else
    [ -z "$(getm "$K")" ] && setm "$K" "FAILED_steps"
    [ -f "$RES/$K/COMPLETE" ] || { [ -d "$CACHE_DIR" ] && say "$K cache KEPT (not finished; free $(free_gb) GB); it is removed automatically if the next download needs the room"; }
  fi
  if [ "$ENV_FAIL" = 1 ]; then say "STOP QUEUE (environmental failure: ${ENV_REASON:-see step log}); remaining models are not started"; break; fi
  if [ $rc -ne 0 ] || [ -n "$suf" ]; then
    if ! env_recheck; then say "STOP QUEUE (environment re-check after the failed $K: $ENV_REASON); remaining models are not started"; FAILED+=("env_after_$K"); break; fi
    say "$K failed with a model-specific problem; environment re-check ok; continuing with the next model"
  fi
done
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
