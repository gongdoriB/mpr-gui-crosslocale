#!/usr/bin/env bash
# MiniCPM-V-4.5 complete 6-language results on the fresh-ish Vast.ai RTX 4090 "vast4090c" (nothing else may run on the GPU):
#   0 setup /workspace/mpr-minicpm (clone, commit 6664400 + MiniCPM adapter patch, venv with the pins of the earlier MiniCPM run, data, 6-language addon, 6x6 data), pinned model
#   1 smoke REL + Exp A (+ gates)                  2 consistency check (600 earlier 4-language REL inferences, report only, never stops the queue)
#   3 REL O/R/C/F new_only (26,057 inferences / 29,148 rows)
#   4 Exp A new pairs, main 2,844 pairs x O/T/R = 8,532 inferences + T_retry 148 pairs x 3 = 444
#   5 6x6 full benchmark (79,020 unique inferences) = unchanged queue_6x6.sh with MODELS=minicpm_v45 (batch 1, english_fixed, score-labels, seed 42, wf wi au ap ael rel)
# Results are packaged to /workspace/queue_results/minicpm_6lang/ after EACH major step (sha256 + per-file sha list).
# Status: /workspace/QUEUE_C2_STATUS   Done marker: /workspace/QUEUE_C2_DONE ("OK", "OK (non-fatal: ...)" or "FAILED steps: ...")
# Usage (tmux, in /workspace, all upload files next to this script):
#   bash queue_minicpm_6lang.sh --plan     # prints the plan, changes nothing
#   DELETE_QWEN3_8B_CACHE=1 bash queue_minicpm_6lang.sh
# Environment: DELETE_QWEN3_8B_CACHE=1 (deletes exactly /workspace/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct, nothing else),
#   FORCE_SLOW_NET=1, IGNORE_ETA=1, NET_MIN_MBPS (20), MIN_FREE_GB (44; lowered automatically for parts already on disk), SKIP_CONSISTENCY=1, SKIP_6X6=1,
#   NO_CONSTRAINTS=1 (pip without the constraints taken from the earlier freeze), REL_H (7), EXPA_H (3), CC_H (1), KEEP_MODEL_CACHE (default 1 for the queue_6x6.sh call: the model cache is NOT deleted after a successful 6x6; set KEEP_MODEL_CACHE=0 to let queue_6x6.sh delete it).
# Failure rule: preflight/setup_env/model failures stop everything that needs them. After that REL, Exp A, consistency and 6x6 are independent: a failed one does not block the others
#   (smoke_rel gates consistency+REL, smoke_expa gates Exp A full+retry, setup_6lang gates consistency/REL/Exp A, setup_6x6 gates 6x6).
# Rerunning resumes (repo, venv, predictions are reused). The instance is NOT stopped. Server text files (AGENTS.md / CLAUDE.md) are not read.
set -uo pipefail

W="${W:-/workspace}"
REPO_URL=https://github.com/dltmdwoOS/mpr-gui-crosslocale.git
BRANCH=followup-rel-4lang-inference
COMMIT=666440011f538cd9336794c9965429de640843c3
MODEL_ID=openbmb/MiniCPM-V-4_5
REVISION=14a2004908140699317e2109e90573c1e6323c36
MODEL_BYTES=17391890184
MC_ADDON=minicpm_addon_20261115.tar.gz;           MC_ADDON_SHA=e638bb79e9795f15ef3a2759f2efe12c1a6a313a40018e4405fe8221620fb150
ADDON6=minicpm_6lang_addon_20261117.tar.gz;       ADDON6_SHA=dbf726db2f9165e5a91d22058959453ddbab1dc2e9dde17fd09b4ad9b53bef05
SIX_ADDON=sixbysix_addon_20261114.tar.gz;         SIX_ADDON_SHA=ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a
SIX_QUEUE=queue_6x6.sh;                           SIX_QUEUE_SHA=d538b0cd799a653d1cc536716e953168db86a15c3c346cbe659bd41a833a805a
IMAGES_TAR=internvl25_rel6_frja_images_20261116.tar; IMAGES_SHA=5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754
MIN_FREE_GB="${MIN_FREE_GB:-44}"; NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}" EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"
T_SETUP=$((90*60)); T_S6=$((30*60)); T_S66=$((3*3600)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_CC=$(( ${CC_H:-1} * 3600 ))
T_REL=$(( ${REL_H:-7} * 3600 )); T_EXPA=$(( ${EXPA_H:-3} * 3600 )); T_RETRY=$((60*60))

R="$W/mpr-minicpm"; RES="$W/queue_results/minicpm_6lang"; LOG="$RES/logs"
ST="$W/QUEUE_C2_STATUS"; DONE="$W/QUEUE_C2_DONE"
RDIR="$R/followup_rel_6lang/results"
OUT_REL="$RDIR/rel6new_minicpm_v45"; OUT_CC="$RDIR/consistency_minicpm_v45"; OUT_EXA="$RDIR/expA6_minicpm_v45"
CACHE_DIR="$R/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
QWEN_CACHE="$W/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
DAY=$(date -u +%Y%m%d)
FAILED=(); NONFATAL=(); NOTRUN=(); FINISHED=0
PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1
say() { local l; l="$(date -u +%FT%TZ) minicpm_6lang: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLANEOF
Plan (nothing is executed, nothing is changed):
 0. refuse to start if a GPU compute process exists or GPU memory used >= ${GPU_BUSY_MIB} MiB, or another queue script runs. No waiting. Optional (DELETE_QWEN3_8B_CACHE=1, done only after all other prechecks passed, right before the disk check): delete exactly
    $QWEN_CACHE. Network check (STOP below ${NET_MIN_MBPS} MB/s from huggingface.co and, while torch is not installed, download.pytorch.org; FORCE_SLOW_NET=1 overrides).
 1. checks: sha256 of $MC_ADDON, $ADDON6, $SIX_ADDON, $SIX_QUEUE (and $IMAGES_TAR if present); python 3.12; RTX 4090; free disk >= ${MIN_FREE_GB} GB
    (model 17.4 + venv ~9 + repo/4-language images ~2 + fr/ja images 0.75 + 6x6 data 2.6 + results ~1.5 + margin; less when parts already exist)
 2. setup_env (limit $((T_SETUP/60)) min): clone $REPO_URL ($BRANCH) to $R at ${COMMIT:0:7}; unpack $MC_ADDON; git am adapter patch (HEAD = base + 1);
    venv + pip with the earlier requirements (torch 2.9.1+cu130, transformers 4.57.6 ...) AND constraints = the earlier MiniCPM pip freeze; key packages compared with that freeze (mismatch = STOP); 4-language data
    setup_6lang (limit $((T_S6/60)) min): 6-language addon (listing checked, own SHA256SUMS), fr/ja images from the tar (or huggingface), data verify, config/system prompt/13 code hashes vs the earlier run contracts
    setup_6x6 (limit $((T_S66/3600)) h): sixbysix addon + data of queue_6x6.sh (reuses raw data of /workspace/mpr-gui-crosslocale when complete, else downloads 2.6 GB), manifest/plan-id check
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION, 4 shards = $MODEL_BYTES bytes, tokenizer A-D ids 32..35
 4. smoke REL (--max-items 2, new_only, 112 inferences) + gate; smoke Exp A (--max-items 3, 162 inferences) + gate (each limit $((T_SMOKE/60)) min)
 5. consistency 600 inferences (seed 20261116, batch 1, limit $((T_CC/3600)) h; report only) -> $OUT_CC/consistency_agreement.json; package 01
 6. REL new_only: 26,057 inferences / 29,148 rows, batch 1, strict_batch_size false (limit $((T_REL/3600)) h, ETA check) + gate; package 02
 7. Exp A new pairs main: 2,844 pairs x 3 = 8,532 (limit $((T_EXPA/3600)) h) + gate (O/R vs REL run); T_retry 148 pairs x 3 = 444 (limit $((T_RETRY/60)) min); package 03
 8. 6x6: MODELS=minicpm_v45 bash $SIX_QUEUE (79,020 unique inferences, batch 1, smoke 216 + gate, full limit 20 h, ETA check); package 04 (links its tar.gz)
 9. final package 00_final, append to $ST, write $DONE.
 Estimated time from the earlier MiniCPM run (mean 0.512 s REL, 0.504 s Exp A, batch 1; not measured on this server): REL 26,057 -> 3.7 h, Exp A 8,976 -> 1.3 h,
 consistency 600 -> 5 min, 6x6 79,020 x 0.5 s -> about 11 h (assumed, 6x6 never measured for MiniCPM); setup+model+smokes 1-1.5 h. Total about 17 h.
PLANEOF
  exit 0
fi

mkdir -p "$LOG" "$RES/tools"
command -v flock >/dev/null && { exec 9> "$W/.queue_c2_minicpm.lock"; flock -n 9 || { echo "another queue_minicpm_6lang.sh is already running"; exit 1; }; }

retry() {
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n failed: $*" >&2; [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i )); done
  return 1
}
step() {  # step <name> <limit_s> <function> [nonfatal]
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

package() {  # package <label> <result dir under results/ ...>: tar.gz + .sha256 + per-file sha list (inside and next to it)
  local label=$1; shift
  local out="$RES/minicpm_6lang_${label}_${DAY}.tar.gz" stage="$RES/_stage_$label" dirs=() d
  rm -rf "$stage"; mkdir -p "$stage"
  for d in "$@"; do if [ -d "$RDIR/$d" ]; then dirs+=("followup_rel_6lang/results/$d"); elif [ -d "$R/$d" ]; then dirs+=("$d"); fi; done   # a name with a slash = repo-relative path
  if [ -d "$R" ]; then
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -3 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    [ ${#dirs[@]} -gt 0 ] && (cd "$R" && find "${dirs[@]}" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true
  fi
  cp "$RES"/freeze_compare.txt "$RES"/net_check.txt "$RES"/model_size.txt "$stage/" 2>/dev/null || true
  cp -r "$LOG" "$stage/queue_logs" 2>/dev/null || true
  cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$stage/disk.txt" 2>&1 || true
  if [ ${#dirs[@]} -gt 0 ]; then tar -czf "$out" -C "$R" "${dirs[@]}" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && (cd "$RES" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" && sha256sum "$(basename "$out")" >> PACKAGES_SHA256.txt) \
    && { [ ! -s "$stage/results_files.sha256" ] || cp "$stage/results_files.sha256" "$RES/minicpm_6lang_${label}_${DAY}.files.sha256"; }
  local rc=$?
  [ $rc -eq 0 ] && say "packaged $out ($(du -h "$out" | cut -f1))" || { say "packaging $label FAILED"; NONFATAL+=("package_$label"); }
  rm -rf "$stage"; return 0
}

finish() {
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  package 00_final
  local verdict="OK"
  [ ${#NONFATAL[@]} -gt 0 ] && verdict="OK (non-fatal: ${NONFATAL[*]} failed)"
  if [ ${#FAILED[@]} -gt 0 ]; then verdict="FAILED steps: ${FAILED[*]}"; [ ${#NOTRUN[@]} -gt 0 ] && verdict="$verdict | not run: ${NOTRUN[*]}"; fi
  say "FINISHED $verdict. Results: $RES/"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_C2_DONE written, rerun the script to resume"; FINISHED=1; exit 130' INT TERM
abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }

# ---- 0. GPU free, no other queue, optional Qwen cache deletion, network -----------------------------
rm -f "$DONE"
say "queue_minicpm_6lang started"
if pgrep -f 'queue_6x6\.sh|queue_qwen3_8b_frja\.sh|queue_minicpm\.sh|run_all_overnight\.sh|queue_sizes_uitars\.sh' >/dev/null 2>&1; then abort others "another queue script is running"; fi
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -i 'no running' | sed '/^$/d')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
[ -z "$used" ] && abort gpu "nvidia-smi not available"
if [ -n "$apps" ] || [ "$used" -ge "$GPU_BUSY_MIB" ]; then
  abort gpu "the GPU is in use (${used} MiB used; processes: $(echo "$apps" | head -3 | tr '\n' ';')). Nothing was started."
fi
say "GPU idle (${used} MiB used, no compute process)"
net_speed() {
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}
net_check() {
  local hf pt slow=0 need_pt=0
  grep -q "2.9.1+cu130" "$R"/.venv/lib/python3*/site-packages/torch/version.py 2>/dev/null || need_pt=1
  hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$MODEL_ID/resolve/$REVISION/model-00001-of-00004.safetensors}")
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${hf} MB/s, download.pytorch.org ${pt} MB/s (torch download needed: $([ $need_pt = 1 ] && echo yes || echo no)); minimum $NET_MIN_MBPS MB/s"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$hf" "$pt" > "$RES/net_check.txt"
  case "$hf" in ERR:4*) say "WARNING: the huggingface test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;;
    *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
  if [ "$need_pt" = 1 ]; then
    case "$pt" in ERR:4*) say "WARNING: the pytorch test URL answered HTTP ${pt#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;;
      *) awk -v v="$pt" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
  fi
  if [ "$slow" = 1 ]; then
    [ "${FORCE_SLOW_NET:-0}" = 1 ] && { say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; }
    abort network "network too slow or unreachable (need >= $NET_MIN_MBPS MB/s). Nothing was downloaded. Use another instance, or FORCE_SLOW_NET=1"
  fi
}
net_check

# ---- 1. checks ----------------------------------------------------------------------------------
for pair in "$MC_ADDON:$MC_ADDON_SHA" "$ADDON6:$ADDON6_SHA" "$SIX_ADDON:$SIX_ADDON_SHA" "$SIX_QUEUE:$SIX_QUEUE_SHA"; do
  f=${pair%%:*}; s=${pair##*:}
  [ -s "$W/$f" ] || abort addon "$f missing in $W"
  echo "$s  $W/$f" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$f sha256 differs"
done
if [ -s "$W/$IMAGES_TAR" ]; then
  echo "$IMAGES_SHA  $W/$IMAGES_TAR" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$IMAGES_TAR sha256 differs (delete it to fetch the images from huggingface.co instead)"
  HAVE_IMAGES_TAR=1
else HAVE_IMAGES_TAR=0; say "no $IMAGES_TAR in $W: fr/ja screenshots will be downloaded from huggingface.co and sha256-checked"; fi
export HAVE_IMAGES_TAR
tar -xzf "$W/$ADDON6" -O followup_rel_6lang/mc6/reference/requirements.freeze.earlier_minicpm_20261116.txt > "$RES/freeze_earlier.txt" 2>>"$LOG/addon_sha.log" || abort addon "cannot read the reference freeze from $ADDON6"
tar -xzf "$W/$ADDON6" -O followup_rel_6lang/mc6/freeze_compare.py > "$RES/tools/freeze_compare.py" 2>>"$LOG/addon_sha.log" || abort addon "cannot read freeze_compare.py from $ADDON6"
grep -v '^-e ' "$RES/freeze_earlier.txt" > "$RES/freeze_constraints.txt"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>> "$LOG/python.log" || abort python "python3 is not 3.12 ($(python3 --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort gpu "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
# Qwen3-8B cache deletion (only when asked, only after every other precheck above passed; the disk check below counts the freed space)
if [ "${DELETE_QWEN3_8B_CACHE:-0}" = 1 ]; then
  if [ -d "$QWEN_CACHE" ] && [ ! -L "$QWEN_CACHE" ] && [ "$(basename "$QWEN_CACHE")" = "models--Qwen--Qwen3-VL-8B-Instruct" ]; then
    rm -rf "$QWEN_CACHE"; say "DELETE_QWEN3_8B_CACHE=1: deleted $QWEN_CACHE; free disk now $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
  else say "DELETE_QWEN3_8B_CACHE=1 but $QWEN_CACHE does not exist (or is a symlink): nothing deleted"; fi
fi
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
df -h "$W" > "$LOG/disk_start.log"
NEED=$MIN_FREE_GB
[ -s "$CACHE_DIR/snapshots/$REVISION/config.json" ] && NEED=$((NEED - 18))
[ -f "$R/.venv/.minicpm_installed" ] && NEED=$((NEED - 9))
[ "$NEED" -lt 12 ] && NEED=12
[ "$FREE_GB" -ge "$NEED" ] || abort disk "only ${FREE_GB} GB free in $W (need ${NEED}+; DELETE_QWEN3_8B_CACHE=1 frees about 17 GB)"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free (need ${NEED})"

# ---- step functions (run in their own bash via export -f) -------------------------------------------
renv() { export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" PIP_CACHE_DIR="$R/.cache/pip" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PIP_NO_CACHE_DIR=1; }
setup_env() {
  cd "$W"; renv
  clone_repo() { rm -rf mpr-minicpm; git clone --branch "$BRANCH" "$REPO_URL" mpr-minicpm; }
  [ -d mpr-minicpm/.git ] || retry 5 clone_repo
  cd mpr-minicpm
  if [ ! -f .git/minicpm_patched ]; then
    git am --abort 2>/dev/null || true
    git checkout -f "$COMMIT"
    tar -xzf "$W/$MC_ADDON"
    sha256sum -c followup_rel_4lang/minicpm/ADDON_SHA256SUMS
    git -c user.name=local -c user.email=local@local am followup_rel_4lang/minicpm_patches/0001-*.patch
    touch .git/minicpm_patched
  else tar -xzf "$W/$MC_ADDON"; sha256sum -c followup_rel_4lang/minicpm/ADDON_SHA256SUMS; fi
  [ "$(git rev-list --count "$COMMIT"..HEAD)" = 1 ]
  git log --oneline -3
  if [ ! -f .venv/bin/activate ]; then
    rm -rf .venv
    python3 -m venv .venv || { apt-get update && apt-get install -y python3.12-venv && python3 -m venv .venv; }
  fi
  source .venv/bin/activate
  if [ ! -f .venv/.minicpm_installed ]; then
    install_pins() { python -m pip install -r 01_code/requirements-qwen3-4090.txt "$@" && python -m pip install -e './01_code[models]' "$@"; }
    if [ "${NO_CONSTRAINTS:-0}" = 1 ]; then
      retry 3 install_pins; echo "unconstrained (NO_CONSTRAINTS=1)" > "$RES/pip_path.txt"
    elif retry 3 install_pins -c "$RES/freeze_constraints.txt"; then echo "constrained by the earlier freeze" > "$RES/pip_path.txt"
    else
      echo "constrained install failed 3 times; one more try without constraints (freeze_compare below decides)"
      retry 1 install_pins; echo "unconstrained (constrained install failed)" > "$RES/pip_path.txt"
    fi
    python -m pip check
    touch .venv/.minicpm_installed
  fi
  python --version
  v=$(python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)'); echo "torch/transformers: $v"
  [ "$v" = "$EXPECT_TORCH 4.57.6" ]
  python -m pip freeze > "$RES/requirements.freeze.new.txt"
  python "$RES/tools/freeze_compare.py" "$RES/freeze_earlier.txt" "$RES/requirements.freeze.new.txt" "$RES/freeze_compare.txt"
  python followup_rel_4lang/data_bundle.py unpack
  retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2
  python followup_rel_4lang/data_bundle.py verify
}
setup_6lang() {
  cd "$R"; renv; source .venv/bin/activate
  tar -tzf "$W/$ADDON6" > "$RES/addon6_listing.txt"
  if grep -v -E '^(01_code(/src(/mpr_crosslocale(/(data|inference))?)?)?/?|01_code/src/mpr_crosslocale/(data/rel_followup_6lang|inference/run_rel_followup_6lang)\.py|followup_rel_6lang(/.*)?)$' "$RES/addon6_listing.txt" | grep -q .; then
    echo "addon contains paths outside its own files"; exit 1; fi
  tar -xzf "$W/$ADDON6"
  ( cd followup_rel_6lang/mc6 && sha256sum -c SHA256SUMS )
  if [ "$HAVE_IMAGES_TAR" = 1 ]; then
    tar -tf "$W/$IMAGES_TAR" > "$RES/images_listing.txt"
    if grep -v -E '^followup_rel_6lang(/assets(/images(/[0-9]+(/(fr|ja)(/[^/]+)?)?)?)?)?/?$' "$RES/images_listing.txt" | grep -q .; then echo "image tar contains unexpected paths"; exit 1; fi
    tar -xf "$W/$IMAGES_TAR"
  fi
  python -c 'import mpr_crosslocale.data.rel_followup_6lang, mpr_crosslocale.inference.run_rel_followup_6lang; print("6-language modules import ok")'
  python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang > "$LOG/bundle6_verify_first.log" 2>&1 \
    || { echo "6-language verify failed; fetching the fr/ja screenshots"; retry 6 python followup_rel_6lang/data_bundle_6lang.py fetch-images --data-root followup_rel_6lang --workers 2; python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang; }
  X=followup_rel_6lang/mc6
  KEY=minicpm_v45 CONFIG=$X/models/minicpm_v4_5_8b_4090.yaml REL_REF=$X/contracts/ref_minicpm_rel_20261116.json EXPA_REF=$X/contracts/ref_minicpm_expA_20261116.json CODE_REF=$X/contracts/ref_minicpm_rel_20261116.json B2_PRE_MODEL=1 python $X/env_check.py
}
setup_6x6() {
  cd "$W"
  tar -xzf "$W/$SIX_ADDON" -C "$W"; ( cd "$W/sixbysix_addon" && sha256sum -c SHA256SUMS )
  W="$W" bash "$W/$SIX_QUEUE" --fn fn_setup mc
}
model() {
  cd "$R"; renv; source .venv/bin/activate
  retry 3 hf download "$MODEL_ID" --revision "$REVISION"
  MODEL_ID="$MODEL_ID" REVISION="$REVISION" MODEL_BYTES="$MODEL_BYTES" RES="$RES" python - <<'PY'
import glob, json, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MODEL_ID"], revision=os.environ["REVISION"], local_files_only=True)
fs = glob.glob(os.path.join(snap, "model-*.safetensors")); size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) == 4 and size == int(os.environ["MODEL_BYTES"]), size
open(os.path.join(os.environ["RES"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\n")
from transformers import AutoProcessor, AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
AutoProcessor.from_pretrained(snap, trust_remote_code=True)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
assert ids == {"A": [32], "B": [33], "C": [34], "D": [35]}, ids
PY
  du -sh "$HF_HOME"
}
# 6x6 smoke (216 inferences + gate) right after the model step; queue_6x6.sh later reuses the passed gate_6x6_smoke.json ("smoke already passed")
smoke_6x6()   { cd "$W"; W="$W" KEY=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-1}" bash "$W/$SIX_QUEUE" --fn fn_smoke; }
smoke_rel()   { cd "$R"; renv; bash followup_rel_6lang/mc6/run_rel6_mc.sh smoke; }
smoke_expa()  { cd "$R"; renv; bash followup_rel_6lang/mc6/run_expA6_mc.sh smoke; }
consistency() { cd "$R"; renv; bash followup_rel_6lang/mc6/run_consistency_mc.sh run; }
full_rel()    { cd "$R"; renv; REL_LIMIT_S="$T_REL" bash followup_rel_6lang/mc6/run_rel6_mc.sh full; }
full_expa()   { cd "$R"; renv; EXPA_LIMIT_S="$T_EXPA" bash followup_rel_6lang/mc6/run_expA6_mc.sh full; }
retry_expa()  { cd "$R"; renv; bash followup_rel_6lang/mc6/run_expA6_mc.sh retry; }
export -f retry renv setup_env setup_6lang setup_6x6 model smoke_6x6 smoke_rel smoke_expa consistency full_rel full_expa retry_expa
export W R RES LOG BRANCH REPO_URL COMMIT MC_ADDON ADDON6 SIX_ADDON SIX_QUEUE IMAGES_TAR MODEL_ID REVISION MODEL_BYTES T_REL T_EXPA

smoke_report() {
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
try: g = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
line = f"{sys.argv[2]}: acc {s.get('accuracy')} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')}"
if g.get("warnings"): line += " WARN " + "; ".join(g["warnings"])
if g.get("problems"): line += " PROBLEMS " + "; ".join(g["problems"])
print(line, end=" | ")
PY
}

# ---- 2. run ---------------------------------------------------------------------------------------
setup_env_ok=0; setup6_ok=0; setup66_ok=0; smoke66_ok=0; rel_ok=0; expa_ok=0; model_ok=0
if step setup_env "$T_SETUP" setup_env; then
  setup_env_ok=1; say "pip install path: $(cat "$RES/pip_path.txt" 2>/dev/null || echo 'venv already existed')"; say "freeze check: $(head -3 "$RES/freeze_compare.txt" 2>/dev/null | tr '\n' ' ')"
  step setup_6lang "$T_S6" setup_6lang && setup6_ok=1
  [ "${SKIP_6X6:-0}" = 1 ] || { step setup_6x6 "$T_S66" setup_6x6 && setup66_ok=1; }
  if step model "$T_MODEL" model; then
    model_ok=1
    # first real run_cross_locale + MiniCPM GPU execution, early: 6x6 smoke (216 inferences + gate). A failure blocks only the 6x6 main step.
    if [ "${SKIP_6X6:-0}" != 1 ] && [ "$setup66_ok" = 1 ]; then
      if step smoke_6x6 "$T_SMOKE" smoke_6x6; then smoke66_ok=1; say "6x6 smoke gate: $(python3 -c "import json;g=json.load(open('$R/followup_rel_4lang/results/6x6_minicpm_v45/gate_6x6_smoke.json'));print(g.get('status'),'acc',g.get('sanity',{}).get('accuracy'),'ETA full h',g.get('eta_hours_full_at_median'),g.get('problems'))" 2>/dev/null)"
      else say "6x6 smoke FAILED: the 6x6 main step will not run; REL/Exp A/consistency are not affected"; fi
    fi
  fi
else
  NOTRUN+=(setup_6lang setup_6x6 model smoke_rel smoke_expa consistency full_rel full_expa retry_expa 6x6)
fi
if [ "$model_ok" = 1 ]; then
  if [ "$setup6_ok" = 1 ]; then
    if step smoke_rel "$T_SMOKE" smoke_rel; then rel_ok=1; say "smoke gate: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "rel fr/ja")"
    else say "REL smoke failed: $(smoke_report "$OUT_REL/smoke/gate_rel_smoke.json" "rel fr/ja")see $LOG/smoke_rel.log; consistency and REL full skipped"; NOTRUN+=(consistency full_rel); fi
    if step smoke_expa "$T_SMOKE" smoke_expa; then expa_ok=1; say "smoke gate: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "expA fr/ja")"
    else say "Exp A smoke failed: $(smoke_report "$OUT_EXA/smoke/gate_smoke.json" "expA fr/ja")see $LOG/smoke_expa.log; Exp A full and retry skipped"; NOTRUN+=(full_expa retry_expa); fi
    if [ "$rel_ok" = 1 ]; then
      if [ "${SKIP_CONSISTENCY:-0}" = 1 ]; then say "consistency check skipped (SKIP_CONSISTENCY=1)"
      else step consistency "$T_CC" consistency nonfatal; say "consistency: $(python3 -c "import json;d=json.load(open('$OUT_CC/consistency_agreement.json'))['overall'];print('label agreement',d['label_agreement'],'accuracy old',d['old_accuracy'],'new',d['new_accuracy'])" 2>/dev/null)"; fi
    fi
    package 01_setup_smoke_consistency rel6new_minicpm_v45 expA6_minicpm_v45 consistency_minicpm_v45 followup_rel_4lang/results/6x6_minicpm_v45
    if [ "$rel_ok" = 1 ]; then step full_rel "$T_REL" full_rel || true; package 02_rel_new_only rel6new_minicpm_v45 consistency_minicpm_v45; fi
    if [ "$expa_ok" = 1 ]; then step full_expa "$T_EXPA" full_expa && { step retry_expa "$T_RETRY" retry_expa || true; } || NOTRUN+=(retry_expa); package 03_expA_new_pairs expA6_minicpm_v45; fi
  else NOTRUN+=(smoke_rel smoke_expa consistency full_rel full_expa retry_expa); fi
  if [ "${SKIP_6X6:-0}" = 1 ]; then say "6x6 skipped (SKIP_6X6=1)"
  elif [ "$setup66_ok" = 1 ] && [ "$smoke66_ok" != 1 ]; then NOTRUN+=(6x6); say "6x6 main step not run: smoke_6x6 failed"
  elif [ "$setup66_ok" = 1 ]; then
    say "step 6x6 started (queue_6x6.sh MODELS=minicpm_v45; its own limits: full 20 h)"
    ( cd "$W" && W="$W" MODELS=minicpm_v45 IGNORE_OTHER_QUEUES=1 FORCE_SLOW_NET=1 KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-1}" bash "$W/$SIX_QUEUE" 2>&1 | while IFS= read -r l; do echo "$l" | sed 's/^/[queue_6x6] /' >> "$ST"; done ) >> "$LOG/6x6.log" 2>&1
    if [ -f "$W/queue_results/6x6_minicpm_v45/COMPLETE" ]; then say "step 6x6 ok"
    else say "step 6x6 NOT COMPLETE (see $ST lines [queue_6x6] and $W/queue_logs6/)"; FAILED+=(6x6); fi
    mkdir -p "$RES/6x6"; for f in "$W"/queue_results/6x6_minicpm_v45/*.tar.gz "$W"/queue_results/6x6_minicpm_v45/SHA256SUMS; do [ -e "$f" ] && { ln -f "$f" "$RES/6x6/" 2>/dev/null || cp "$f" "$RES/6x6/"; }; done
    O6="$R/followup_rel_4lang/results/6x6_minicpm_v45"
    [ -d "$O6" ] && (cd "$(dirname "$O6")" && find "$(basename "$O6")" -type f -exec sha256sum {} + > "$RES/minicpm_6lang_04_6x6_${DAY}.files.sha256") 2>/dev/null
    (cd "$RES/6x6" 2>/dev/null && sha256sum ./*.tar.gz > "$RES/minicpm_6lang_04_6x6_${DAY}.tarlist.sha256") 2>/dev/null
    say "6x6 package(s) linked in $RES/6x6/ with sha list $RES/minicpm_6lang_04_6x6_${DAY}.files.sha256"
  else NOTRUN+=(6x6); say "6x6 not run: setup_6x6 failed"; fi
else
  [ "$setup_env_ok" = 1 ] && NOTRUN+=(smoke_rel smoke_expa consistency full_rel full_expa retry_expa 6x6)
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
