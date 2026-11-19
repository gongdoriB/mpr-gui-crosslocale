#!/usr/bin/env bash
# Full 6x6 cross-locale benchmark (79,020 unique inferences per model) for additional models,
# on ONE Vast.ai RTX 4090 (24 GB), batch size 1: --dimensions wf wi au ap ael rel --sample-unit semantic_items --seed 42
#  --batch-size 1 --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --resume.
# Same call as the InternVL3.5-8B run (batch 1). The Qwen3-VL-8B 6x6 run used batch 4 (batch_size is 4 in all 79,020 rows of its
# full_6x6.jsonl), so Qwen3 sizes differ from their 8B reference in batch size.
#
# Per model: pinned download -> dry-run plan check -> 6x6 smoke (--sample-size 6, 216 inferences) + gate
#   -> full 6x6 + gate -> summary -> package to /workspace/queue_results/6x6_<key>/ -> delete model cache -> next model.
# A failed model does not stop the queue. Works on a fresh instance (clone, patch, venv, data, manifest per family) and on an
# instance that already has /workspace/mpr-gui-crosslocale, /workspace/mpr-ivl, /workspace/mpr-minicpm (reused when their
# commit/patch state is right).
#
# Usage (inside tmux, in /workspace, with sixbysix_addon_20261114.tar.gz and this script there):
#   bash queue_6x6.sh                 # all 7 models in the default order
#   bash queue_6x6.sh --plan          # print the plan, change nothing
#   bash queue_6x6.sh --net           # only the network speed check
# Environment:
#   MODELS="qwen3_vl_4b internvl35_4b"   order/subset (default: qwen3_vl_4b internvl35_4b uitars_15_7b minicpm_v45 qwen3_vl_2b internvl35_2b internvl35_1b)
#   WAIT_FOR=/workspace/MINICPM_DONE     wait until this file EXISTS (any content, also "FAILED steps: ..." or "STOPPED rc=1") before starting
#                                        (default: do not wait; WAIT_FOR_MAX_H=72). While run_all_overnight.sh still runs, it keeps waiting
#                                        (a queue that STOPped early writes its DONE file although the overnight job is still on the GPU).
#   FORCE_SLOW_NET=1                     start although the network check is below NET_MIN_MBPS (default 20)
#   IGNORE_ETA=1                         start the full run although the smoke timing says it cannot finish within its timeout
#   KEEP_MODEL_CACHE=1                   never delete the model cache
#   Cache rule: a model that finished everything loses its cache. After a failure or timeout the cache is KEPT (a rerun resumes without
#   downloading; hf download resumes partial files) as long as the free disk still covers the largest MINFREE_GB of the models that come
#   next in MODELS (after the last model nothing is needed); otherwise it is deleted.
#   IGNORE_OTHER_QUEUES=1                do not stop when another queue_*.sh / run_all_overnight.sh is running
# Progress: tail /workspace/QUEUE6I_STATUS   Logs: /workspace/queue_logs6i/   Finished: /workspace/QUEUE6I_DONE exists.
# Rerunning resumes (finished stages are skipped). The instance is NOT stopped or destroyed.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
W="${W:-/workspace}"
LOG="$W/queue_logs6i"; RES="$W/queue_results"; ST="$W/QUEUE6I_STATUS"
# 2026-11-19 InternVL2.5-8B variant (queue_6x6_ivl25.sh): own status/log/done names so the earlier MiniCPM 6x6 records on this server stay untouched;
# gate = check_6x6_gate_v2.py (adds the InternVL2.5 visual-token range); everything else identical to queue_6x6.sh d538b0cd.
GATE="$W/check_6x6_gate_v2.py"; GATE_SHA=18179a5824fd73adf1ea149c9df50786f21c98306f0b6d1f12adbcc5015a29ef
ADDON_NAME=sixbysix_addon_20261114.tar.gz
ADDON_SHA=ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a
ADDON="$W/$ADDON_NAME"; AD="$W/sixbysix_addon"
REPO_URL="${REPO_URL:-https://github.com/dltmdwoOS/mpr-gui-crosslocale.git}"
BRANCH="${BRANCH:-followup-rel-4lang-inference}"
COMMIT=666440011f538cd9336794c9965429de640843c3
QWEN="$W/mpr-gui-crosslocale"; IVL="$W/mpr-ivl"; MC="$W/mpr-minicpm"
EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"; EXPECT_TRANSFORMERS=4.57.6
MANIFEST=01_code/data/manifests/mpr_gui_manifest.jsonl
# sha256 of the manifest and of the ordered 79,020 input ids used by the InternVL3.5-8B 6x6 run (2026-11-09 contract); rebuilt locally 2026-11-14 with the same values
MANIFEST_SHA=4dd919fc9ddfeee6b3b807bba79191614bf7a475bafebc401329497c2a61aa59
PLAN_IDS_SHA=22ac60aa9b02a10ddc840e8e1a2d44c0bac4c3015678e9ff42b0a072ecd7b4c8
DEFAULT_MODELS="internvl25_8b"
MODELS="${MODELS:-${ONLY:-$DEFAULT_MODELS}}"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"
T_SETUP="${T_SETUP:-2h}"; T_DOWNLOAD="${T_DOWNLOAD:-2h}"
RETRY_SLEEP="${RETRY_SLEEP:-${DL_RETRY_SLEEP:-30}}"
# Local test hooks only (unset on the server): Q6_TEST=1 -> mock model, no GPU/4090 check, no pip, stub venv.
Q6_TEST="${Q6_TEST:-0}"
MOCK=""; [ "$Q6_TEST" = 1 ] && MOCK="--mock-model"

mkdir -p "$LOG" "$RES" 2>/dev/null
say() { echo "$(date -u +%FT%TZ) 6x6: $*" | tee -a "$ST"; }
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc 0-9; }
retry() {  # retry <tries> <cmd...>: network steps on flaky hosts (SSL EOF / gnutls handshake errors seen on the previous server)
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "retry $i/$n failed: $*" >&2
    [ "$i" = "$n" ] || sleep $(( RETRY_SLEEP * i ))
  done
  return 1
}
dur_s() {  # "90m", "24h" -> seconds (GNU timeout takes ONE unit)
  local d=$1 s=0
  [[ $d =~ ([0-9]+)h ]] && s=$((s + ${BASH_REMATCH[1]} * 3600))
  [[ $d =~ ([0-9]+)m ]] && s=$((s + ${BASH_REMATCH[1]} * 60))
  echo "$s"
}

# key -> FAM MID REV CFG MINFREE_GB T_SMOKE T_FULL SPI(s/inference, assumed) HRS(full, assumed)
# MINFREE_GB = model files x about 1.3 + 8 GB. Timeouts: about 1.6-1.8x the upper estimate.
model_info() {
  KEY=$1
  case "$1" in
    qwen3_vl_4b)   FAM=qwen; MID=Qwen/Qwen3-VL-4B-Instruct;      REV=ebb281ec70b05090aa6165b016eac8ec08e71b17; CFG=qwen3_vl_4b_4090_channels_last; MINFREE=20; T_SMOKE=45m; T_FULL=24h; SPI="0.45-0.65"; HRS="9.9-14.3" ;;
    internvl35_4b) FAM=ivl;  MID=OpenGVLab/InternVL3_5-4B;       REV=481f6e32467eab4e922ccd7fd6cf420441a62331; CFG=internvl3_5_4b_4090;          MINFREE=20; T_SMOKE=45m; T_FULL=26h; SPI="0.45-0.70"; HRS="9.9-15.4" ;;
    uitars_15_7b)  FAM=qwen; MID=ByteDance-Seed/UI-TARS-1.5-7B;  REV=683d002dd99d8f95104d31e70391a39348857f4e; CFG=uitars_1_5_7b_4090;            MINFREE=52; T_SMOKE=1h;  T_FULL=36h; SPI="0.70-1.00"; HRS="15.4-22.0" ;;
    minicpm_v45)   FAM=mc;   MID=openbmb/MiniCPM-V-4_5;          REV=14a2004908140699317e2109e90573c1e6323c36; CFG=minicpm_v4_5_8b_4090;        MINFREE=32; T_SMOKE=1h;  T_FULL=20h; SPI="0.30-0.60"; HRS="6.6-13.2" ;;
    qwen3_vl_2b)   FAM=qwen; MID=Qwen/Qwen3-VL-2B-Instruct;      REV=89644892e4d85e24eaac8bacfd4f463576704203; CFG=qwen3_vl_2b_4090_channels_last; MINFREE=14; T_SMOKE=45m; T_FULL=16h; SPI="0.30-0.45"; HRS="6.6-9.9" ;;
    internvl35_2b) FAM=ivl;  MID=OpenGVLab/InternVL3_5-2B;       REV=7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0; CFG=internvl3_5_2b_4090;          MINFREE=14; T_SMOKE=45m; T_FULL=18h; SPI="0.30-0.50"; HRS="6.6-11.0" ;;
    internvl25_8b) FAM=ivl;  MID=OpenGVLab/InternVL2_5-8B;       REV=e9e4c0dc1db56bfab10458671519b7fa3dd29463; CFG=internvl2_5_8b;               MINFREE=30; T_SMOKE=45m; T_FULL=16h; SPI="0.25-0.45"; HRS="5.5-9.9" ;;
    internvl35_1b) FAM=ivl;  MID=OpenGVLab/InternVL3_5-1B;       REV=2f71cf52542334823e48a46ffba0e2bc9add3446; CFG=internvl3_5_1b_4090;          MINFREE=12; T_SMOKE=45m; T_FULL=14h; SPI="0.25-0.40"; HRS="5.5-8.8" ;;
    *) return 1 ;;
  esac
  case "$FAM" in qwen) REPO=$QWEN ;; ivl) REPO=$IVL ;; mc) REPO=$MC ;; esac
  CACHE_DIR="$REPO/.cache/huggingface/hub/models--${MID//\//--}"
  OUT="$REPO/followup_rel_4lang/results/6x6_$KEY"
  CFGPATH="01_code/configs/models/$CFG.yaml"
  return 0
}

repo_env() {  # same cache layout as run_all_overnight.sh / the sizes queue
  export XDG_CACHE_HOME="$1/.cache" HF_HOME="$1/.cache/huggingface" TORCH_HOME="$1/.cache/torch" PIP_CACHE_DIR="$1/.cache/pip"
  export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
}

# ----------------------------------------------------------------------------------------------------------------
# Functions below run in their own bash (timeout ... bash "$SELF" --fn NAME args), env KEY etc. exported by the caller.
# ----------------------------------------------------------------------------------------------------------------
fam_repo() { case "$1" in qwen) echo "$QWEN" ;; ivl) echo "$IVL" ;; mc) echo "$MC" ;; esac; }
fam_cfgs() {
  case "$1" in
    qwen) echo "qwen3_vl_2b_4090_channels_last qwen3_vl_4b_4090_channels_last uitars_1_5_7b_4090" ;;
    ivl)  echo "internvl3_5_1b_4090 internvl3_5_2b_4090 internvl3_5_4b_4090" ;;
    mc)   echo "" ;;
  esac
}

patch_files() {  # $1 family -> patch files in order
  case "$1" in
    ivl) ls "$AD"/ivl_patches/000*.patch ;;
    mc)  ls "$AD"/minicpm_patches/000*.patch ;;
    *)   true ;;
  esac
}

ensure_commit_and_patches() {  # cwd = repo; $1 family; FRESH=1 when this run just cloned
  local fam=$1 np n i c p1 p2
  local -a P
  P=($(patch_files "$fam")); np=${#P[@]}
  git cat-file -e "$COMMIT^{commit}" 2>/dev/null || retry 3 git fetch origin "$BRANCH" || return 1
  if [ "$np" = 0 ]; then
    # plain family: the 8B runs used exactly this commit (same as run_all_overnight.sh)
    [ "$(git rev-parse HEAD)" = "$COMMIT" ] || git checkout -q -f "$COMMIT" || return 1
    echo "commit ok: ${COMMIT:0:7}"; return 0
  fi
  [ "${FRESH:-0}" = 1 ] && { git checkout -q -f "$COMMIT" || return 1; }
  git merge-base --is-ancestor "$COMMIT" HEAD || { echo "HEAD is not a descendant of $COMMIT"; return 1; }
  n=$(git rev-list --count "$COMMIT"..HEAD)
  if [ "$n" = 0 ]; then
    git am --abort 2>/dev/null || true
    git -c user.name=local -c user.email=local@local am "${P[@]}" || { git am --abort; echo "git am failed"; return 1; }
  elif [ "$n" = "$np" ]; then
    i=0
    for c in $(git rev-list --reverse "$COMMIT"..HEAD); do
      p1=$(git show "$c" | git patch-id --stable | cut -d' ' -f1)
      p2=$(git patch-id --stable < "${P[$i]}" | cut -d' ' -f1)
      [ "$p1" = "$p2" ] || { echo "commit $c differs from ${P[$i]} (patch-id)"; return 1; }
      i=$((i+1))
    done
  else
    echo "repo is $n commits after $COMMIT, expected 0 or $np patch commits"; return 1
  fi
  echo "commit/patch state ok: $(git rev-parse --short HEAD) ($np patches over ${COMMIT:0:7})"
}

venv_ok() {  # cwd = repo; $1 family
  [ -x .venv/bin/python ] || return 1
  local extra="qwen_vl_utils"; [ "$1" = ivl ] && extra="timm, einops"; [ "$1" = mc ] && extra="timm"
  EXPECT_TORCH="$EXPECT_TORCH" EXPECT_TRANSFORMERS="$EXPECT_TRANSFORMERS" REPO_DIR="$PWD" \
  .venv/bin/python - <<PY >/dev/null 2>&1
import os, sys
import torch, transformers, PIL, yaml, accelerate, mpr_crosslocale
assert torch.__version__ == os.environ["EXPECT_TORCH"], torch.__version__
assert transformers.__version__ == os.environ["EXPECT_TRANSFORMERS"], transformers.__version__
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"])), mpr_crosslocale.__file__
import ${extra}
PY
}

ensure_venv() {  # cwd = repo; $1 family
  local fam=$1
  if venv_ok "$fam"; then echo "venv ok (reused)"; return 0; fi
  if [ "$Q6_TEST" = 1 ]; then  # test hook: stub venv that runs the local test python with this repo's src
    mkdir -p .venv/bin
    printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec "%s" "$@"\n' "$PWD" "${Q6_TEST_PY:?}" > .venv/bin/python; chmod +x .venv/bin/python
    printf 'export PATH="%s/.venv/bin:$PATH"\n' "$PWD" > .venv/bin/activate
    venv_ok "$fam" && return 0 || { echo "stub venv check failed"; return 1; }
  fi
  python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "python3 is not 3.12"; return 1; }
  if [ ! -f .venv/bin/activate ]; then
    rm -rf .venv
    python3 -m venv .venv || { apt-get update && apt-get install -y python3.12-venv && python3 -m venv .venv; } || return 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  local c=""
  [ "$fam" = ivl ] && c="-c $AD/internvl35_constraints_20261109.txt"
  retry 3 python -m pip install -r 01_code/requirements-qwen3-4090.txt $c || return 1
  retry 3 python -m pip install -e './01_code[models]' $c || return 1
  python -m pip check || true
  venv_ok "$fam" || { echo "venv still not ok after install"; return 1; }
}

raw_src() {  # echo a repo whose 01_code/data/raw/mpr_gui_bench is a complete real download (not this one)
  local r
  for r in "$QWEN" "$IVL" "$MC"; do
    [ "$r" = "$PWD" ] && continue
    [ -f "$r/01_code/data/raw/mpr_gui_bench/SOURCE.json" ] && [ ! -L "$r/01_code/data/raw/mpr_gui_bench" ] && { echo "$r"; return 0; }
  done
  return 1
}

ensure_data() {  # cwd = repo
  local try d=01_code/data/raw/mpr_gui_bench m=$MANIFEST src
  mkdir -p 01_code/data/raw 01_code/data/manifests
  if [ ! -f "$d/SOURCE.json" ]; then
    if src=$(raw_src); then
      echo "reusing raw data of $src"
      rm -rf "$d"; ln -s "$src/$d" "$d"
    else
      echo "downloading MPR-GUI-Bench (about 2.6 GB, locked revisions)"
      source .venv/bin/activate
      # 3,685 small files: Hugging Face may rate-limit an anonymous IP (seen locally 2026-11-14 with 4 workers). Resumes from the staging dir.
      for try in 1 2 3 4 5; do
        ( cd 01_code && python -m mpr_crosslocale.data.download --source-lock data/manifests/source_lock.json --download-workers 2 ) && break
        echo "data download attempt $try failed (rate limit or network); waiting before resuming"
        [ "$try" = 5 ] && return 1
        sleep "${DATA_RETRY_SLEEP:-360}"
      done
    fi
  fi
  [ -f "$d/SOURCE.json" ] || { echo "raw data incomplete"; return 1; }
  if [ ! -s "$m" ]; then
    if src=$(raw_src) && [ -s "$src/$m" ]; then
      echo "copying manifest of $src"; cp "$src/$m" "$m"
    else
      source .venv/bin/activate
      rn=$(basename "$PWD"); ( cd 01_code && bash scripts/audit_release.sh > "$LOG/audit_release_$rn.log" 2>&1 ) || { echo "audit_release failed"; return 1; }
    fi
  fi
  [ -s "$m" ] || { echo "manifest missing"; return 1; }
  local got; got=$(sha256sum "$m" | cut -d' ' -f1)
  echo "manifest sha256: $got  ($(wc -l < "$m") lines)"
  if [ "$got" != "$MANIFEST_SHA" ]; then
    echo "manifest sha256 differs from the one used by the 8B 6x6 runs ($MANIFEST_SHA)"
    [ "${ALLOW_MANIFEST_DIFF:-0}" = 1 ] || return 1
  fi
}

fn_setup() {  # $1 family. Idempotent.
  local fam=$1 repo cfg
  repo=$(fam_repo "$fam"); repo_env "$repo"
  cd "$W" || return 1
  FRESH=0
  if [ ! -d "$repo/.git" ]; then
    clone_repo() { rm -rf "$repo"; git clone --branch "$BRANCH" "$REPO_URL" "$repo"; }
    retry 5 clone_repo || return 1; FRESH=1
  fi
  cd "$repo" || return 1
  ensure_commit_and_patches "$fam" || return 1
  for cfg in $(fam_cfgs "$fam"); do
    if [ ! -f "01_code/configs/models/$cfg.yaml" ] || ! cmp -s "$AD/configs/$cfg.yaml" "01_code/configs/models/$cfg.yaml"; then
      cp "$AD/configs/$cfg.yaml" "01_code/configs/models/$cfg.yaml"
    fi
  done
  ensure_venv "$fam" || return 1
  ensure_data || return 1
  echo "family $fam ready in $repo"
}

check_cfg() {  # cwd = repo; compares config with the pinned table
  KEY_MID="$MID" KEY_REV="$REV" CFGPATH="$CFGPATH" .venv/bin/python - <<'PY'
import os, yaml
c = yaml.safe_load(open(os.environ["CFGPATH"]))
assert c["model_id"] == os.environ["KEY_MID"], c["model_id"]
assert c["revision"] == os.environ["KEY_REV"], c["revision"]
print("config ok:", c["model_id"], c["revision"], "family", c.get("model_family"))
PY
}

fn_download() {  # env KEY
  model_info "$KEY"; repo_env "$REPO"; cd "$REPO" || return 1
  source .venv/bin/activate
  # 3 tries; files already downloaded or partly downloaded stay in the cache and hf resumes them (also after a timeout)
  retry 3 timeout "$T_DOWNLOAD" hf download "$MID" --revision "$REV" || return 1
  [ -s "$CACHE_DIR/snapshots/$REV/config.json" ] || { echo "no config.json in snapshot"; return 1; }
  if [ "$KEY" = minicpm_v45 ] && [ "$Q6_TEST" != 1 ]; then
    python - <<PY || return 1
import glob, os
snap = "$CACHE_DIR/snapshots/$REV"
fs = glob.glob(os.path.join(snap, "model-*.safetensors"))
size = sum(os.path.getsize(p) for p in fs)
print("safetensors files", len(fs), "bytes", size)
assert len(fs) == 4 and size == 17391890184, size  # r1: file sizes incl. headers (HF API), not index total_size
PY
  fi
  du -sh "$CACHE_DIR" | cut -f1 > "$LOG/${KEY}_size_on_disk.txt"
}

dry_plan() {  # cwd = repo. Full plan count must be 79,020 (earlier runs). Output is removed afterwards.
  local d="$OUT/dry_run"
  rm -rf "$d"; mkdir -p "$d"
  python 01_code/run_cross_locale.py --manifest "$MANIFEST" --repo-root 01_code --model-config "$CFGPATH" \
    --dimensions wf wi au ap ael rel --sample-unit semantic_items --seed 42 --batch-size 1 \
    --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --dry-run \
    --output "$d/dry.jsonl" --failures-out "$d/dry_failures.jsonl" --missing-out "$d/dry_missing.jsonl" \
    --sample-manifest-out "$d/plan.json" > "$d/dry.log" 2>&1 || { tail -5 "$d/dry.log"; return 1; }
  python - <<PY || return 1
import json
p = json.load(open("$d/plan.json"))
n, u = len(p["input_ids"]), len(set(p["input_ids"]))
print("dry-run plan: input_ids", n, "unique", u, "semantic_items", p["n_semantic_items"])
assert n == u == 79020, (n, u)
import hashlib
h = hashlib.sha256("\\n".join(p["input_ids"]).encode()).hexdigest()
print("plan ids sha256", h)
assert h == "$PLAN_IDS_SHA" or "${ALLOW_MANIFEST_DIFF:-0}" == "1", "plan differs from the 8B 6x6 run"
PY
  rm -rf "$d"
}

fn_smoke() {  # env KEY
  model_info "$KEY"; repo_env "$REPO"; cd "$REPO" || return 1
  source .venv/bin/activate
  mkdir -p "$OUT/logs"
  check_cfg || return 1
  dry_plan || return 1
  python 01_code/run_cross_locale.py --manifest "$MANIFEST" --repo-root 01_code \
    --model-config "$CFGPATH" --dimensions wf wi au ap ael rel \
    --sample-size 6 --sample-unit semantic_items --seed 42 --batch-size 1 \
    --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --resume $MOCK \
    --output "$OUT/smoke_6x6.jsonl" --failures-out "$OUT/smoke_6x6_failures.jsonl" \
    --sample-manifest-out "$OUT/smoke_6x6_plan.json" 2>&1 | tee -a "$OUT/logs/smoke_6x6.log"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  python "$GATE" smoke "$OUT/smoke_6x6.jsonl" "$OUT/smoke_6x6_plan.json" "$OUT/gate_6x6_smoke.json" --expected 216 --model "$KEY" \
    > "$OUT/logs/smoke_gate.log" 2>&1
  local rc=$?
  tail -1 "$OUT/logs/smoke_gate.log"
  return $rc
}

fn_full() {  # env KEY. Two passes: the second retries inputs whose record is a failure.
  model_info "$KEY"; repo_env "$REPO"; cd "$REPO" || return 1
  source .venv/bin/activate
  mkdir -p "$OUT/logs"
  local pass rc=1
  for pass in 1 2; do
    python 01_code/run_cross_locale.py --manifest "$MANIFEST" --repo-root 01_code \
      --model-config "$CFGPATH" --dimensions wf wi au ap ael rel \
      --sample-unit semantic_items --seed 42 --batch-size 1 \
      --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --resume $MOCK \
      --output "$OUT/full_6x6.jsonl" --failures-out "$OUT/full_6x6_failures.jsonl" \
      --sample-manifest-out "$OUT/full_6x6_plan.json" 2>&1 | tee -a "$OUT/logs/full_6x6.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] && break
    echo "full 6x6 pass $pass exited $rc"
  done
  return $rc
}

fn_post() {  # env KEY. Gate on whatever is in full_6x6.jsonl, then summary. Not under the long timeout.
  model_info "$KEY"; repo_env "$REPO"; cd "$REPO" || return 1
  source .venv/bin/activate
  local rc=0
  if [ -s "$OUT/full_6x6_plan.json" ]; then
    python "$GATE" complete "$OUT/full_6x6.jsonl" "$OUT/full_6x6_plan.json" "$OUT/gate_6x6_full.json" --expected 79020 --model "$KEY" \
      > "$OUT/logs/full_gate.log" 2>&1 || rc=1
    tail -1 "$OUT/logs/full_gate.log"
  else
    echo "GATE failed: no full_6x6_plan.json"; rc=1
  fi
  if [ -s "$OUT/full_6x6.jsonl" ]; then
    python -m mpr_crosslocale.analysis.summarize_cross_locale --input "$OUT/full_6x6.jsonl" \
      --json-out "$OUT/full_6x6_summary.json" --csv-out "$OUT/full_6x6_summary.csv" > "$OUT/logs/summary.log" 2>&1 \
      || { echo "summary failed (see $OUT/logs/summary.log)"; rc=1; }
  fi
  return $rc
}

if [ "${1:-}" = "--fn" ]; then
  fn=$2; shift 2
  "$fn" "$@"; exit $?
fi

# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
if [ "${1:-}" = "--plan" ]; then
  echo "Plan (nothing is executed). Default order: $DEFAULT_MODELS"
  echo "Selected: $MODELS"
  printf '%-15s %-34s %-6s %-11s %-12s %-8s %s\n' key model repo s/infer hours_full t_full min_free_GB
  for k in $MODELS; do
    if model_info "$k"; then printf '%-15s %-34s %-6s %-11s %-12s %-8s %s\n' "$k" "$MID" "$FAM" "$SPI" "$HRS" "$T_FULL" "$MINFREE"; else echo "$k: unknown key"; fi
  done
  cat <<EOF
Steps: wait for WAIT_FOR (${WAIT_FOR:-none}); GPU idle + 4090 check; network check (>= $NET_MIN_MBPS MB/s from huggingface.co and download.pytorch.org, else STOP unless FORCE_SLOW_NET=1);
addon sha256 + unpack; per model: family setup (clone/commit/patches/venv/data/manifest), disk check, pinned download, dry-run plan (79,020),
smoke 216 + gate (sanity rules, ETA), full 79,020 + gate (not started if the smoke ETA exceeds the timeout), summary, package to $RES/6x6_<key>/,
model cache: deleted when the model finished, kept after a failure/timeout while the free disk covers the next model (KEEP_MODEL_CACHE=1: never deleted);
then touch $W/QUEUE6I_DONE. Network steps (clone, fetch, pip, data, hf download) are retried.
Timeouts: setup $T_SETUP, download $T_DOWNLOAD (x3 tries), smoke and full per the table above.
EOF
  exit 0
fi

net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}

net_check() {  # sets NET_HF, NET_PT; STOPs when too slow
  local need_pt=0 k
  for k in $MODELS; do
    model_info "$k" || continue
    grep -q "$EXPECT_TORCH" "$REPO"/.venv/lib/python3*/site-packages/torch/version.py 2>/dev/null || need_pt=1
  done
  NET_HF=$(net_speed "${NET_HF_URL:-https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/model-00001-of-00002.safetensors}")
  NET_PT=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${NET_HF} MB/s, download.pytorch.org ${NET_PT} MB/s (torch download needed: $([ $need_pt = 1 ] && echo yes || echo no)); minimum $NET_MIN_MBPS"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$NET_HF" "$NET_PT" > "$RES/net_check.txt"
  local slow=0
  case "$NET_HF" in
    ERR:4*) say "WARNING: the huggingface test URL answered HTTP ${NET_HF#ERR:}; speed unknown, continuing" ;;
    ERR*) slow=1 ;;
    *) awk -v v="$NET_HF" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
  esac
  if [ "$need_pt" = 1 ]; then
    case "$NET_PT" in
      ERR:4*) say "WARNING: the pytorch test URL answered HTTP ${NET_PT#ERR:}; speed unknown, continuing" ;;
      ERR*) slow=1 ;;
      *) awk -v v="$NET_PT" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
    esac
  fi
  if [ "$slow" = 1 ]; then
    if [ "${FORCE_SLOW_NET:-0}" = 1 ]; then say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; fi
    say "STOP: network too slow or unreachable (need >= $NET_MIN_MBPS MB/s). Nothing was downloaded. Move to another instance, or FORCE_SLOW_NET=1 to continue anyway."
    exit 1
  fi
}

if [ "${1:-}" = "--net" ]; then
  MODELS="${MODELS:-$DEFAULT_MODELS}"; net_check; exit 0
fi

command -v flock >/dev/null && { exec 9> "$W/.queue6.lock"; flock -n 9 || { echo "another queue_6x6.sh is already running"; exit 1; }; }
trap 'say "interrupted by signal; no QUEUE6I_DONE written. Rerun the same command to resume."; exit 130' INT TERM
command -v timeout >/dev/null || { say "STOP: GNU timeout missing"; exit 1; }
rm -f "$W/QUEUE6I_DONE"
say "queue started: $MODELS"

# 0. wait for an earlier queue, then make sure nothing else uses the GPU
if [ -n "${WAIT_FOR:-}" ]; then
  say "waiting for $WAIT_FOR (max ${WAIT_FOR_MAX_H:-72} h)"
  waited=0
  until [ -e "$WAIT_FOR" ]; do
    sleep "${POLL_S:-60}"; waited=$((waited + ${POLL_S:-60}))
    [ $((waited % 1800)) -lt "${POLL_S:-60}" ] && say "still waiting for $WAIT_FOR (${waited}s)"
    [ "$waited" -ge $(( ${WAIT_FOR_MAX_H:-72} * 3600 )) ] && { say "STOP: $WAIT_FOR did not appear within ${WAIT_FOR_MAX_H:-72} h"; exit 1; }
  done
  say "$WAIT_FOR found: $(head -c 200 "$WAIT_FOR" | tr '\n' ' ')"
  pgrep -f 'run_all_overnight\.sh' >/dev/null 2>&1 && say "run_all_overnight.sh is still running; waiting for it to end"
  while pgrep -f 'run_all_overnight\.sh' >/dev/null 2>&1; do
    sleep "${POLL_S:-60}"; waited=$((waited + ${POLL_S:-60}))
    [ "$waited" -ge $(( ${WAIT_FOR_MAX_H:-72} * 3600 )) ] && { say "STOP: run_all_overnight.sh still running after ${WAIT_FOR_MAX_H:-72} h"; exit 1; }
  done
elif [ "${IGNORE_OTHER_QUEUES:-0}" != 1 ] && pgrep -f 'queue_minicpm.sh|queue_sizes_uitars.sh|run_all_overnight.sh' >/dev/null 2>&1; then
  say "STOP: another queue script is running (pgrep). Start with WAIT_FOR=<its DONE file>, or IGNORE_OTHER_QUEUES=1 if you checked."
  exit 1
fi
if [ "$Q6_TEST" != 1 ]; then
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 4090 || { say "STOP: GPU is not an RTX 4090"; exit 1; }
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
    [ -z "$used" ] && { say "STOP: nvidia-smi not available"; exit 1; }
    [ "$used" -lt 1500 ] && break
    sleep 60
  done
  [ "${used:-99999}" -lt 1500 ] || { say "STOP: GPU still busy (${used} MiB used) after 30 min. Check nvidia-smi"; exit 1; }
fi

# 1. network, addon
net_check
[ -s "$ADDON" ] || { say "STOP: $ADDON is missing"; exit 1; }
echo "$ADDON_SHA  $ADDON" | sha256sum -c - > "$LOG/addon_sha.log" 2>&1 || { say "STOP: addon sha256 differs (see $LOG/addon_sha.log)"; exit 1; }
tar -xzf "$ADDON" -C "$W" && ( cd "$AD" && sha256sum -c SHA256SUMS ) > "$LOG/addon_unpack.log" 2>&1 || { say "STOP: addon unpack/verify failed"; exit 1; }
echo "$GATE_SHA  $GATE" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || { say "STOP: $GATE missing or sha256 differs"; exit 1; }
say "addon verified; free disk $(free_gb) GB"

run_step() {  # run_step <name> <limit> <fn> [args]; sets STEP_RC
  local name=$1 limit=$2; shift 2
  timeout -k 60 "$limit" bash "$SELF" --fn "$@" >> "$LOG/$name.log" 2>&1
  STEP_RC=$?
  return $STEP_RC
}
rc_text() { case "$1" in 0) echo ok ;; 124|137) echo TIMEOUT ;; *) echo "FAILED(exit $1)" ;; esac; }

package() {  # $1 suffix; cwd irrelevant
  local dest="$RES/6x6_$KEY" tarf stage="$OUT/package_meta"
  mkdir -p "$dest" "$OUT/queue_logs" "$stage"
  cp "$LOG/${KEY}"_*.log "$OUT/queue_logs/" 2>/dev/null
  cp "$ST" "$OUT/queue_logs/QUEUE6_STATUS_at_packaging.txt" 2>/dev/null
  cp "$REPO/$CFGPATH" "$stage/" 2>/dev/null
  ( cd "$REPO" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -8 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt"
    sha256sum "$MANIFEST" > "$stage/manifest_sha256.txt" 2>/dev/null
    [ -x .venv/bin/python ] && .venv/bin/python -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null ) 2>/dev/null
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1; df -h "$W" > "$stage/disk.txt" 2>&1
  cp "$AD/SHA256SUMS" "$stage/addon_SHA256SUMS" 2>/dev/null; echo "$ADDON_SHA  $ADDON_NAME" > "$stage/addon_tar_sha256.txt"
  tarf="$dest/6x6_${KEY}_$(date -u +%Y%m%d_%H%M)$1.tar.gz"
  tar -czf "$tarf" -C "$(dirname "$OUT")" "$(basename "$OUT")" && tar -tzf "$tarf" > /dev/null \
    && ( cd "$dest" && sha256sum "$(basename "$tarf")" >> SHA256SUMS ) && echo "$tarf"
}

next_need_gb() {  # largest MINFREE of the models after $KEY in MODELS that are not complete; 0 after the last one
  local k seen=0 need=0 m
  for k in $MODELS; do
    if [ "$seen" = 1 ] && [ ! -f "$RES/6x6_$k/COMPLETE" ]; then m=$( model_info "$k" && echo "$MINFREE" ); [ -n "$m" ] && [ "$m" -gt "$need" ] && need=$m; fi
    [ "$k" = "$KEY" ] && seen=1
  done
  echo "$need"
}

cache_rule() {  # $1 = success | notdone
  local f need; f=$(free_gb); need=$(next_need_gb)
  if [ "$1" = success ] && [ "${KEEP_MODEL_CACHE:-0}" != 1 ]; then rm -rf "$CACHE_DIR"; say "$KEY model cache deleted (model finished)"; return; fi
  if [ "${KEEP_MODEL_CACHE:-0}" = 1 ] || [ "$1" = success ]; then say "$KEY model cache kept (KEEP_MODEL_CACHE=1)"; return; fi
  if [ "$f" -ge "$need" ]; then
    say "$KEY model cache KEPT (not finished; free $f GB >= $need GB needed next). A rerun resumes without downloading."
  else
    rm -rf "$CACHE_DIR"; say "$KEY model cache deleted (not finished; free $f GB < $need GB needed by the next model); free $(free_gb) GB"
  fi
}

eta_guard() {  # smoke timing vs the full-run timeout. Returns 3 (and says why) if the full run cannot finish in T_FULL
  [ "${IGNORE_ETA:-0}" = 1 ] && return 0
  local msg rc=0
  msg=$(python3 - "$OUT/gate_6x6_smoke.json" "$(dur_s "$T_FULL")" "$OUT/full_6x6.jsonl" <<'PY' 2>&1
import json, sys
gate, limit, preds = sys.argv[1], float(sys.argv[2]), sys.argv[3]
total = 79020
try:
    eta = json.load(open(gate))["eta_full_seconds"]
except Exception:
    eta = None
if eta is None:
    print("no ETA in the smoke gate; full run starts without ETA check"); sys.exit(0)
done = 0
try:
    done = sum(1 for line in open(preds) if '"status": "success"' in line)
except OSError:
    pass
remaining = eta * max(0, total - done) / total
print(f"ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs timeout {limit / 3600:.1f} h")
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then say "$KEY full 6x6 STOPPED, not started: $msg. It could not finish in time. IGNORE_ETA=1 overrides."; return 3; fi
  say "$KEY ETA check: $msg"; return 0
}

gate_problems() {  # $1 gate json -> "PROBLEMS a; b"
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try:
    g = json.load(open(sys.argv[1]))
    if g.get("problems"):
        print("PROBLEMS " + "; ".join(g["problems"]) + ". ", end="")
    notes = [f"{k}: {v}" for k, v in g.get("sanity", {}).items() if isinstance(v, str) and ("not evaluated" in v or "not recorded" in v)]
    if notes:
        print("NOT CHECKED (field missing) " + "; ".join(notes) + ". ", end="")
except Exception:
    pass
PY
}

run_one() {
  model_info "$1" || { say "$1: unknown key, skipped"; return 1; }
  export KEY FAM MID REV CFG REPO CACHE_DIR OUT CFGPATH MINFREE
  if [ -f "$RES/6x6_$KEY/COMPLETE" ]; then say "$KEY already complete, skipped"; return 0; fi
  local f t0=$SECONDS
  say "$KEY start: $MID@$REV free $(free_gb) GB; expected full run $HRS h at $SPI s/inference (assumed, unmeasured)"
  # family setup (once per family; a failure skips every model of that family)
  if [ "$(eval echo \${FAMSTATE_$FAM:-})" = FAILED ]; then say "$KEY SKIPPED: $FAM setup failed earlier"; return 1; fi
  if [ "$(eval echo \${FAMSTATE_$FAM:-})" != ok ]; then
    f=$(free_gb)
    if [ ! -x "$REPO/.venv/bin/python" ] && [ "$f" -lt 25 ]; then say "$KEY SKIPPED: only ${f} GB free for a new venv/data setup (need 25)"; return 1; fi
    say "$KEY: setting up family $FAM in $REPO (limit $T_SETUP)"
    run_step "setup_$FAM" "$T_SETUP" fn_setup "$FAM"
    if [ "$STEP_RC" = 0 ]; then eval "FAMSTATE_$FAM=ok"; say "$KEY: family $FAM ready ($(tail -1 "$LOG/setup_$FAM.log"))"
    else eval "FAMSTATE_$FAM=FAILED"; say "$KEY SETUP FAILED $(rc_text $STEP_RC) (see $LOG/setup_$FAM.log)"; return 1; fi
  fi
  mkdir -p "$OUT/logs"
  # download
  if [ ! -s "$CACHE_DIR/snapshots/$REV/config.json" ]; then
    f=$(free_gb)
    if [ "$f" -lt "$MINFREE" ]; then say "$KEY SKIPPED: only ${f} GB free (need $MINFREE). Free space (e.g. old 8B caches) and rerun"; return 1; fi
  fi
  say "$KEY: download (limit ${T_DOWNLOAD} x3)"
  if ! run_step "${KEY}_download" "$(( 3 * 2 * 3600 + 600 ))" fn_download; then
    say "$KEY DOWNLOAD FAILED (see $LOG/${KEY}_download.log); partial files stay for the next run"; cache_rule notdone; return 1
  fi
  say "$KEY downloaded, $(cat "$LOG/${KEY}_size_on_disk.txt" 2>/dev/null) on disk; free $(free_gb) GB"
  local smoke=FAIL full=skipped gate=skipped
  if [ -s "$OUT/gate_6x6_smoke.json" ] && grep -q '"status": "passed"' "$OUT/gate_6x6_smoke.json"; then
    smoke=ok; say "$KEY smoke already passed"
  else
    say "$KEY smoke 6x6 (limit $T_SMOKE)"
    if run_step "${KEY}_smoke" "$T_SMOKE" fn_smoke; then smoke=ok; else smoke="$(rc_text $STEP_RC)"; fi
    say "$KEY smoke $smoke: $(grep '^GATE' "$OUT/logs/smoke_gate.log" 2>/dev/null | tail -1) $(grep '^WARNING' "$OUT/logs/smoke_gate.log" 2>/dev/null | tr '\n' ' ')$(gate_problems "$OUT/gate_6x6_smoke.json")"
  fi
  if [ "$smoke" = ok ]; then
    if [ -s "$OUT/gate_6x6_full.json" ] && grep -q '"status": "passed"' "$OUT/gate_6x6_full.json"; then
      full=ok; gate=ok; say "$KEY full 6x6 already complete"
    else
      if eta_guard; then
        say "$KEY full 6x6 started (limit $T_FULL)"
        if run_step "${KEY}_full" "$T_FULL" fn_full; then full=ok; else full="$(rc_text $STEP_RC)"; fi
        if run_step "${KEY}_post" 1h fn_post; then gate=ok; else gate=FAILED; fi
        say "$KEY full 6x6 $full; gate $gate: $(grep '^GATE' "$OUT/logs/full_gate.log" 2>/dev/null | tail -1) $(gate_problems "$OUT/gate_6x6_full.json")"
      else
        full=STOPPED_ETA
      fi
    fi
  else
    say "$KEY smoke/gate did not pass; full run skipped (see $LOG/${KEY}_smoke.log)"
  fi
  local suffix=""; { [ "$smoke" = ok ] && [ "$full" = ok ] && [ "$gate" = ok ]; } || suffix="_PARTIAL"
  local tarf; if tarf=$(package "$suffix"); then say "$KEY packaged: $tarf ($(du -h "$tarf" | cut -f1))"; else say "$KEY PACKAGING FAILED"; tarf=""; fi
  [ -z "$suffix" ] && [ -n "$tarf" ] && touch "$RES/6x6_$KEY/COMPLETE"
  if [ -z "$suffix" ]; then cache_rule success; else cache_rule notdone; fi
  say "$KEY finished in $(( (SECONDS - t0) / 60 )) min: smoke=$smoke full=$full gate=$gate; free $(free_gb) GB"
  [ -z "$suffix" ]
}

summary=""
for k in $MODELS; do
  if run_one "$k"; then summary="$summary $k=ok"; else summary="$summary $k=NOT_COMPLETE"; fi
done
say "QUEUE FINISHED:$summary. Download $RES/ and $ST, then DESTROY the instance (Stop keeps charging disk)."
df -h "$W" > "$RES/disk_end6i.log" 2>&1; cp "$ST" "$RES/QUEUE6I_STATUS.txt"; cp -r "$LOG" "$RES/queue_logs6i" 2>/dev/null
touch "$W/QUEUE6I_DONE"
