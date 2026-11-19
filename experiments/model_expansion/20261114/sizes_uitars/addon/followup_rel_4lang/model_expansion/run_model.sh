#!/usr/bin/env bash
# One new model (batch size 1) on ONE RTX 4090: smoke, REL O/R/C/F full, Exp A O/T/R full.
# Run from anywhere; it changes to the repo root (the folder that holds followup_rel_4lang/). Called by queue_sizes_uitars.sh.
#   bash followup_rel_4lang/model_expansion/run_model.sh <key> smoke|rel|expa
# keys: qwen3_vl_2b qwen3_vl_4b internvl35_1b internvl35_2b internvl35_4b uitars_15_7b
# Qwen3-VL and UI-TARS go in the Qwen repo (mpr-gui-crosslocale), InternVL3.5 in mpr-ivl; the caller sets HF_HOME/XDG_CACHE_HOME.
# Rerunning the same command resumes; a stage whose gate already passed is skipped.
# Optional env (set by the queue): REL_LIMIT_S / EXPA_LIMIT_S = timeout of the rel / expa step in seconds. If the ETA taken from the
# smoke timing (gate json, "timing.eta_full_seconds", minus work already done) is longer than that, the step is NOT started: exit 3 and a
# STOPPED line in STATUS. IGNORE_ETA=1 turns this off.
set -Eeuo pipefail
KEY="${1:?model key}"; STAGE="${2:?smoke, rel or expa}"
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}" HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
source .venv/bin/activate
case "$KEY" in
  qwen3_vl_2b)    CONFIG=01_code/configs/models/qwen3_vl_2b_4090_channels_last.yaml; FAM=qwen ;;
  qwen3_vl_4b)    CONFIG=01_code/configs/models/qwen3_vl_4b_4090_channels_last.yaml; FAM=qwen ;;
  uitars_15_7b)   CONFIG=01_code/configs/models/uitars_1_5_7b_4090.yaml;            FAM=qwen ;;
  internvl35_1b)  CONFIG=01_code/configs/models/internvl3_5_1b_4090.yaml;           FAM=ivl ;;
  internvl35_2b)  CONFIG=01_code/configs/models/internvl3_5_2b_4090.yaml;           FAM=ivl ;;
  internvl35_4b)  CONFIG=01_code/configs/models/internvl3_5_4b_4090.yaml;           FAM=ivl ;;
  *) echo "Unknown model key: $KEY"; exit 2 ;;
esac
if [ "$FAM" = qwen ]; then
  source followup_rel_4lang/qwen3_4090_env.sh
  REFERENCE=followup_rel_4lang/rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json   # system prompt check only
  PKGS="torch transformers qwen_vl_utils"
else
  source followup_rel_4lang/internvl35_4090_env.sh
  REFERENCE=followup_rel_4lang/rstem_ropts_inputs/run_contract_internvl35_rel_20261109.json
  PKGS="torch transformers timm"
fi
export HF_HOME XDG_CACHE_HOME   # the env scripts may set HF_HOME; both point to <repo>/.cache/huggingface when the queue sets them
export CONFIG REFERENCE PKGS
MX=followup_rel_4lang/model_expansion
OUT=followup_rel_4lang/results/sizes_${KEY}
mkdir -p "$OUT/logs"
BATCH=1
# Local test hooks only (unset on the server): LOCAL_MOCK=1 skips the GPU/model-snapshot check and adds --mock-model.
MOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && MOCK="--mock-model"
status() { echo "$(date -u +%FT%TZ) [$KEY] $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at stage ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
passed() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }

eta_guard() {  # $1 kind, $2 smoke gate json, $3 limit seconds ("" = no check), $4 predictions file of the full run, $5 full inference count
  [ -z "${3:-}" ] || [ "${IGNORE_ETA:-0}" = 1 ] && return 0
  local msg rc=0
  msg=$(python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys
kind, gate, limit, preds, total = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
try:
    eta = json.load(open(gate))["timing"]["eta_full_seconds"]
except Exception:
    eta = None
if eta is None:
    print(f"no ETA available in {gate}; step starts without ETA check"); sys.exit(0)
done = 0
try:
    done = sum(1 for line in open(preds) if '"status": "success"' in line)
except OSError:
    pass
remaining = eta * max(0, total - done) / total
text = f"{kind} ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs step timeout {limit / 3600:.1f} h"
print(text)
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then status "STOPPED: $msg. The $1 step is not started (it could not finish in time). IGNORE_ETA=1 overrides."; return 3; fi
  status "ETA check ok: $msg"; return 0
}

preflight() {
  CURRENT=preflight
  (cd followup_rel_4lang/expA && sha256sum -c expA_inputs/EXPA_SHA256SUMS) > "$OUT/logs/expA_inputs_verify.log"
  [ -z "$MOCK" ] && python - <<'PY'
import importlib, json, os, sys
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
assert cfg["revision"] and len(cfg["revision"]) == 40, cfg["revision"]
name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and "4090" in name, name
for pkg in os.environ["PKGS"].split():
    importlib.import_module(pkg)
from huggingface_hub import snapshot_download
snap = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
assert Path(snap, "config.json").is_file(), snap
print("GPU", name, "| torch", torch.__version__, "| model", cfg["model_id"], cfg["revision"], "|", snap)
PY
  python followup_rel_4lang/data_bundle.py verify > "$OUT/logs/bundle_verify.log"
  python followup_rel_4lang/run_inference.py --model-config "$CONFIG" --batch-size "$BATCH" \
    --output-dir "$OUT/dry_run_rel" --dry-run > "$OUT/logs/dry_run_rel.log"
  python "$MX/run_rel_T_generic.py" --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
    --output-dir "$OUT/dry_run_expa" --dry-run > "$OUT/logs/dry_run_expa.log"
  status "preflight passed"
}

meta() {
  mkdir -p "$OUT/meta"
  cp "$CONFIG" "$OUT/meta/"
  git rev-parse HEAD > "$OUT/meta/git_commit.txt" 2>&1 || true
  git status --short > "$OUT/meta/git_status.txt" 2>&1 || true
  python -m pip freeze > "$OUT/meta/requirements.freeze.txt" || true
  nvidia-smi > "$OUT/meta/nvidia_smi.txt" || true
  sha256sum "$CONFIG" "$MX"/*.py "$MX"/run_model.sh > "$OUT/meta/addon_hashes.txt" || true
}

run_smoke() {
  CURRENT=smoke_rel
  passed "$OUT/rel_smoke/gate_rel_smoke.json" || {
    status "REL smoke started (--max-items 2)"
    python followup_rel_4lang/run_inference.py --model-config "$CONFIG" --batch-size "$BATCH" --strict-batch-size \
      --max-items 2 --resume $MOCK --output-dir "$OUT/rel_smoke" 2>&1 | tee -a "$OUT/logs/rel_smoke.log"
    CURRENT=smoke_rel_gate
    python "$MX/check_generic_gate.py" rel smoke "$OUT/rel_smoke" --model "$KEY" > "$OUT/logs/rel_smoke_gate.log"
  }
  CURRENT=smoke_expa
  passed "$OUT/expa_smoke/gate_expa_smoke.json" || {
    status "Exp A smoke started (--max-items 3, 108 inferences)"
    python "$MX/run_rel_T_generic.py" --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
      --split main --max-items 3 --resume $MOCK --output-dir "$OUT/expa_smoke" 2>&1 | tee -a "$OUT/logs/expa_smoke.log"
    CURRENT=smoke_expa_gate
    python "$MX/check_generic_gate.py" expa smoke "$OUT/expa_smoke" --model "$KEY" > "$OUT/logs/expa_smoke_gate.log"
  }
  meta
  status "smoke passed"
}

run_rel() {
  CURRENT=rel_full
  passed "$OUT/rel_full/gate_rel_complete.json" && { status "REL full already done"; return 0; }
  eta_guard REL "$OUT/rel_smoke/gate_rel_smoke.json" "${REL_LIMIT_S:-}" "$OUT/rel_full/predictions.jsonl" 18432 || exit 3
  status "REL O/R/C/F full started (18,432 inferences)"
  python followup_rel_4lang/run_inference.py --model-config "$CONFIG" --batch-size "$BATCH" --strict-batch-size \
    --resume $MOCK --output-dir "$OUT/rel_full" 2>&1 | tee -a "$OUT/logs/rel_full.log"
  CURRENT=rel_gate
  python "$MX/check_generic_gate.py" rel complete "$OUT/rel_full" 18432 --model "$KEY" > "$OUT/logs/rel_complete_gate.log"
  status "REL full done"
}

run_expa() {
  CURRENT=expa_full
  passed "$OUT/expa_full/gate_expa_complete.json" && { status "Exp A full already done"; return 0; }
  eta_guard "Exp A" "$OUT/expa_smoke/gate_expa_smoke.json" "${EXPA_LIMIT_S:-}" "$OUT/expa_full/predictions.jsonl" 5688 || exit 3
  status "Exp A O/T/R full started (5,688 inferences)"
  python "$MX/run_rel_T_generic.py" --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
    --split main --resume $MOCK --output-dir "$OUT/expa_full" 2>&1 | tee -a "$OUT/logs/expa_full.log"
  CURRENT=expa_gate
  python "$MX/check_generic_gate.py" expa complete "$OUT/expa_full" 5688 --model "$KEY" > "$OUT/logs/expa_complete_gate.log"
  status "Exp A full done"
}

case "$STAGE" in
  smoke) preflight; run_smoke ;;
  rel) preflight; run_rel ;;
  expa) preflight; run_expa ;;
  *) echo "Unknown stage: $STAGE"; exit 2 ;;
esac
