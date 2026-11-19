#!/usr/bin/env bash
# Experiment A (O, T, R; 1,896 pairs x 3 = 5,688 inferences) for OpenGVLab/InternVL2_5-8B on ONE RTX 4090, batch size 1.
# Run from anywhere; it changes to the repo root (/workspace/mpr-ivl). Called by queue_internvl25_expA.sh.
#   bash followup_rel_4lang/expA_ivl25/run_expA_ivl25.sh smoke|full
# Rerunning the same command resumes; a stage whose gate already passed is skipped.
# Optional env (set by the queue): EXPA_LIMIT_S = timeout of the full step in seconds. If the ETA taken from the smoke timing
# (gate json, timing.eta_full_seconds, minus work already done) is longer, the step is NOT started: exit 3 and a STOPPED line in STATUS.
# IGNORE_ETA=1 turns this off. LOCAL_MOCK=1 (local tests only): no GPU / no model snapshot, adds --mock-model.
set -Eeuo pipefail
STAGE="${1:?smoke or full}"
KEY=internvl25_8b
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
X=followup_rel_4lang/expA_ivl25
CONFIG=$X/internvl2_5_8b.yaml
REFERENCE=$X/reference_contract_internvl25_rel_20261005.json   # InternVL2.5-8B REL run contract: system prompt / template check only
OUT=followup_rel_4lang/results/expA_${KEY}
export CONFIG REFERENCE
mkdir -p "$OUT/logs"
MOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && MOCK="--mock-model"
status() { echo "$(date -u +%FT%TZ) [$KEY] $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at stage ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
passed() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }

eta_guard() {  # $1 smoke gate json, $2 limit seconds ("" = no check), $3 predictions file of the full run, $4 full inference count
  [ -z "${2:-}" ] || [ "${IGNORE_ETA:-0}" = 1 ] && return 0
  local msg rc=0
  msg=$(python - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
gate, limit, preds, total = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
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
print(f"Exp A ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs step timeout {limit / 3600:.1f} h")
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then status "STOPPED: $msg. The step is not started (it could not finish in time). IGNORE_ETA=1 overrides."; return 3; fi
  status "ETA check ok: $msg"; return 0
}

preflight() {
  CURRENT=preflight
  (cd followup_rel_4lang/expA && sha256sum -c expA_inputs/EXPA_SHA256SUMS) > "$OUT/logs/expA_inputs_verify.log"
  (cd followup_rel_4lang/expA_ivl25 && sha256sum -c SHA256SUMS) > "$OUT/logs/addon_verify.log"
  # the config in the addon must be byte-identical to the one in the repo (= the one of the 2026-10-05 REL run)
  cmp "$CONFIG" 01_code/configs/models/internvl2_5_8b.yaml
  [ -z "$MOCK" ] && python - <<'PY'
import importlib, json, os
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
assert cfg["model_id"] == "OpenGVLab/InternVL2_5-8B" and cfg["revision"] == "e9e4c0dc1db56bfab10458671519b7fa3dd29463", cfg
assert cfg["model_family"] == "internvl2_5" and cfg["max_num"] == 7 and cfg["attn_implementation"] == "eager"
name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and "4090" in name, name
for pkg in ("torch", "transformers", "timm", "einops", "sentencepiece"):
    importlib.import_module(pkg)
from huggingface_hub import snapshot_download
snap = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
assert Path(snap, "config.json").is_file(), snap
print("GPU", name, "| torch", torch.__version__, "| model", cfg["model_id"], cfg["revision"], "|", snap)
PY
  python followup_rel_4lang/data_bundle.py verify > "$OUT/logs/bundle_verify.log"
  python followup_rel_4lang/expA_ivl25/run_rel_T_generic.py --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
    --split main --output-dir "$OUT/dry_run_expa" --dry-run > "$OUT/logs/dry_run_expa.log"
  python - "$OUT/dry_run_expa/plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["inferences"] == 5688 and p["pairs"] == 1896 and p["batch_size"] == 1, p
ref = p["reference_contract_check"]
print("plan ok:", p["inferences"], "inferences,", p["pairs"], "pairs; config keys differing from the REL contract:", ref["config_keys_that_differ"])
PY
  status "preflight passed"
}

meta() {
  mkdir -p "$OUT/meta"
  cp "$CONFIG" "$OUT/meta/"
  git rev-parse HEAD > "$OUT/meta/git_commit.txt" 2>&1 || true
  git status --short > "$OUT/meta/git_status.txt" 2>&1 || true
  python -m pip freeze > "$OUT/meta/requirements.freeze.txt" || true
  nvidia-smi > "$OUT/meta/nvidia_smi.txt" || true
  sha256sum "$CONFIG" "$X"/*.py "$X"/run_expA_ivl25.sh > "$OUT/meta/addon_hashes.txt" || true
}

run_smoke() {
  CURRENT=smoke_expa
  passed "$OUT/expa_smoke/gate_expa_smoke.json" || {
    status "Exp A smoke started (--max-items 3, 108 inferences)"
    python "$X/run_rel_T_generic.py" --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
      --split main --max-items 3 --resume $MOCK --output-dir "$OUT/expa_smoke" 2>&1 | tee -a "$OUT/logs/expa_smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_expa_gate
    python "$X/check_generic_gate.py" expa smoke "$OUT/expa_smoke" --model "$KEY" > "$OUT/logs/expa_smoke_gate.log"
  }
  meta
  status "smoke passed"
}

run_full() {
  CURRENT=expa_full
  passed "$OUT/expa_full/gate_expa_complete.json" && { status "Exp A full already done"; return 0; }
  eta_guard "$OUT/expa_smoke/gate_expa_smoke.json" "${EXPA_LIMIT_S:-}" "$OUT/expa_full/predictions.jsonl" 5688 || exit 3
  status "Exp A O/T/R full started (5,688 inferences)"
  python "$X/run_rel_T_generic.py" --model "$KEY" --model-config "$CONFIG" --reference-contract "$REFERENCE" \
    --split main --resume $MOCK --output-dir "$OUT/expa_full" 2>&1 | tee -a "$OUT/logs/expa_full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=expa_gate
  python "$X/check_generic_gate.py" expa complete "$OUT/expa_full" 5688 --model "$KEY" > "$OUT/logs/expa_complete_gate.log"
  status "Exp A full done"
}

case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  *) echo "Unknown stage: $STAGE (smoke or full)"; exit 2 ;;
esac
