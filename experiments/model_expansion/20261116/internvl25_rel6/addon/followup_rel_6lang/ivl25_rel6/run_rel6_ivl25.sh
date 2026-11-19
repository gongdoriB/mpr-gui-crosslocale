#!/usr/bin/env bash
# 6-language REL O/R/C/F (en, zh, th, ru, fr, ja; 52,572 condition rows -> 44,489 canonical inferences) for OpenGVLab/InternVL2_5-8B on ONE
# RTX 4090, batch size 1. Run from anywhere; it changes to the repo root (/workspace/mpr-ivl). Called by queue_internvl25_rel6.sh.
#   bash followup_rel_6lang/ivl25_rel6/run_rel6_ivl25.sh smoke|full
# Rerunning the same command resumes; a stage whose gate already passed is skipped.
# Optional env: SUBSET=all|new_only (default all; new_only = rows with fr or ja, 29,148 rows / 26,057 inferences, separate output folder),
#   EXPECT_FULL_N / EXPECT_FULL_ROWS (override the expected counts), REL_LIMIT_S (timeout of the full step in seconds: if the smoke-based ETA is longer
#   the step is NOT started, exit 3), IGNORE_ETA=1, LOCAL_MOCK=1 (local tests only: no GPU / no model snapshot, adds --mock-model).
set -Eeuo pipefail
STAGE="${1:?smoke or full}"
KEY=internvl25_8b
SUBSET="${SUBSET:-all}"
case "$SUBSET" in
  all) N_FULL="${EXPECT_FULL_N:-44489}"; ROWS_FULL="${EXPECT_FULL_ROWS:-52572}"; OUTNAME=rel6_${KEY} ;;
  new_only) N_FULL="${EXPECT_FULL_N:-26057}"; ROWS_FULL="${EXPECT_FULL_ROWS:-29148}"; OUTNAME=rel6new_${KEY} ;;
  *) echo "SUBSET must be all or new_only"; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
X=followup_rel_6lang/ivl25_rel6
DATA=followup_rel_6lang
LEGACY=followup_rel_4lang
CONFIG=$X/internvl2_5_8b.yaml
REFERENCE=$X/reference_contract_internvl25_rel_20261005.json   # InternVL2.5-8B 4-language REL run contract: model config and system prompt must equal it
OUT=$DATA/results/$OUTNAME      # never under followup_rel_4lang/results: no earlier result can be touched
export CONFIG REFERENCE SUBSET
mkdir -p "$OUT/logs"
MOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && MOCK="--mock-model"
status() { echo "$(date -u +%FT%TZ) [$KEY rel6/$SUBSET] $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at stage ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
passed() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }
RUN=(python "$DATA/run_inference_6lang.py" --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size 1 --languages-subset "$SUBSET")

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
print(f"REL ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs step timeout {limit / 3600:.1f} h")
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then status "STOPPED: $msg. The step is not started (it could not finish in time). IGNORE_ETA=1 overrides."; return 3; fi
  status "ETA check ok: $msg"; return 0
}

preflight() {
  CURRENT=preflight
  (cd "$X" && sha256sum -c SHA256SUMS) > "$OUT/logs/addon_verify.log"
  # the config in the addon must be byte-identical to the one in the repo (= the one of the 2026-10-05 REL run)
  cmp "$CONFIG" 01_code/configs/models/internvl2_5_8b.yaml
  [ -z "$MOCK" ] && python - <<'PY'
import importlib, os
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
  # model config and system prompt must equal the 2026-10-05 InternVL2.5 REL contract (same prompt, same model settings)
  python - <<'PY'
import json, os
from pathlib import Path
import yaml
from mpr_crosslocale.inference.system_prompts import build_system_prompt
ref = json.loads(Path(os.environ["REFERENCE"]).read_text())
cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
assert cfg == ref["model_config"], "model config differs from the 2026-10-05 REL contract"
system_prompt, _, template = build_system_prompt("english_fixed", "en")
assert system_prompt == ref["system_prompt"] and template == ref["prompt_template_version"], "system prompt differs from the REL contract"
print("model config and system prompt equal the 2026-10-05 REL contract")
PY
  # 6-language data: sha256 of every data file and of every screenshot (fr/ja in followup_rel_6lang, en/zh/th/ru in followup_rel_4lang)
  python "$DATA/data_bundle_6lang.py" verify --data-root "$DATA" --legacy-data-root "$LEGACY" > "$OUT/logs/bundle_verify.log"
  # the en/zh/th/ru rows must equal the 4-language file used on this server (object_extraction metadata aside), when that file is present
  if [ -s "$LEGACY/data/rel_all_366/conditions_4lang.jsonl" ]; then
    python "$DATA/data_bundle_6lang.py" check-legacy-subset --data-root "$DATA" --legacy-data-root "$LEGACY" \
      --legacy-conditions "$LEGACY/data/rel_all_366/conditions_4lang.jsonl" > "$OUT/logs/legacy_subset_check.log"
  else
    echo "4-language conditions file not found; subset check skipped" > "$OUT/logs/legacy_subset_check.log"
  fi
  "${RUN[@]}" --dry-run --output-dir "$OUT/dry_run" > "$OUT/logs/dry_run.log"
  python - "$OUT/dry_run/dry_run_plan.json" "$N_FULL" "$ROWS_FULL" "$SUBSET" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
n, rows, subset = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert p["actual_inferences"] == n and p["condition_rows"] == rows and p["items"] == 366 and p["batch_size"] == 1, p
assert p["languages_subset"] == subset and p["languages"] == ["en", "zh", "th", "ru", "fr", "ja"], p
print("plan ok:", p["actual_inferences"], "inferences,", p["condition_rows"], "condition rows, fingerprint", p["experiment_fingerprint"][:12])
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
  sha256sum "$CONFIG" "$X"/*.py "$X"/run_rel6_ivl25.sh "$DATA"/*.py "$DATA/bundle_manifest.json" "$DATA/data/rel_all_366/conditions_6lang.jsonl" > "$OUT/meta/addon_hashes.txt" || true
}

run_smoke() {
  CURRENT=smoke
  passed "$OUT/smoke/gate_rel_smoke.json" || {
    status "smoke started (--max-items 2)"
    "${RUN[@]}" --max-items 2 --resume $MOCK --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_gate
    python "$X/check_rel6_gate.py" rel smoke "$OUT/smoke" --model "$KEY" --full-n "$N_FULL" > "$OUT/logs/smoke_gate.log"
  }
  meta
  status "smoke passed"
}

run_full() {
  CURRENT=full
  passed "$OUT/full/gate_rel_complete.json" && { status "REL6 full already done"; return 0; }
  eta_guard "$OUT/smoke/gate_rel_smoke.json" "${REL_LIMIT_S:-}" "$OUT/full/predictions.jsonl" "$N_FULL" || exit 3
  status "REL6 full started ($N_FULL inferences, $ROWS_FULL condition rows)"
  "${RUN[@]}" --resume $MOCK --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=full_gate
  python "$X/check_rel6_gate.py" rel complete "$OUT/full" "$N_FULL" --model "$KEY" --full-n "$N_FULL" --full-rows "$ROWS_FULL" > "$OUT/logs/complete_gate.log"
  status "REL6 full done"
}

case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  *) echo "Unknown stage: $STAGE (smoke or full)"; exit 2 ;;
esac
