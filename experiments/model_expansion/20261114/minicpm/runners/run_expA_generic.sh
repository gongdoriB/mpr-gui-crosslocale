#!/usr/bin/env bash
# Experiment A (O, T, R) for a model WITHOUT an earlier R_stem/R_opts run (generic mode of
# run_rel_T.py / check_T_gate.py): no anchor gate, batch 1. Run from the repo root of the
# model's own setup, after minicpm_env.sh style env (MODEL_CONFIG, CFG_SHA, model snapshot).
#   bash followup_rel_4lang/expA/run_expA_generic.sh minicpm smoke   # 3 items = 108 inferences + gate
#   bash followup_rel_4lang/expA/run_expA_generic.sh minicpm full    # 5,688 inferences + gate
# Gate (smoke and complete): every row succeeded; parse failures are reported and fail only if >50% unparsed; plus the sanity rules of
# check_T_gate.py generic (identical labels, median A-D mass < 0.5, visual tokens outside 576..640). The smoke gate json also holds the ETA.
set -Eeuo pipefail
NAME="${1:?Give a short model name (minicpm)}"
STAGE="${2:?Give smoke or full}"
cd "$(dirname "$0")/../.."
[ "$NAME" = minicpm ] || { echo "Only minicpm is wired (env file minicpm_env.sh)"; exit 1; }
source .venv/bin/activate
source followup_rel_4lang/minicpm/minicpm_env.sh
OUT=followup_rel_4lang/results/expA_${NAME}
mkdir -p "$OUT/logs"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
RUN=(python followup_rel_4lang/expA/run_rel_T.py --model generic --batch-size 1 --model-config "$MODEL_CONFIG")

preflight() {
  CURRENT=preflight
  (cd followup_rel_4lang/expA && sha256sum -c expA_inputs/EXPA_SHA256SUMS) > "$OUT/logs/addon_verify.log"
  python - <<'PY'
import hashlib, os
from pathlib import Path
assert hashlib.sha256(Path(os.environ["MODEL_CONFIG"]).read_bytes()).hexdigest() == os.environ["CFG_SHA"], "Model config changed"
import torch
name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and "4090" in name, name
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MINICPM_MODEL_ID"], revision=os.environ["MINICPM_REVISION"], local_files_only=True)
assert Path(snap, "config.json").is_file(), snap
print("GPU", name, "| torch", torch.__version__, "| model", snap)
PY
  python followup_rel_4lang/data_bundle.py verify > "$OUT/logs/bundle_verify.log"
  "${RUN[@]}" --output-dir "$OUT/dry_run" --dry-run > "$OUT/logs/dry_run.log"
  status "preflight passed"
}

case "$STAGE" in
  smoke)
    preflight
    CURRENT=smoke; status "smoke started (108 inferences)"
    "${RUN[@]}" --split main --max-items 3 --resume --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    CURRENT=smoke_gate
    python followup_rel_4lang/expA/check_T_gate.py generic smoke "$OUT/smoke" > "$OUT/logs/smoke_gate.log"
    status "smoke passed" ;;
  full)
    preflight
    CURRENT=full; status "full started (5,688 inferences)"
    "${RUN[@]}" --split main --resume --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
    CURRENT=complete_gate
    python followup_rel_4lang/expA/check_T_gate.py generic complete "$OUT/full" > "$OUT/logs/complete_gate.log"
    status "full done" ;;
  *) echo "Unknown stage: $STAGE"; exit 1 ;;
esac
