#!/usr/bin/env bash
# MiniCPM-V-4.5 REL O/R/C/F on ONE RTX 4090, batch 1 (same call as the 2026-11-07/09 runs:
# followup_rel_4lang/run_inference.py --model-config ... --batch-size 1 --resume).
#   bash followup_rel_4lang/minicpm/run_rel_minicpm.sh smoke   # --max-items 2 (80 inferences) + gate
#   bash followup_rel_4lang/minicpm/run_rel_minicpm.sh full    # all 366 items (18,432 inferences) + gate
# Gates (check_minicpm_gates.py): all inferences succeeded, <= 50% unparsed, and the sanity rules (identical labels, median A-D mass < 0.5,
# visual tokens outside 576..640); the smoke gate json also holds the ETA. Rerunning the same command resumes. Progress: cat $OUT/STATUS
set -Eeuo pipefail
STAGE="${1:?Give smoke or full}"
cd "$(dirname "$0")/../.."
source .venv/bin/activate
source followup_rel_4lang/minicpm/minicpm_env.sh
OUT=followup_rel_4lang/results/minicpm_rel
mkdir -p "$OUT/logs"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR

preflight() {
  CURRENT=preflight
  python - <<'PY'
import hashlib, os, sys
from pathlib import Path
assert hashlib.sha256(Path(os.environ["MODEL_CONFIG"]).read_bytes()).hexdigest() == os.environ["CFG_SHA"], "Model config changed"
import torch
name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and "4090" in name, name
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MINICPM_MODEL_ID"], revision=os.environ["MINICPM_REVISION"], local_files_only=True)
assert Path(snap, "config.json").is_file() and Path(snap, "modeling_minicpmv.py").is_file(), snap
print("GPU", name, "| torch", torch.__version__, "| model", snap)
PY
  python followup_rel_4lang/data_bundle.py verify > "$OUT/logs/bundle_verify.log"
  python followup_rel_4lang/run_inference.py --model-config "$MODEL_CONFIG" --batch-size 1 \
    --dry-run --output-dir "$OUT/dry_run" > "$OUT/logs/dry_run.log"
  status "preflight passed"
}

case "$STAGE" in
  smoke)
    preflight
    CURRENT=smoke; status "smoke started (80 inferences)"
    python followup_rel_4lang/run_inference.py --model-config "$MODEL_CONFIG" --batch-size 1 \
      --max-items 2 --resume --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    CURRENT=smoke_gate
    python followup_rel_4lang/minicpm/check_minicpm_gates.py rel smoke "$OUT/smoke" > "$OUT/logs/smoke_gate.log"
    status "smoke passed" ;;
  full)
    preflight
    CURRENT=full; status "full started (18,432 inferences)"
    python followup_rel_4lang/run_inference.py --model-config "$MODEL_CONFIG" --batch-size 1 \
      --resume --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
    CURRENT=complete_gate
    python followup_rel_4lang/minicpm/check_minicpm_gates.py rel complete "$OUT/full" > "$OUT/logs/complete_gate.log"
    status "full done" ;;
  *) echo "Unknown stage: $STAGE"; exit 1 ;;
esac
