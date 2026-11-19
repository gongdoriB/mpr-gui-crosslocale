#!/usr/bin/env bash
# Consistency check: re-runs 600 EARLIER MiniCPM-V-4.5 4-language REL canonical inferences (fixed seed 20261116, stratified, consistency_subset_ids.json; the same 600 ids the Qwen3 and
# InternVL checks use) on THIS server with the same config and compares the answers with the earlier MiniCPM run (2026-11-16, batch size 1). Report only: writes
# consistency_agreement.json, never stops the queue. Derived from q3_frja/run_consistency_qwen3.sh (sha ...). Results: followup_rel_6lang/results/consistency_minicpm_v45/.
#   bash followup_rel_6lang/mc6/run_consistency_mc.sh run
# Env: LOCAL_MOCK=1 (local tests only; the comparison is then not meaningful).
set -Eeuo pipefail
STAGE="${1:?run}"
BATCH=1
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/consistency_minicpm_v45"
TAG="minicpm_v45 consistency/b$BATCH"
source followup_rel_6lang/mc6/common_mc6.sh
RUN=(python "$X/consistency_run.py" --ids "$X/consistency_subset_ids.json" -- --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size "$BATCH" --languages-subset all)
N=600
GATE_FAILED=0

do_run() {
  CURRENT=preflight
  preflight_common
  python - "$X/consistency_subset_ids.json" "$N" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n"] == len(d["ids"]) == len(set(d["ids"])) == int(sys.argv[2]), d["n"]
assert all(i.startswith("rel_orcf::") for i in d["ids"]) and d["seed"] == 20261116 and d["source_batch_size"] == 1
print("subset ok:", d["n"], "ids,", d["cells"], "cells, by condition", d["by_condition"])
PY
  "${RUN[@]}" --dry-run --output-dir "$OUT/dry_run" > "$OUT/logs/dry_run.log"
  python - "$OUT/dry_run/dry_run_plan.json" "$N" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["actual_inferences"] == p["condition_rows"] == int(sys.argv[2]) and p["reused_condition_rows"] == 0, p
print("consistency plan ok:", p["actual_inferences"], "inferences, fingerprint", p["experiment_fingerprint"][:12])
PY
  CURRENT=run
  passed "$OUT/run/gate_cc_complete.json" || {
    status "consistency run started ($N inferences, batch $BATCH)"
    "${RUN[@]}" --resume $MOCK --output-dir "$OUT/run" 2>&1 | tee -a "$OUT/logs/run.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=gate
    if python "$X/check_rel6_gate.py" rel complete "$OUT/run" "$N" --model "$KEY" --full-n "$N" --full-rows "$N" > "$OUT/logs/gate.log"; then
      cp "$OUT/run/gate_rel_complete.json" "$OUT/run/gate_cc_complete.json"
    else
      GATE_FAILED=1; status "consistency gate FAILED (see $OUT/logs/gate.log); the comparison is still written"
    fi
  }
  CURRENT=compare
  python "$X/consistency_compare.py" --run "$OUT/run" --reference "$X/consistency_reference.jsonl" --ids "$X/consistency_subset_ids.json" --out "$OUT/consistency_agreement.json" | tee "$OUT/logs/compare.log"
  meta
  status "consistency check done: $(python -c "import json;d=json.load(open('$OUT/consistency_agreement.json'))['overall'];print('label agreement',d['label_agreement'],'(',d['same_label'],'of',$N,') accuracy old',d['old_accuracy'],'new',d['new_accuracy'])")"
  [ "$GATE_FAILED" = 0 ]   # a failed gate makes this step exit non-zero (non-fatal in the queue) after the report was written
}

case "$STAGE" in
  run) do_run ;;
  *) echo "Unknown stage: $STAGE (run)"; exit 2 ;;
esac
