#!/usr/bin/env bash
# Consistency check: re-runs 600 OLD 4-language REL canonical inferences (fixed seed, stratified, consistency_subset_ids.json) of Qwen3-VL-8B on THIS server with the
# same config and compares the answers with the earlier 4090 run (batch size 4 there). Report only: writes consistency_agreement.json, never stops the queue.
# Results go to followup_rel_6lang/results/consistency_qwen3_vl_8b/ (separate from the fr/ja results).
#   bash followup_rel_6lang/q3_frja/run_consistency_qwen3.sh run
# Env: CC_BATCH (default = REL_BATCH, default 1), LOCAL_MOCK=1 (local tests only; the comparison is then not meaningful).
set -Eeuo pipefail
STAGE="${1:?run}"
KEY=qwen3_vl_8b
BATCH="${CC_BATCH:-${REL_BATCH:-1}}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/consistency_${KEY}"
TAG="$KEY consistency/b$BATCH"
source followup_rel_6lang/q3_frja/common_q3.sh
STRICT=(); [ "$BATCH" -gt 1 ] && STRICT=(--strict-batch-size)
RUN=(python "$X/consistency_run.py" --ids "$X/consistency_subset_ids.json" -- --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size "$BATCH" ${STRICT[@]+"${STRICT[@]}"} --languages-subset all)
N=600

do_run() {
  CURRENT=preflight
  preflight_common
  python - "$X/consistency_subset_ids.json" "$N" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n"] == len(d["ids"]) == len(set(d["ids"])) == int(sys.argv[2]), d["n"]
assert all(i.startswith("rel_orcf::") for i in d["ids"]) and d["seed"] == 20261116
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
    python "$X/check_q3_gate.py" cc complete "$OUT/run" "$N" --model "$KEY" --batch "$BATCH" --full-n "$N" --full-rows "$N" $GMOCK > "$OUT/logs/gate.log"
  }
  CURRENT=compare
  python "$X/consistency_compare.py" --run "$OUT/run" --reference "$X/consistency_reference.jsonl" --ids "$X/consistency_subset_ids.json" --out "$OUT/consistency_agreement.json" | tee "$OUT/logs/compare.log"
  meta
  status "consistency check done: $(python -c "import json;d=json.load(open('$OUT/consistency_agreement.json'))['overall'];print('label agreement',d['label_agreement'],'(',d['same_label'],'of',$N,') accuracy old',d['old_accuracy'],'new',d['new_accuracy'])")"
}

case "$STAGE" in
  run) do_run ;;
  *) echo "Unknown stage: $STAGE (run)"; exit 2 ;;
esac
