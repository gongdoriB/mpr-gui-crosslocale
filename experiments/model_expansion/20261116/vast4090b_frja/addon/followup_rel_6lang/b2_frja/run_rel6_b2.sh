#!/usr/bin/env bash
# REL O/R/C/F rows that involve fr or ja (new_only: 29,148 condition rows -> 26,057 canonical inferences), batch 1, for internvl35_8b or uitars_15_7b (earlier 4-language REL: batch 1;
# strict_batch_size false for internvl35_8b, true for uitars_15_7b). Uses the reviewed run_rel_followup_6lang.py unchanged. Result folder followup_rel_6lang/results/rel6new_<KEY>.
#   bash followup_rel_6lang/b2_frja/run_rel6_b2.sh <KEY> smoke|full
# Env: REL_LIMIT_S (full step timeout seconds; ETA check, exit 3), IGNORE_ETA=1, LOCAL_MOCK=1 (tests only).
set -Eeuo pipefail
KEY="${1:?model key}"; STAGE="${2:?smoke or full}"
case "$KEY" in internvl35_8b|uitars_15_7b) ;; *) echo "REL is run for internvl35_8b or uitars_15_7b only (InternVL2.5 REL6 was done by its own queue)"; exit 2 ;; esac
N_FULL="${EXPECT_FULL_N:-26057}"; ROWS_FULL="${EXPECT_FULL_ROWS:-29148}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/rel6new_${KEY}"      # never under followup_rel_4lang/results
TAG="$KEY rel6/new_only/b1"
source followup_rel_6lang/b2_frja/common_b2.sh
RUN=(python "$DATA/run_inference_6lang.py" --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size 1 ${STRICT_REL[@]+"${STRICT_REL[@]}"} --languages-subset new_only)
preflight() {
  CURRENT=preflight
  preflight_common
  "${RUN[@]}" --dry-run --output-dir "$OUT/dry_run" > "$OUT/logs/dry_run.log"
  python - "$OUT/dry_run/dry_run_plan.json" "$N_FULL" "$ROWS_FULL" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
n, rows = int(sys.argv[2]), int(sys.argv[3])
assert p["actual_inferences"] == n and p["condition_rows"] == rows and p["items"] == 366 and p["batch_size"] == 1, p
assert p["languages_subset"] == "new_only" and p["languages"] == ["en", "zh", "th", "ru", "fr", "ja"], p
print("plan ok:", p["actual_inferences"], "inferences,", p["condition_rows"], "condition rows, fingerprint", p["experiment_fingerprint"][:12])
PY
  status "preflight passed"
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
  passed "$OUT/full/gate_rel_complete.json" && { status "REL fr/ja full already done"; return 0; }
  eta_guard "$OUT/smoke/gate_rel_smoke.json" "${REL_LIMIT_S:-}" "$OUT/full/predictions.jsonl" "$N_FULL" || exit 3
  status "REL fr/ja full started ($N_FULL inferences, $ROWS_FULL condition rows)"
  "${RUN[@]}" --resume $MOCK --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=full_gate
  python "$X/check_rel6_gate.py" rel complete "$OUT/full" "$N_FULL" --model "$KEY" --full-n "$N_FULL" --full-rows "$ROWS_FULL" > "$OUT/logs/complete_gate.log"
  status "REL fr/ja full done"
}
case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  *) echo "Unknown stage: $STAGE (smoke or full)"; exit 2 ;;
esac
