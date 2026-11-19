#!/usr/bin/env bash
# REL O/R/C/F rows that involve fr or ja (new_only: 29,148 condition rows -> 26,057 canonical inferences) for MiniCPM-V-4.5, batch 1, strict_batch_size false
# (= the earlier 4-language REL run of 2026-11-16). Derived from run_rel6_b2.sh (sha d1b51a57...). Result folder followup_rel_6lang/results/rel6new_minicpm_v45.
#   bash followup_rel_6lang/mc6/run_rel6_mc.sh smoke|full
# Env: REL_LIMIT_S (full step timeout seconds; ETA check, exit 3), IGNORE_ETA=1, LOCAL_MOCK=1 (tests only).
set -Eeuo pipefail
STAGE="${1:?smoke or full}"
N_FULL="${EXPECT_FULL_N:-26057}"; ROWS_FULL="${EXPECT_FULL_ROWS:-29148}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/rel6new_minicpm_v45"      # never under followup_rel_4lang/results
TAG="minicpm_v45 rel6/new_only/b1"
source followup_rel_6lang/mc6/common_mc6.sh
RUN=(python "$DATA/run_inference_6lang.py" --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size 1 --languages-subset new_only)
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
