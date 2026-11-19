#!/usr/bin/env bash
# REL O/R/C/F rows that involve fr or ja (new_only: 29,148 condition rows -> 26,057 canonical inferences) for Qwen/Qwen3-VL-8B-Instruct on ONE RTX 4090.
# Run from anywhere; it changes to the repo root (/workspace/mpr-gui-crosslocale). Called by queue_qwen3_8b_frja.sh.
#   bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh smoke|full
# Rerunning the same command resumes; a stage whose gate already passed is skipped.
# Env: REL_BATCH (default 1; the earlier 4-language REL run of this model used 4 with --strict-batch-size), REL_LIMIT_S (timeout of the full step in seconds: if the
#   smoke-based ETA is longer the step is NOT started, exit 3), IGNORE_ETA=1, EXPECT_FULL_N / EXPECT_FULL_ROWS, LOCAL_MOCK=1 (local tests only).
set -Eeuo pipefail
STAGE="${1:?smoke or full}"
KEY=qwen3_vl_8b
SUBSET=new_only
N_FULL="${EXPECT_FULL_N:-26057}"; ROWS_FULL="${EXPECT_FULL_ROWS:-29148}"
BATCH="${REL_BATCH:-1}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/rel6new_${KEY}"      # never under followup_rel_4lang/results: no earlier result can be touched
TAG="$KEY rel6/new_only/b$BATCH"
source followup_rel_6lang/q3_frja/common_q3.sh
STRICT=(); [ "$BATCH" -gt 1 ] && STRICT=(--strict-batch-size)
RUN=(python "$DATA/run_inference_6lang.py" --model-config "$CONFIG" --data-root "$DATA" --legacy-data-root "$LEGACY" --batch-size "$BATCH" ${STRICT[@]+"${STRICT[@]}"} --languages-subset "$SUBSET")

preflight() {
  CURRENT=preflight
  preflight_common
  "${RUN[@]}" --dry-run --output-dir "$OUT/dry_run" > "$OUT/logs/dry_run.log"
  python - "$OUT/dry_run/dry_run_plan.json" "$N_FULL" "$ROWS_FULL" "$SUBSET" "$BATCH" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
n, rows, subset, batch = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
assert p["actual_inferences"] == n and p["condition_rows"] == rows and p["items"] == 366 and p["batch_size"] == batch, p
assert p["languages_subset"] == subset and p["languages"] == ["en", "zh", "th", "ru", "fr", "ja"], p
print("plan ok:", p["actual_inferences"], "inferences,", p["condition_rows"], "condition rows, batch", batch, ", fingerprint", p["experiment_fingerprint"][:12])
PY
  status "preflight passed (REL batch size $BATCH)"
}

run_smoke() {
  CURRENT=smoke
  passed "$OUT/smoke/gate_rel_smoke.json" || {
    status "smoke started (--max-items 2)"
    "${RUN[@]}" --max-items 2 --resume $MOCK --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_gate
    python "$X/check_q3_gate.py" rel smoke "$OUT/smoke" --model "$KEY" --batch "$BATCH" --full-n "$N_FULL" $GMOCK > "$OUT/logs/smoke_gate.log"
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
  python "$X/check_q3_gate.py" rel complete "$OUT/full" "$N_FULL" --model "$KEY" --batch "$BATCH" --full-n "$N_FULL" --full-rows "$ROWS_FULL" $GMOCK > "$OUT/logs/complete_gate.log"
  status "REL fr/ja full done"
}

case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  *) echo "Unknown stage: $STAGE (smoke or full)"; exit 2 ;;
esac
