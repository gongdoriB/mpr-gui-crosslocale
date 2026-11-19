#!/usr/bin/env bash
# Experiment A (O, T, R) for the NEW directed language pairs that involve fr or ja, Qwen/Qwen3-VL-8B-Instruct, ONE RTX 4090.
#   bash followup_rel_6lang/q3_frja/run_expA6_qwen3.sh smoke|full|retry
# smoke : --max-items 3 (3 items x 18 pairs x 3 conditions = 162 inferences) + gate
# full  : T_gemma, split all (3,060 pairs x 3 = 9,180 inferences; EXPA_SPLIT=main -> 2,844 pairs = 8,532) + gate (O/R answers are compared with the REL new_only run when present)
# retry : sensitivity run with T_gemma_retry for the pairs whose question changed (159 pairs x 3 = 477 inferences) + gate
# Env: EXPA_BATCH (1 or 3, default 3 = the earlier 4-language Exp A run of this model; 3 = one pair per batch), EXPA_SPLIT (all|main|pilot, default all),
#   EXPA_LIMIT_S (timeout of the full step in seconds for the ETA check), IGNORE_ETA=1, LOCAL_MOCK=1 (local tests only).
set -Eeuo pipefail
STAGE="${1:?smoke, full or retry}"
KEY=qwen3_vl_8b
BATCH="${EXPA_BATCH:-3}"
SPLIT="${EXPA_SPLIT:-all}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/expA6_${KEY}"
TAG="$KEY expA6/b$BATCH"
source followup_rel_6lang/q3_frja/common_q3.sh
case "$SPLIT" in all) PAIRS=3060; RPAIRS=159 ;; main) PAIRS=2844; RPAIRS=148 ;; pilot) PAIRS=216; RPAIRS=11 ;; *) echo "EXPA_SPLIT must be all, main or pilot"; exit 2 ;; esac
N_FULL=$((PAIRS * 3)); N_RETRY=$((RPAIRS * 3))
REL_RESULTS=followup_rel_6lang/results/rel6new_${KEY}/full/results.jsonl
EXA=(python "$X/expA6/run_rel_T_6lang.py" --batch-size "$BATCH" --model-config "$CONFIG" --split "$SPLIT")

preflight() {
  CURRENT=preflight
  preflight_common
  for v in main retry; do
    "${EXA[@]}" --variant "$v" --output-dir "$OUT/dry_run_$v" --dry-run > "$OUT/logs/dry_run_$v.log"
  done
  python - "$OUT" "$N_FULL" "$N_RETRY" "$BATCH" <<'PY'
import json, sys
out, n, nr, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m, r = json.load(open(f"{out}/dry_run_main/plan.json")), json.load(open(f"{out}/dry_run_retry/plan.json"))
assert m["inferences"] == n and m["by_condition"] == {c: n // 3 for c in ("O", "T", "R")} and m["batch_size"] == batch, m
assert r["inferences"] == nr and r["by_condition"] == {c: nr // 3 for c in ("O", "T_retry", "R")} and r["batch_size"] == batch, r
print("plans ok: main", m["inferences"], "inferences", m["pairs"], "pairs, T=O pairs", m["T_equals_O_pairs"], ", T=R pairs", m["T_equals_R_pairs"],
      "| retry", r["inferences"], "inferences", r["pairs"], "pairs | batch", batch)
PY
  status "preflight passed (Exp A batch size $BATCH, split $SPLIT)"
}

run_smoke() {
  CURRENT=smoke
  passed "$OUT/smoke/gate_smoke.json" || {
    status "smoke started (--max-items 3)"
    "${EXA[@]}" --variant main --max-items 3 --resume $MOCK --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_gate
    python "$X/expA6/check_T6_gate.py" main smoke "$OUT/smoke" --batch "$BATCH" --full-n "$N_FULL" $GMOCK > "$OUT/logs/smoke_gate.log"
  }
  meta
  status "smoke passed"
}

run_full() {
  CURRENT=full
  passed "$OUT/full/gate_complete.json" && { status "Exp A fr/ja full already done"; return 0; }
  eta_guard "$OUT/smoke/gate_smoke.json" "${EXPA_LIMIT_S:-}" "$OUT/full/predictions.jsonl" "$N_FULL" || exit 3
  status "Exp A fr/ja full started ($N_FULL inferences)"
  "${EXA[@]}" --variant main --resume $MOCK --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=full_gate
  python "$X/expA6/check_T6_gate.py" main complete "$OUT/full" --batch "$BATCH" --expected-rows "$N_FULL" --full-n "$N_FULL" --rel-results "$REL_RESULTS" $GMOCK > "$OUT/logs/complete_gate.log"
  status "Exp A fr/ja full done"
}

run_retry() {
  CURRENT=retry
  passed "$OUT/retry/gate_complete.json" && { status "Exp A T_retry already done"; return 0; }
  status "Exp A T_retry started ($N_RETRY inferences)"
  "${EXA[@]}" --variant retry --resume $MOCK --output-dir "$OUT/retry" 2>&1 | tee -a "$OUT/logs/retry.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=retry_gate
  python "$X/expA6/check_T6_gate.py" retry complete "$OUT/retry" --batch "$BATCH" --expected-rows "$N_RETRY" --full-n "$N_RETRY" --rel-results "$REL_RESULTS" $GMOCK > "$OUT/logs/retry_gate.log"
  status "Exp A T_retry done"
}

case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  retry) preflight; run_retry ;;
  *) echo "Unknown stage: $STAGE (smoke, full or retry)"; exit 2 ;;
esac
