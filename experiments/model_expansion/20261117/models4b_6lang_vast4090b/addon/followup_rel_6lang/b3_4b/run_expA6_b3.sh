#!/usr/bin/env bash
# Experiment A (O, T, R) 6-language, qwen3_vl_4b | internvl35_4b, ONE RTX 4090.
#   bash followup_rel_6lang/b3_4b/run_expA6_b3.sh <KEY> smoke|full|retry
# SCOPE=new (default): the new fr/ja pairs, split main: 2,844 pairs -> 8,532 inferences (retry: 148 pairs -> 444).
# SCOPE=all6: every main pair of the 6 languages (1,896 earlier 4-language pairs with the earlier T rows + 2,844 new) = 4,740 pairs -> 14,220 inferences (retry: 262 pairs -> 786). (not used by the 4B queue).
# smoke: --max-items 3 (new: 162 inferences; all6: 270) + gate. Env: EXPA_BATCH (default 1 = the earlier 4B runs; only 1 is accepted), EXPA_SPLIT (main default; all|pilot only with SCOPE=new),
#   EXPA_LIMIT_S (full step timeout for the ETA check), IGNORE_ETA=1, LOCAL_MOCK=1 (tests only).
set -Eeuo pipefail
KEY="${1:?model key}"; STAGE="${2:?smoke, full or retry}"
BATCH="${EXPA_BATCH:-1}"; SPLIT="${EXPA_SPLIT:-main}"; SCOPE="${SCOPE:-new}"
cd "$(dirname "$0")/../.."
OUT="$PWD/followup_rel_6lang/results/expA6_${KEY}"
TAG="$KEY expA6/$SCOPE/b$BATCH"
source followup_rel_6lang/b3_4b/common_b3.sh
case "$SCOPE:$SPLIT" in
  new:all) PAIRS=3060; RPAIRS=159; SMOKE_N=162 ;; new:main) PAIRS=2844; RPAIRS=148; SMOKE_N=162 ;; new:pilot) PAIRS=216; RPAIRS=11; SMOKE_N=162 ;;
  all6:main) PAIRS=4740; RPAIRS=262; SMOKE_N=270 ;;
  *) echo "unsupported SCOPE:SPLIT $SCOPE:$SPLIT"; exit 2 ;;
esac
N_FULL=$((PAIRS * 3)); N_RETRY=$((RPAIRS * 3))
# anchor: the O/R answers of the REL run of the same model on this server (reported; a problem below 0.90 at the complete stage)
REL_RESULTS=""
for d in rel6new_${KEY} rel6_${KEY}; do [ -s "followup_rel_6lang/results/$d/full/results.jsonl" ] && { REL_RESULTS="followup_rel_6lang/results/$d/full/results.jsonl"; break; }; done
EXA=(python "$X/expA6/run_rel_T_6lang_multi.py" --model "$KEY" --scope "$SCOPE" --batch-size "$BATCH" --model-config "$CONFIG" --split "$SPLIT")
GREL=(); [ -n "$REL_RESULTS" ] && GREL=(--rel-results "$REL_RESULTS")
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
  status "preflight passed (Exp A scope $SCOPE, batch $BATCH, split $SPLIT)"
}
run_smoke() {
  CURRENT=smoke
  passed "$OUT/smoke/gate_smoke.json" || {
    status "smoke started (--max-items 3)"
    "${EXA[@]}" --variant main --max-items 3 --resume $MOCK --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_gate
    python "$X/expA6/check_T6_gate_multi.py" main smoke "$OUT/smoke" --model "$KEY" --batch "$BATCH" --full-n "$N_FULL" $GMOCK > "$OUT/logs/smoke_gate.log"
  }
  meta
  status "smoke passed"
}
run_full() {
  CURRENT=full
  passed "$OUT/full/gate_complete.json" && { status "Exp A full already done"; return 0; }
  eta_guard "$OUT/smoke/gate_smoke.json" "${EXPA_LIMIT_S:-}" "$OUT/full/predictions.jsonl" "$N_FULL" || exit 3
  status "Exp A full started ($N_FULL inferences, scope $SCOPE)"
  "${EXA[@]}" --variant main --resume $MOCK --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=full_gate
  python "$X/expA6/check_T6_gate_multi.py" main complete "$OUT/full" --model "$KEY" --batch "$BATCH" --expected-rows "$N_FULL" --full-n "$N_FULL" ${GREL[@]+"${GREL[@]}"} $GMOCK > "$OUT/logs/complete_gate.log"
  status "Exp A full done"
}
run_retry() {
  CURRENT=retry
  passed "$OUT/retry/gate_complete.json" && { status "Exp A T_retry already done"; return 0; }
  status "Exp A T_retry started ($N_RETRY inferences)"
  "${EXA[@]}" --variant retry --resume $MOCK --output-dir "$OUT/retry" 2>&1 | tee -a "$OUT/logs/retry.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=retry_gate
  python "$X/expA6/check_T6_gate_multi.py" retry complete "$OUT/retry" --model "$KEY" --batch "$BATCH" --expected-rows "$N_RETRY" --full-n "$N_RETRY" ${GREL[@]+"${GREL[@]}"} $GMOCK > "$OUT/logs/retry_gate.log"
  status "Exp A T_retry done"
}
case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  retry) preflight; run_retry ;;
  *) echo "Unknown stage: $STAGE (smoke, full or retry)"; exit 2 ;;
esac
