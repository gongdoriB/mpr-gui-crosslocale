#!/usr/bin/env bash
# R_stem / R_opts 6-language run of ONE model on ONE RTX 4090 (called by queue_rstem6_c.sh; can be run by hand from the repository root).
#   bash followup_rel_6lang/rstem6/run_rstem6.sh <KEY> smoke|full
# KEY: qwen3_vl_8b | minicpm_v45 | internvl35_8b | uitars_15_7b | internvl25_8b | qwen3_vl_4b | internvl35_4b (4B keys added 2026-11-19)
# smoke: --max-items 2 = 2 items x 30 language pairs x 4 conditions = 240 inferences, then the smoke gate.  full: 18,960 inferences (4,740 pairs x 4), --resume, then the completion gate.
# Result folder: followup_rel_6lang/results/rstem6_<KEY>/{smoke,full}.  Env: EXPECT_TORCH (tests), EXPECT_SKIP (tests), IGNORE_ETA=1, FULL_LIMIT_S (ETA check against the step timeout), LOCAL_MOCK=1 (tests: mock model, no GPU).
set -Eeuo pipefail
KEY="${1:?model key}"; STAGE="${2:?smoke or full}"
cd "$(dirname "$0")/../.."
X=followup_rel_6lang/rstem6
OUT="$PWD/followup_rel_6lang/results/rstem6_$KEY"
TAG="rstem6 $KEY"
case "$KEY" in
  qwen3_vl_8b)   CONFIG=$X/models/qwen3_vl_8b_4090_channels_last.yaml; CFG_SHA=2775bb6b981d35922c46545ef58af0ac8e8ded1a3615902c73ac141609adbc8b; REPO_CFG=01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml; CONTRACT=$X/contracts/ref_qwen3_rel_20261107.json;   BATCH=4 ;;
  minicpm_v45)   CONFIG=$X/models/minicpm_v4_5_8b_4090.yaml;          CFG_SHA=298953fe68ae62368e1456b9a61766988058617f9cdab22c4eec296c236bb775; REPO_CFG=01_code/configs/models/minicpm_v4_5_8b_4090.yaml;          CONTRACT=$X/contracts/ref_minicpm_rel_20261116.json;      BATCH=1 ;;
  internvl35_8b) CONFIG=$X/models/internvl3_5_8b_4090.yaml;           CFG_SHA=7fed517cf7905dd8158487e25754b716c48c31c55b91469381d0332e4beebf03; REPO_CFG=01_code/configs/models/internvl3_5_8b_4090.yaml;           CONTRACT=$X/contracts/ref_internvl35_rel_20261109.json;   BATCH=1 ;;
  uitars_15_7b)  CONFIG=$X/models/uitars_1_5_7b_4090.yaml;            CFG_SHA=e16c3fc0b810fef837c4ca1f9c27099cabf93c53b504bb2d6d337fd18fb95d93; REPO_CFG=01_code/configs/models/uitars_1_5_7b_4090.yaml;            CONTRACT=$X/contracts/ref_uitars_rel_20261114.json;       BATCH=1 ;;
  internvl25_8b) CONFIG=$X/models/internvl2_5_8b.yaml;                CFG_SHA=c78cd7ea609667239f278040f4fa484db1b20c295f719833e5b7efea19c12b8b; REPO_CFG=01_code/configs/models/internvl2_5_8b.yaml;                CONTRACT=$X/contracts/ref_internvl25_expA_20261114.json;  BATCH=1 ;;
  qwen3_vl_4b)   CONFIG=$X/models/qwen3_vl_4b_4090_channels_last.yaml; CFG_SHA=4a3de63bc0036755c7e4633be03544b3320fac0388c1fbc897aa50d2ec167544; REPO_CFG=01_code/configs/models/qwen3_vl_4b_4090_channels_last.yaml; CONTRACT=$X/contracts/ref_qwen3_4b_expA_20261117.json;     BATCH=1 ;;
  internvl35_4b) CONFIG=$X/models/internvl3_5_4b_4090.yaml;           CFG_SHA=6a3158fc8cac7214fe6edb9febbd64165274281513bcbb75aa24aa89fb3dc6e8; REPO_CFG=01_code/configs/models/internvl3_5_4b_4090.yaml;           CONTRACT=$X/contracts/ref_internvl35_4b_expA_20261117.json; BATCH=1 ;;
  *) echo "unknown model key $KEY"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$PWD/.cache" HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
export KEY CONFIG CONTRACT
mkdir -p "$OUT/logs"
MOCK=""; GMOCK=""; PRE=""
[ "${LOCAL_MOCK:-0}" = 1 ] && { MOCK="--mock-model"; GMOCK="--mock"; }
OR_SRC=(); [ -n "${OR_SOURCE:-}" ] && OR_SRC=(--or-source "$OR_SOURCE")
status() { echo "$(date -u +%FT%TZ) [$TAG] $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at stage ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
passed() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }
RUN=(python "$X/run_rstem6.py" --model "$KEY" --model-config "$CONFIG" --batch-size "$BATCH" ${OR_SRC[@]+"${OR_SRC[@]}"})

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
print(f"ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs step timeout {limit / 3600:.1f} h")
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then status "STOPPED: $msg. The step is not started (it could not finish in time). IGNORE_ETA=1 overrides."; return 3; fi
  status "ETA check ok: $msg"; return 0
}

preflight() {
  CURRENT=preflight
  (cd "$X" && sha256sum -c SHA256SUMS) > "$OUT/logs/addon_verify.log"
  [ "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CFG_SHA" ]
  [ ! -f "$REPO_CFG" ] || cmp "$CONFIG" "$REPO_CFG"          # the repository's copy (if any) equals the addon copy
  python "$X/env_check_rstem6.py"
  "${RUN[@]}" --output-dir "$OUT/dry_run" --dry-run > "$OUT/logs/dry_run.log"
  python - "$OUT" "$BATCH" <<'PY'
import json, sys
out, batch = sys.argv[1], int(sys.argv[2])
p = json.load(open(f"{out}/dry_run/plan.json"))
assert p["inferences"] == 18960 and p["pairs"] == 4740 and p["batch_size"] == batch, p
assert p["by_condition"] == {c: 4740 for c in ("O", "R_stem", "R_opts", "R")}, p
print("plan ok:", p["inferences"], "inferences", p["pairs"], "pairs, batch", batch, "strict", p["strict_batch_size"], "fingerprint", p["experiment_fingerprint"][:12])
PY
  status "preflight passed"
}
meta() {
  mkdir -p "$OUT/meta"
  cp "$CONFIG" "$CONTRACT" "$OUT/meta/"
  git rev-parse HEAD > "$OUT/meta/git_commit.txt" 2>&1 || true
  git status --short > "$OUT/meta/git_status.txt" 2>&1 || true
  python -m pip freeze > "$OUT/meta/requirements.freeze.txt" || true
  nvidia-smi > "$OUT/meta/nvidia_smi.txt" || true
  sha256sum "$X"/*.py "$X"/*.sh "$X"/inputs/* "$X"/SHA256SUMS "$CONFIG" > "$OUT/meta/addon_hashes.txt" || true
}
run_smoke() {
  CURRENT=smoke
  passed "$OUT/smoke/gate_smoke.json" || {
    status "smoke started (--max-items 2 = 240 inferences)"
    "${RUN[@]}" --max-items 2 --resume $MOCK --output-dir "$OUT/smoke" 2>&1 | tee -a "$OUT/logs/smoke.log"
    [ "${PIPESTATUS[0]}" = 0 ]
    CURRENT=smoke_gate
    python "$X/check_rstem6_gate.py" smoke "$OUT/smoke" --model "$KEY" --batch "$BATCH" --full-n 18960 $GMOCK > "$OUT/logs/smoke_gate.log"
  }
  meta
  status "smoke passed"
}
run_full() {
  CURRENT=full
  passed "$OUT/full/gate_complete.json" && { status "full already done"; return 0; }
  eta_guard "$OUT/smoke/gate_smoke.json" "${FULL_LIMIT_S:-}" "$OUT/full/predictions.jsonl" 18960 || exit 3
  status "full started (18,960 inferences)"
  "${RUN[@]}" --resume $MOCK --output-dir "$OUT/full" 2>&1 | tee -a "$OUT/logs/full.log"
  [ "${PIPESTATUS[0]}" = 0 ]
  CURRENT=full_gate
  python "$X/check_rstem6_gate.py" complete "$OUT/full" --model "$KEY" --batch "$BATCH" --expected-rows 18960 --full-n 18960 $GMOCK > "$OUT/logs/complete_gate.log"
  status "full done"
}
case "$STAGE" in
  smoke) preflight; run_smoke ;;
  full) preflight; run_full ;;
  *) echo "Unknown stage: $STAGE (smoke or full)"; exit 2 ;;
esac
