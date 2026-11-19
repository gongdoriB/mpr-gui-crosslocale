# Sourced by run_rel6_b2.sh and run_expA6_b2.sh (not run on its own). Caller sets KEY (internvl25_8b | internvl35_8b | uitars_15_7b), OUT, TAG; cwd = repository root.
# Optional env: LOCAL_MOCK=1 (local tests only: no GPU / no model snapshot / no package pins, adds --mock-model), IGNORE_ETA=1, EXPECT_TORCH, EXPECT_SKIP (comma list of packages not compared; tests only).
X=followup_rel_6lang/b2_frja
DATA=followup_rel_6lang
LEGACY=followup_rel_4lang
case "$KEY" in
  internvl25_8b) CONFIG=$X/models/internvl2_5_8b.yaml;        CFG_SHA=c78cd7ea609667239f278040f4fa484db1b20c295f719833e5b7efea19c12b8b; REPO_CFG=01_code/configs/models/internvl2_5_8b.yaml
                 REL_REF=""; EXPA_REF=$X/contracts/ref_internvl25_expA_20261114.json; CODE_REF=$EXPA_REF; STRICT_REL=() ;;
  internvl35_8b) CONFIG=$X/models/internvl3_5_8b_4090.yaml;   CFG_SHA=7fed517cf7905dd8158487e25754b716c48c31c55b91469381d0332e4beebf03; REPO_CFG=01_code/configs/models/internvl3_5_8b_4090.yaml
                 REL_REF=$X/contracts/ref_internvl35_rel_20261109.json; EXPA_REF=$X/contracts/ref_internvl35_expA_20261114.json; CODE_REF=$REL_REF; STRICT_REL=() ;;      # earlier REL + Exp A: strict_batch_size false
  uitars_15_7b)  CONFIG=$X/models/uitars_1_5_7b_4090.yaml;    CFG_SHA=e16c3fc0b810fef837c4ca1f9c27099cabf93c53b504bb2d6d337fd18fb95d93; REPO_CFG=01_code/configs/models/uitars_1_5_7b_4090.yaml
                 REL_REF=$X/contracts/ref_uitars_rel_20261114.json; EXPA_REF=$X/contracts/ref_uitars_expA_20261114.json; CODE_REF=$REL_REF; STRICT_REL=(--strict-batch-size) ;; # earlier REL + Exp A: strict_batch_size true
  *) echo "unknown model key $KEY"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
export CONFIG CFG_SHA REPO_CFG EXPA_REF REL_REF CODE_REF KEY
mkdir -p "$OUT/logs"
MOCK=""; GMOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && { MOCK="--mock-model"; GMOCK="--mock"; }
status() { echo "$(date -u +%FT%TZ) [$TAG] $*" | tee -a "$OUT/STATUS"; }
trap 'status "FAILED at stage ${CURRENT:-?} (exit $?). See $OUT/logs/"' ERR
passed() { [ -s "$1" ] && grep -q '"status": "passed"' "$1"; }

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

preflight_common() {
  (cd "$X" && sha256sum -c SHA256SUMS) > "$OUT/logs/addon_verify.log"
  [ "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CFG_SHA" ]
  # the repo's copy of the config (when it has one) must be byte-identical to the addon copy (= the one of the earlier 4-language runs)
  [ ! -f "$REPO_CFG" ] || cmp "$CONFIG" "$REPO_CFG"
  python "$X/env_check.py"
  python "$DATA/data_bundle_6lang.py" verify --data-root "$DATA" --legacy-data-root "$LEGACY" > "$OUT/logs/bundle_verify.log"
  if [ -s "$LEGACY/data/rel_all_366/conditions_4lang.jsonl" ]; then
    python "$DATA/data_bundle_6lang.py" check-legacy-subset --data-root "$DATA" --legacy-data-root "$LEGACY" \
      --legacy-conditions "$LEGACY/data/rel_all_366/conditions_4lang.jsonl" > "$OUT/logs/legacy_subset_check.log"
  else
    echo "4-language conditions file not found; subset check skipped" > "$OUT/logs/legacy_subset_check.log"
  fi
}

meta() {
  mkdir -p "$OUT/meta"
  cp "$CONFIG" "$OUT/meta/"
  git rev-parse HEAD > "$OUT/meta/git_commit.txt" 2>&1 || true
  git status --short > "$OUT/meta/git_status.txt" 2>&1 || true
  python -m pip freeze > "$OUT/meta/requirements.freeze.txt" || true
  nvidia-smi > "$OUT/meta/nvidia_smi.txt" || true
  sha256sum "$CONFIG" "$X"/*.py "$X"/*.sh "$X"/expA6/*.py "$DATA"/*.py "$DATA/bundle_manifest.json" "$DATA/data/rel_all_366/conditions_6lang.jsonl" > "$OUT/meta/addon_hashes.txt" || true
}
