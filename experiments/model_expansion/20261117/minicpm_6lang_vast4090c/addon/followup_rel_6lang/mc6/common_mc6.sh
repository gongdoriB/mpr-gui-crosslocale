# Sourced by run_rel6_mc.sh, run_expA6_mc.sh and run_consistency_mc.sh (not run on its own). Caller sets OUT (result folder) and TAG (status prefix); cwd = repository root (/workspace/mpr-minicpm).
# Derived from vast4090b_frja/.../b2_frja/common_b2.sh (sha 1e62fad9...): one model only (MiniCPM-V-4.5), references = run contracts of the earlier MiniCPM 4-language runs (2026-11-16).
# Optional env: LOCAL_MOCK=1 (local tests only: no GPU / no model snapshot / no package pins, adds --mock-model), IGNORE_ETA=1, EXPECT_TORCH, EXPECT_SKIP (comma list of packages not compared; tests only).
KEY=minicpm_v45
X=followup_rel_6lang/mc6
DATA=followup_rel_6lang
LEGACY=followup_rel_4lang
CONFIG=$X/models/minicpm_v4_5_8b_4090.yaml
CFG_SHA=298953fe68ae62368e1456b9a61766988058617f9cdab22c4eec296c236bb775      # sha256 of the config of the earlier REL (2026-11-16) and Exp A (2026-11-16) runs
REPO_CFG=01_code/configs/models/minicpm_v4_5_8b_4090.yaml                      # the copy the adapter patch puts into the repository
REL_REF=$X/contracts/ref_minicpm_rel_20261116.json                             # run_contract.json of the earlier REL run (batch 1, strict_batch_size false)
EXPA_REF=$X/contracts/ref_minicpm_expA_20261116.json                           # run_contract.json of the earlier Exp A run (batch 1, strict_batch_size false)
CODE_REF=$REL_REF
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$PWD/.cache" HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch"   # as minicpm_env.sh of the earlier run (an inherited HF_HOME is not kept)
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
  # the repo's copy of the config must be byte-identical to the addon copy (= the one of the earlier 4-language runs)
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
