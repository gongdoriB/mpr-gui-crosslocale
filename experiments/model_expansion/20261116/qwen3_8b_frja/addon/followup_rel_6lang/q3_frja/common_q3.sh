# Sourced by run_rel6_qwen3.sh, run_consistency_qwen3.sh, run_expA6_qwen3.sh (not run on its own).
# Caller sets before sourcing: OUT (result folder), TAG (status prefix). cwd = repository root (/workspace/mpr-gui-crosslocale).
# Optional env: LOCAL_MOCK=1 (local tests only: no GPU / no model snapshot / no package pins, adds --mock-model), IGNORE_ETA=1.
X=followup_rel_6lang/q3_frja
DATA=followup_rel_6lang
LEGACY=followup_rel_4lang
CONFIG=$X/qwen3_vl_8b_4090_channels_last.yaml
REFERENCE=$X/reference_contract_qwen3_rel_20261107.json   # Qwen3-VL-8B 4-language REL run contract of 2026-11-07: model config, system prompt, code hashes, package versions
CFG_SHA=2775bb6b981d35922c46545ef58af0ac8e8ded1a3615902c73ac141609adbc8b
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
export CONFIG REFERENCE CFG_SHA
mkdir -p "$OUT/logs"
MOCK=""; GMOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && { MOCK="--mock-model"; GMOCK="--mock"; }   # GMOCK: gate flag, mock prompts have no chat template
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
  # the config of the repo (when it has one) must be byte-identical to the addon copy (= the one of the 2026-11-07 REL run and the 2026-11-14 Exp A run)
  [ ! -f 01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml ] || cmp "$CONFIG" 01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml
  LOCAL_MOCK="${LOCAL_MOCK:-0}" python - <<'PY'
import importlib, importlib.metadata as md, json, os
from pathlib import Path
import yaml
from mpr_crosslocale.inference.run_rel_followup import code_identity
from mpr_crosslocale.inference.system_prompts import build_system_prompt
ref = json.loads(Path(os.environ["REFERENCE"]).read_text())
cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
assert cfg == ref["model_config"], "model config differs from the 2026-11-07 Qwen3-VL-8B REL contract"
assert cfg["model_id"] == "Qwen/Qwen3-VL-8B-Instruct" and cfg["revision"] == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" and cfg["patch_embed_memory_format"] == "channels_last_3d", cfg
system_prompt, _, template = build_system_prompt("english_fixed", "en")
assert system_prompt == ref["system_prompt"] and template == ref["prompt_template_version"], "system prompt differs from the REL contract"
now = code_identity()
diff = sorted(k for k in set(now) | set(ref["code_sha256"]) if k != "run_rel_T.py" and ref["code_sha256"].get(k) != now.get(k))
assert not diff, f"inference code of the repository differs from the 2026-11-07 run: {diff}"
print("model config, system prompt and the 12 inference-code hashes equal the 2026-11-07 Qwen3-VL-8B REL contract")
if os.environ.get("LOCAL_MOCK") != "1":
    import torch
    name = torch.cuda.get_device_name(0)
    assert torch.cuda.device_count() == 1 and "4090" in name, name
    for pkg in ("torch", "transformers", "accelerate", "qwen-vl-utils", "torchvision"):
        have = md.version(pkg)
        assert have == ref["software_versions"][pkg], f"{pkg} {have} differs from the earlier run {ref['software_versions'][pkg]}"
    from huggingface_hub import snapshot_download
    snap = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    assert Path(snap, "config.json").is_file(), snap
    print("GPU", name, "| torch", torch.__version__, "| transformers", md.version("transformers"), "| model", snap)
PY
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
