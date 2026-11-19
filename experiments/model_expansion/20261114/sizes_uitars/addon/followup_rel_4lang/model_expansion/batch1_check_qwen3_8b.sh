#!/usr/bin/env bash
# Batch-size check: Qwen3-VL-8B (cached from the overnight job) at batch size 1 on a small fixed subset, compared with the batch-4 answers
# of the 2026-11-07 REL run. Run from anywhere; called by queue_sizes_uitars.sh BEFORE the 8B cache is deleted.
#   BATCH1_OUT=/workspace/queue_results/batch1_check_qwen3_8b bash followup_rel_4lang/model_expansion/batch1_check_qwen3_8b.sh
# Subset: Exp A main split, first BATCH1_ITEMS items (sorted parallel_id; default 28 items = 28 x 12 pairs x 3 conditions = 1,008 inferences),
# of which the O and R rows (672) have a batch-4 reference answer. Needs the GPU alone. No network (HF_HUB_OFFLINE=1).
# Local test hook: LOCAL_MOCK=1 uses --mock-model.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
: "${BATCH1_OUT:?set BATCH1_OUT}"
ITEMS="${BATCH1_ITEMS:-28}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}" HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
source .venv/bin/activate
source followup_rel_4lang/qwen3_4090_env.sh
MX=followup_rel_4lang/model_expansion
CONFIG=01_code/configs/models/qwen3_vl_8b_4090_channels_last.yaml
CFG_SHA=2775bb6b981d35922c46545ef58af0ac8e8ded1a3615902c73ac141609adbc8b   # same config as the Exp A qwen3 run (run_expA.sh)
CONTRACT=followup_rel_4lang/rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json
REF=followup_rel_4lang/rstem_ropts_inputs/reference_qwen3_20261107_OR.jsonl
MOCK=""; [ "${LOCAL_MOCK:-0}" = 1 ] && MOCK="--mock-model"
mkdir -p "$BATCH1_OUT"
echo "$(date -u +%FT%TZ) batch1 check: $ITEMS items, batch size 1, config $CONFIG" | tee -a "$BATCH1_OUT/STATUS"
got=$(sha256sum "$CONFIG" | cut -d' ' -f1)
[ "$got" = "$CFG_SHA" ] || { echo "config sha256 $got differs from the earlier Qwen3-8B run ($CFG_SHA)" | tee -a "$BATCH1_OUT/STATUS"; exit 2; }
(cd followup_rel_4lang/expA && sha256sum -c expA_inputs/EXPA_SHA256SUMS) > "$BATCH1_OUT/expA_inputs_verify.log"
python "$MX/run_rel_T_generic.py" --model qwen3_vl_8b_batch1_check --model-config "$CONFIG" --reference-contract "$CONTRACT" \
  --split main --max-items "$ITEMS" --resume $MOCK --output-dir "$BATCH1_OUT/run_batch1" 2>&1 | tee -a "$BATCH1_OUT/run.log"
python "$MX/batch1_compare.py" "$BATCH1_OUT/run_batch1/predictions.jsonl" "$REF" "$BATCH1_OUT/batch1_vs_batch4.json" | tee "$BATCH1_OUT/compare.log" | tail -1
echo "$(date -u +%FT%TZ) batch1 check done: $(tail -1 "$BATCH1_OUT/compare.log")" >> "$BATCH1_OUT/STATUS"
