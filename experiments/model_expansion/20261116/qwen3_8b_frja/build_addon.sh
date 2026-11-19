#!/usr/bin/env bash
# Local only: rebuild the server addon from addon/ (after editing files there), write its sha256 into queue_qwen3_8b_frja.sh and SHA256SUMS.
# Run in this folder. Nothing is uploaded. Writes only here. The fr/ja screenshot tar is NOT rebuilt: the queue uses the InternVL2.5 one (internvl25_rel6_frja_images_20261116.tar, sha in the queue).
set -euo pipefail
cd "$(dirname "$0")"
NAME=qwen3_8b_frja_addon_20261116.tar.gz
find addon -name .DS_Store -delete; find addon -name __pycache__ -prune -exec rm -rf {} +
( cd addon/followup_rel_6lang/q3_frja && shasum -a 256 qwen3_vl_8b_4090_channels_last.yaml reference_contract_qwen3_rel_20261107.json common_q3.sh run_rel6_qwen3.sh run_consistency_qwen3.sh run_expA6_qwen3.sh \
    check_q3_gate.py consistency_run.py consistency_compare.py consistency_subset_ids.json consistency_reference.jsonl \
    expA6/run_rel_T_6lang.py expA6/check_T6_gate.py expA6_inputs/OR_gemma_conditions.jsonl expA6_inputs/T_gemma_conditions.jsonl expA6_inputs/T_gemma_retry_conditions.jsonl expA6_inputs/expA6_lock.json \
    ../../01_code/src/mpr_crosslocale/data/rel_followup_6lang.py ../../01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py \
    ../run_inference_6lang.py ../data_bundle_6lang.py ../bundle_manifest.json ../data/rel_all_366/conditions_6lang.jsonl > SHA256SUMS )
# SHA256SUMS paths are relative to q3_frja/: 01_code lives at the repo root, two levels up
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_6lang 2>/dev/null \
  || COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_6lang
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1)
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_qwen3_8b_frja.sh
bash -n queue_qwen3_8b_frja.sh
shasum -a 256 "$NAME" queue_qwen3_8b_frja.sh > SHA256SUMS
tar -tzf "$NAME" | grep -v '/$' | head -60; echo; ls -la "$NAME"; cat SHA256SUMS
