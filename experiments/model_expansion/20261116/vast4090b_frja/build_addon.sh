#!/usr/bin/env bash
# Local only: rebuild the server addon from addon/ (after editing files there), write its sha256 into queue_b2_frja.sh and SHA256SUMS. Run in this folder. Nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")"
NAME=b2_frja_addon_20261116.tar.gz
find addon -name .DS_Store -delete; find addon -name __pycache__ -prune -exec rm -rf {} +
( cd addon/followup_rel_6lang/b2_frja && shasum -a 256 check_rel6_gate.py env_check.py common_b2.sh run_rel6_b2.sh run_expA6_b2.sh expA6/run_rel_T_6lang_multi.py expA6/check_T6_gate_multi.py \
    models/*.yaml contracts/*.json expA6_inputs/* expA6_all6_inputs/* \
    ../../01_code/src/mpr_crosslocale/data/rel_followup_6lang.py ../../01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py \
    ../run_inference_6lang.py ../data_bundle_6lang.py ../bundle_manifest.json ../data/rel_all_366/conditions_6lang.jsonl > SHA256SUMS )
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_6lang 2>/dev/null \
  || COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_6lang
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1)
[ -f queue_b2_frja.sh ] && sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_b2_frja.sh && bash -n queue_b2_frja.sh
shasum -a 256 "$NAME" $( [ -f queue_b2_frja.sh ] && echo queue_b2_frja.sh ) step1_ivl35_rel/queue_b2_step1_ivl35_rel.sh > SHA256SUMS
tar -tzf "$NAME" | grep -v '/$' | head -50; ls -la "$NAME"; cat SHA256SUMS
