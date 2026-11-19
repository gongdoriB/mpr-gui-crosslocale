#!/usr/bin/env bash
# Local only: rebuild the server addon from addon/ (after editing files there), write its sha256 into queue_b3_4b.sh, rebuild upload/ and SHA256SUMS. Run in this folder. Nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")"
NAME=b3_4b_addon_20261117.tar.gz
find addon -name .DS_Store -delete; find addon -name __pycache__ -prune -exec rm -rf {} +
( cd addon/followup_rel_6lang/b3_4b && shasum -a 256 check_rel6_gate.py env_check.py common_b3.sh run_rel6_b3.sh run_expA6_b3.sh verify_6x6_data.py expA6/run_rel_T_6lang_multi.py expA6/check_T6_gate_multi.py \
    models/*.yaml contracts/*.json expA6_inputs/* \
    ../../01_code/src/mpr_crosslocale/data/rel_followup_6lang.py ../../01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py \
    ../run_inference_6lang.py ../data_bundle_6lang.py ../bundle_manifest.json ../data/rel_all_366/conditions_6lang.jsonl > SHA256SUMS )
mkdir -p upload
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "upload/$NAME" -C addon 01_code followup_rel_6lang 2>/dev/null \
  || COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --numeric-owner -czf "upload/$NAME" -C addon 01_code followup_rel_6lang
H=$(shasum -a 256 "upload/$NAME" | cut -d' ' -f1)
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_b3_4b.sh && bash -n queue_b3_4b.sh
cp queue_b3_4b.sh upload/queue_b3_4b.sh
ROOT="$(cd .. && pwd)/../model_expansion_20261114/sixbysix"
cp "$ROOT/queue_6x6.sh" "$ROOT/sixbysix_addon_20261114.tar.gz" upload/
( cd upload && shasum -a 256 queue_b3_4b.sh "$NAME" queue_6x6.sh sixbysix_addon_20261114.tar.gz > SHA256SUMS && cat SHA256SUMS )
tar -tzf "upload/$NAME" | grep -v '/$' | head -60; ls -la "upload/$NAME"
