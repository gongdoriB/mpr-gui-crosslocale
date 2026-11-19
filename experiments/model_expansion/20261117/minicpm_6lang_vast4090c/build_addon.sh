#!/usr/bin/env bash
# Builds mc6/SHA256SUMS and upload/minicpm_6lang_addon_20261117.tar.gz from addon/. Local only.
set -euo pipefail
cd "$(dirname "$0")"
M=addon/followup_rel_6lang/mc6
( cd "$M" && find . -type f ! -name SHA256SUMS | sort | sed 's|^\./||' | xargs shasum -a 256 > SHA256SUMS
  shasum -a 256 ../../01_code/src/mpr_crosslocale/data/rel_followup_6lang.py ../../01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py ../run_inference_6lang.py ../data_bundle_6lang.py ../bundle_manifest.json ../data/rel_all_366/conditions_6lang.jsonl >> SHA256SUMS )
mkdir -p upload
COPYFILE_DISABLE=1 tar -czf upload/minicpm_6lang_addon_20261117.tar.gz -C addon 01_code followup_rel_6lang
shasum -a 256 upload/minicpm_6lang_addon_20261117.tar.gz
