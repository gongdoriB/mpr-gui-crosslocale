#!/usr/bin/env bash
# Local only: rebuild the addon tar.gz from addon/ (after editing files there), write its sha256 into queue_internvl25_expA.sh and SHA256SUMS.
# Run in this folder. Nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")"
NAME=internvl25_expA_addon_20261114.tar.gz
( cd addon/followup_rel_4lang/expA_ivl25 && shasum -a 256 internvl2_5_8b.yaml reference_contract_internvl25_rel_20261005.json run_rel_T_generic.py check_generic_gate.py run_expA_ivl25.sh > SHA256SUMS )
find addon -name .DS_Store -delete
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon followup_rel_4lang
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1)
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_internvl25_expA.sh
bash -n queue_internvl25_expA.sh
shasum -a 256 "$NAME" queue_internvl25_expA.sh > SHA256SUMS
tar -tzf "$NAME"; cat SHA256SUMS
