#!/usr/bin/env bash
# Local only: rebuild the server addon from addon/ (after editing files there), write the sha256 values into queue_internvl25_rel6.sh and SHA256SUMS.
# Run in this folder. Nothing is uploaded. Writes only here. The fr/ja screenshots are read from ../../followup_rel_6lang/assets (not copied).
set -euo pipefail
cd "$(dirname "$0")"
SRC_ROOT="${PROJECT_ROOT}/followup_rel_6lang"
NAME=internvl25_rel6_addon_20261116.tar.gz
IMG=internvl25_rel6_frja_images_20261116.tar
find addon -name .DS_Store -delete; find addon -name __pycache__ -prune -exec rm -rf {} +
( cd addon/followup_rel_6lang/ivl25_rel6 && shasum -a 256 internvl2_5_8b.yaml reference_contract_internvl25_rel_20261005.json run_rel6_ivl25.sh check_rel6_gate.py \
    ../../01_code/src/mpr_crosslocale/data/rel_followup_6lang.py ../../01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py \
    ../run_inference_6lang.py ../data_bundle_6lang.py ../bundle_manifest.json ../data/rel_all_366/conditions_6lang.jsonl > SHA256SUMS )
# the SHA256SUMS paths are relative to ivl25_rel6/: 01_code lives at the repo root, two levels up
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_6lang
# fr/ja screenshots listed in the manifest (location "own"), stored under followup_rel_6lang/assets/images/...
python3 - <<'PY' > /tmp/rel6_img_list.$$
import json
m = json.load(open("addon/followup_rel_6lang/bundle_manifest.json"))
for e in m["images"]:
    if e["location"] == "own":
        print(e["path"])
PY
COPYFILE_DISABLE=1 /usr/bin/tar --no-xattrs --uid 0 --gid 0 --numeric-owner -cf "$IMG" -C "$SRC_ROOT" -s ',^assets/,followup_rel_6lang/assets/,' -T /tmp/rel6_img_list.$$
rm -f /tmp/rel6_img_list.$$
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1); HI=$(shasum -a 256 "$IMG" | cut -d' ' -f1)
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/; s/^IMAGES_SHA=.*/IMAGES_SHA=$HI/" queue_internvl25_rel6.sh
bash -n queue_internvl25_rel6.sh
shasum -a 256 "$NAME" "$IMG" queue_internvl25_rel6.sh > SHA256SUMS
tar -tzf "$NAME" | head -40; echo; tar -tf "$IMG" | head -3; echo "images in tar: $(tar -tf "$IMG" | wc -l)"; ls -la "$NAME" "$IMG"; cat SHA256SUMS
