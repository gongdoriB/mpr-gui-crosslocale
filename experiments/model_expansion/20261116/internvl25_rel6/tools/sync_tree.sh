#!/usr/bin/env bash
# Local test tree (scratch): original 01_code + the new modules overlaid; followup_rel_6lang = addon files + hard-linked fr/ja images;
# followup_rel_4lang = hard-linked copy of the generator-side 4-language folder (images only needed). Originals are never written.
set -euo pipefail
B="${PROJECT_ROOT}"; O="$B/model_expansion_20261116/internvl25_rel6"; T="$1"
if [ ! -d "$T/followup_rel_4lang" ]; then mkdir -p "$T"; cp -Rl "$B/followup_rel_4lang" "$T/followup_rel_4lang"; cp -Rl "$B/followup_rel_6lang/assets" "$T/_frja_assets"; fi
rm -rf "$T/01_code" "$T/followup_rel_6lang"
cp -R "$B/internvl35_prep/mpr-gui-crosslocale/01_code" "$T/01_code"
find "$T/01_code" -name __pycache__ -prune -exec rm -rf {} +
cp -R "$O/addon/01_code/src/mpr_crosslocale/." "$T/01_code/src/mpr_crosslocale/"
cp -R "$O/addon/followup_rel_6lang" "$T/followup_rel_6lang"
cp -Rl "$T/_frja_assets" "$T/followup_rel_6lang/assets"
