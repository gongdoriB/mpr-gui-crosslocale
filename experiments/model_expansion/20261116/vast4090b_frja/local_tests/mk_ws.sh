#!/usr/bin/env bash
# Builds the two workspace repos for test_b2_expA.py in the scratchpad: wsI = InternVL repo stand-in (mpr-ivl: ivl25_repo), wsQ = Qwen repo stand-in (mpr-gui-crosslocale: q3frja template + sizes_uitars generic runner + uitars config).
# Both get the b2 addon unpacked and the fr/ja images (hard links). Usage: mk_ws.sh
set -euo pipefail
S=${TMPDIR}/work
ROOT="${PROJECT_ROOT}"; M="$(cd "$(dirname "$0")/.." && pwd)"
for n in wsI wsQ; do rm -rf "${S:?}/b2/$n"; done
cp -Rl "$S/ivl25_repo" "$S/b2/wsI"; cp -Rl "$S/q3frja/template" "$S/b2/wsQ"
SZ="$ROOT/model_expansion_20261114/sizes_uitars/addon"
mkdir -p "$S/b2/wsQ/followup_rel_4lang/model_expansion"; cp "$SZ/followup_rel_4lang/model_expansion/run_rel_T_generic.py" "$S/b2/wsQ/followup_rel_4lang/model_expansion/"
rm -f "$S/b2/wsQ/01_code/configs/models/uitars_1_5_7b_4090.yaml"; cp "$SZ/01_code/configs/models/uitars_1_5_7b_4090.yaml" "$S/b2/wsQ/01_code/configs/models/"
for n in wsI wsQ; do ( cd "$S/b2/$n" && tar -xzf "$M/b2_frja_addon_20261116.tar.gz" && cp -Rl "$ROOT/followup_rel_6lang/assets" followup_rel_6lang/assets ); done
echo ok
