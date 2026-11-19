#!/usr/bin/env bash
# Builds, in a scratch folder, a stand-in for the Qwen repo of the old server (/workspace/mpr-gui-crosslocale = commit 6664400 state) for the LOCAL tests:
#  - 01_code and followup_rel_4lang tree of the (InternVL-patched) local copy internvl35_prep/mpr-gui-crosslocale with patches 0001-0005 REVERSED for 01_code and 02_tests
#    (the result reproduces the 12 code hashes of the Qwen3-VL-8B run contracts, checked by test_code_identity), 4-language images hard-linked from the InternVL2.5 test repo,
#    rstem_ropts_inputs of the Qwen3 addon (reference contract and files the old Exp A runner needs), the old Exp A runner (expA/), and a venv stub.
# usage: make_q3_template.sh <target dir>      Nothing outside <target dir> is written. Needs the Mac test venv python (TEST_PY).
set -euo pipefail
T="${1:?target dir}"; ROOT="${PROJECT_ROOT}"
S=${TMPDIR}/work
TEST_PY="${TEST_PY:-$S/mc/venv/bin/python}"
IVL_REPO="${IVL_REPO:-$S/ivl25_repo}"          # InternVL2.5 test repo (4-language images); only its followup_rel_4lang/assets is used (hard links)
rm -rf "$T"; mkdir -p "$T"
rsync -a --exclude results --exclude __pycache__ --exclude assets --exclude .git "$ROOT/internvl35_prep/mpr-gui-crosslocale/" "$T/"
( cd "$T" && git init -q . && for n in 5 4 3 2 1; do p=$(ls "$ROOT/internvl35_prep/"000$n-*.patch); git apply -R --include='01_code/*' --include='02_tests/*' "$p"; done; rm -rf .git )
cp -Rl "$IVL_REPO/followup_rel_4lang/assets" "$T/followup_rel_4lang/assets"
mkdir -p "$T/followup_rel_4lang/rstem_ropts_inputs"
cp "$ROOT/rstem_ropts_qwen3_20261109/addon/followup_rel_4lang/rstem_ropts_inputs/"* "$T/followup_rel_4lang/rstem_ropts_inputs/"
cp -R "$ROOT/expA_translation_20261112/addon/followup_rel_4lang/expA" "$T/followup_rel_4lang/expA"
mkdir -p "$T/.venv/bin"
printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec %s "$@"\n' "$T" "$TEST_PY" > "$T/.venv/bin/python"; chmod +x "$T/.venv/bin/python"; ln -sf python "$T/.venv/bin/python3"
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$T" > "$T/.venv/bin/activate"
echo "template ready: $T"
