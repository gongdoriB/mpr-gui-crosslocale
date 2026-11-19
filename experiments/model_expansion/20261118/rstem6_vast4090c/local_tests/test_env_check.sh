#!/bin/bash
# Local: env_check_rstem6.py against the five earlier run contracts, in local copies of the repository states (q = 6664400, mc = +MiniCPM patch, ivl = +InternVL patches 0001-0005).
# Usage: TESTROOT=<dir with q mc ivl> PY=<python with torch/transformers> bash test_env_check.sh
set -u
TESTROOT=${TESTROOT:?}; PY=${PY:?}
for f in "q qwen3_vl_8b qwen3_vl_8b_4090_channels_last.yaml ref_qwen3_rel_20261107.json" \
         "q uitars_15_7b uitars_1_5_7b_4090.yaml ref_uitars_rel_20261114.json" \
         "mc minicpm_v45 minicpm_v4_5_8b_4090.yaml ref_minicpm_rel_20261116.json" \
         "ivl internvl35_8b internvl3_5_8b_4090.yaml ref_internvl35_rel_20261109.json" \
         "ivl internvl25_8b internvl2_5_8b.yaml ref_internvl25_expA_20261114.json"; do
  set -- $f
  echo "== $2 (repo $1)"
  ( cd "$TESTROOT/$1" && KEY=$2 CONFIG=followup_rel_6lang/rstem6/models/$3 CONTRACT=followup_rel_6lang/rstem6/contracts/$4 LOCAL_MOCK=1 EXPECT_SKIP=${EXPECT_SKIP:-} \
    "$PY" -I -c "import sys,runpy; sys.path.insert(0,'01_code/src'); runpy.run_path('followup_rel_6lang/rstem6/env_check_rstem6.py')" 2>&1 | tail -3 )
done
