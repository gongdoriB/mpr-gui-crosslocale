#!/usr/bin/env bash
# debug helper (local mock only): run the Exp A stages in the debug workspace
S=${TMPDIR}/work
cd $S/q3frja/dbg/mpr-gui-crosslocale || exit 1
export LOCAL_MOCK=1 EXPECT_TORCH=2.9.1
for st in smoke full retry; do
  echo "== $st"
  ( time bash followup_rel_6lang/q3_frja/run_expA6_qwen3.sh $st ) 2>&1 | grep -E "\[qwen3|Error|error:|Traceback|Assertion|^real|ValueError" | head -8
done
ls followup_rel_6lang/results/expA6_qwen3_vl_8b/*
