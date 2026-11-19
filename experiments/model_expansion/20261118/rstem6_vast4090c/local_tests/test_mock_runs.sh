#!/bin/bash
# Local: smoke + full with --mock-model (LOCAL_MOCK=1) for the five model keys in local copies of the repository states, each with its gate. No GPU, no model.
# Env: TESTROOT (dir with q mc ivl, addon installed, stub .venv), OR_SOURCE (OR_gemma_conditions.jsonl, optional), OUTLOG (dir for logs)
TESTROOT=${TESTROOT:?}; OUTLOG=${OUTLOG:?}; mkdir -p "$OUTLOG"
export LOCAL_MOCK=1 EXPECT_SKIP=qwen-vl-utils,timm,protobuf,sentencepiece,torchvision,torch
for pair in "q qwen3_vl_8b" "q uitars_15_7b" "mc minicpm_v45" "ivl internvl35_8b" "ivl internvl25_8b"; do
  set -- $pair
  ( cd "$TESTROOT/$1" && rm -rf followup_rel_6lang/results/rstem6_$2
    for st in smoke full; do FULL_LIMIT_S=99999 bash followup_rel_6lang/rstem6/run_rstem6.sh $2 $st > "$OUTLOG/mock_$2_$st.log" 2>&1; echo "$2 $st rc=$?"; done
    python3 - "$2" <<'PY'
import json, sys
k = sys.argv[1]
for m, f in (("smoke", "smoke/gate_smoke.json"), ("full", "full/gate_complete.json")):
    g = json.load(open(f"followup_rel_6lang/results/rstem6_{k}/{f}"))
    rc = {n: v['common_rows'] for n, v in (g['references'] or {}).items()}
    print(f"  {k} {m}: status {g['status']} rows {g['rows']} batch {g['batch']} sizes {g['batch_sizes']} parse_fail {g['parse_failures']} problems {g['problems']} ref_common {rc}")
PY
    tail -2 followup_rel_6lang/results/rstem6_$2/STATUS )
done
