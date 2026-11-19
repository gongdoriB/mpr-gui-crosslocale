#!/usr/bin/env bash
set -u
S=${TMPDIR}/work
PY="${PROJECT_ROOT}/expA_translation_20261112/.venv/bin/python"
O=$S/sz_out; rm -rf "$O"; mkdir -p "$O"
for k in qwen3_vl_2b qwen3_vl_4b internvl35_1b internvl35_2b internvl35_4b uitars_15_7b; do
  case $k in qwen3_vl_2b) REPO=qwen_sizes; CF=qwen3_vl_2b_4090_channels_last;; qwen3_vl_4b) REPO=qwen_sizes; CF=qwen3_vl_4b_4090_channels_last;; uitars_15_7b) REPO=qwen_sizes; CF=uitars_1_5_7b_4090;; internvl35_1b) REPO=ivl_sizes; CF=internvl3_5_1b_4090;; internvl35_2b) REPO=ivl_sizes; CF=internvl3_5_2b_4090;; internvl35_4b) REPO=ivl_sizes; CF=internvl3_5_4b_4090;; esac
  cd $S/$REPO; C=01_code/configs/models/$CF.yaml; M=followup_rel_4lang/model_expansion
  if [[ $k == internvl* ]]; then R=followup_rel_4lang/rstem_ropts_inputs/run_contract_internvl35_rel_20261109.json; else R=followup_rel_4lang/rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json; fi
  run(){ name=$1; shift; "$PY" -I "$@" > $O/$k.$name.log 2>&1; echo "$k $name exit=$?"; }
  run rel_dry followup_rel_4lang/run_inference.py --model-config $C --batch-size 1 --strict-batch-size --output-dir $O/$k/rel_dry --dry-run
  run expa_dry $M/run_rel_T_generic.py --model $k --model-config $C --reference-contract $R --output-dir $O/$k/expa_dry --dry-run
  run rel_smoke followup_rel_4lang/run_inference.py --model-config $C --batch-size 1 --strict-batch-size --max-items 2 --mock-model --resume --output-dir $O/$k/rel_smoke
  run rel_smoke_gate $M/check_generic_gate.py rel smoke $O/$k/rel_smoke --model $k
  run expa_smoke $M/run_rel_T_generic.py --model $k --model-config $C --reference-contract $R --split main --max-items 3 --mock-model --resume --output-dir $O/$k/expa_smoke
  run expa_smoke_gate $M/check_generic_gate.py expa smoke $O/$k/expa_smoke --model $k
  run rel_full followup_rel_4lang/run_inference.py --model-config $C --batch-size 1 --strict-batch-size --mock-model --resume --output-dir $O/$k/rel_full
  run rel_full_gate $M/check_generic_gate.py rel complete $O/$k/rel_full 18432 --model $k
  run expa_full $M/run_rel_T_generic.py --model $k --model-config $C --reference-contract $R --split main --mock-model --resume --output-dir $O/$k/expa_full
  run expa_full_gate $M/check_generic_gate.py expa complete $O/$k/expa_full 5688 --model $k
done
