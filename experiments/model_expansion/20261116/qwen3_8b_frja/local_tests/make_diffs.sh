#!/usr/bin/env bash
# writes the diff files of local_tests (changes vs the originals this work was derived from)
set -u
rm -f /tmp/x_stub.sh
ROOT="${PROJECT_ROOT}"
O="$ROOT/model_expansion_20261116/qwen3_8b_frja"
I="$ROOT/model_expansion_20261116/internvl25_rel6"
T="$O/local_tests"
A="$O/addon/followup_rel_6lang"
{
  echo "# Files copied UNCHANGED from the InternVL2.5 REL6 addon (same bytes; sha256 of both sides)"
  for f in 01_code/src/mpr_crosslocale/data/rel_followup_6lang.py 01_code/src/mpr_crosslocale/inference/run_rel_followup_6lang.py followup_rel_6lang/run_inference_6lang.py followup_rel_6lang/data_bundle_6lang.py followup_rel_6lang/bundle_manifest.json \
           followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl followup_rel_6lang/data/rel_all_366/generation_report.json followup_rel_6lang/data/rel_all_366/source_issues.json followup_rel_6lang/data/rel_all_366/config.json \
           followup_rel_6lang/data/rel_all_366/manifest_366_6lang.json followup_rel_6lang/data/rel_all_366/reference_mappings.jsonl; do
    a=$(shasum -a 256 "$I/addon/$f" | cut -c1-64); b=$(shasum -a 256 "$O/addon/$f" | cut -c1-64)
    [ "$a" = "$b" ] && echo "SAME  $a  $f" || echo "DIFF  $a / $b  $f"
  done
  echo
  echo "# Exp A input files vs expA_translation_6lang_20261115/work (same bytes)"
  for n in OR_gemma_conditions.jsonl T_gemma_conditions.jsonl T_gemma_retry_conditions.jsonl; do
    a=$(shasum -a 256 "$ROOT/expA_translation_6lang_20261115/work/$n" | cut -c1-64); b=$(shasum -a 256 "$A/q3_frja/expA6_inputs/$n" | cut -c1-64)
    [ "$a" = "$b" ] && echo "SAME  $a  $n" || echo "DIFF  $a / $b  $n"
  done
  echo
  echo "# reference_contract_qwen3_rel_20261107.json vs the earlier run_contract.json (full_rel) and the repo's rstem_ropts_inputs copy"
  shasum -a 256 "$A/q3_frja/reference_contract_qwen3_rel_20261107.json" "$ROOT/internvl35_prep/mpr-gui-crosslocale/followup_rel_4lang/artifacts/qwen3_single4090_20261107/full_rel/run_contract.json" "$ROOT/rstem_ropts_qwen3_20261109/addon/followup_rel_4lang/rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json"
  echo "# qwen3_vl_8b_4090_channels_last.yaml vs the reconstructed repo state"
  shasum -a 256 "$A/q3_frja/qwen3_vl_8b_4090_channels_last.yaml"
} > "$T/unchanged_files_check.txt" 2>&1
diff -u "$I/addon/followup_rel_6lang/ivl25_rel6/check_rel6_gate.py" "$A/q3_frja/check_q3_gate.py" > "$T/diff_gate_vs_ivl25_rel6.txt"
diff -u "$ROOT/expA_translation_20261112/addon/followup_rel_4lang/expA/run_rel_T.py" "$A/q3_frja/expA6/run_rel_T_6lang.py" > "$T/diff_runner_expA_vs_4lang.txt"
diff -u "$ROOT/expA_translation_20261112/addon/followup_rel_4lang/expA/check_T_gate.py" "$A/q3_frja/expA6/check_T6_gate.py" > "$T/diff_gate_expA_vs_4lang.txt"
diff -u "$I/addon/followup_rel_6lang/ivl25_rel6/run_rel6_ivl25.sh" "$A/q3_frja/run_rel6_qwen3.sh" > "$T/diff_run_rel6_script_vs_ivl25.txt"
diff -u "$ROOT/expA_translation_20261112/addon/followup_rel_4lang/expA/run_expA.sh" "$A/q3_frja/run_expA6_qwen3.sh" > "$T/diff_run_expA6_script_vs_4lang.txt"
diff -u "$I/queue_internvl25_rel6.sh" "$O/queue_qwen3_8b_frja.sh" > "$T/diff_queue_vs_ivl25_rel6.txt"
wc -l "$T"/diff_*.txt "$T/unchanged_files_check.txt"
cat "$T/unchanged_files_check.txt"
