#!/bin/bash
# 2026-11-22 17:20: parallel replacement for the remainder of run.sh (stopped during qwen3_vl_4b core5 to run the three models concurrently).
# Same commands and settings as run.sh; arg1 = model, arg2 = first step (build|core5). Then REL text-dependency (same as run_textdep.sh).
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
m="$1"; START="$2"
H="${PROJECT_ROOT}/analysis_6x6_rest_20261122"; cd "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/code_01"
A="${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference/annotation/rel_text_dependency"
case "$m" in
 qwen3_vl_4b) IN="${PROJECT_ROOT}/model_results_20261118/qwen3_vl_4b_6x6/unpacked/6x6_qwen3_vl_4b/full_6x6.jsonl";;
 internvl35_4b) IN="${PROJECT_ROOT}/model_results_20261118/internvl35_4b_6x6/unpacked/6x6_internvl35_4b/full_6x6.jsonl";;
 internvl25_8b) IN="${PROJECT_ROOT}/model_results_20261118/internvl25_8b_6x6_4090/unpacked/6x6_internvl25_8b/full_6x6.jsonl";;
esac
D="$H/derived/${m}_glmm.csv"; L="$H/logs/par"; mkdir -p "$L"; F="$L/done_${m}.txt"
if [ "$START" = build ]; then
  PYTHONPATH=src python3 -m mpr_crosslocale.analysis.build_glmm_table --input "$IN" --output "$D" --validation-out "$H/derived/${m}_validation.json" > "$L/${m}_build.log" 2>&1; echo "build exit $?" >> "$F"
  Rscript analysis/rq1_glmm.R "$D" "$H/results/$m/rq1_generation" generation_correct 1 > "$L/${m}_rq1.log" 2>&1; echo "rq1 exit $?" >> "$F"
fi
Rscript analysis/core5_glmm.R "$D" "$H/results/$m/core5_rq1_generation" standard 1 rq1 > "$L/${m}_core5.log" 2>&1; echo "core5 exit $?" >> "$F"
PYTHONPATH=src python3 -m mpr_crosslocale.analysis.descriptive_glmm --table "$m=$D" --csv-out "$H/results/descriptive_bootstrap_$m.csv" --json-out "$H/results/descriptive_details_$m.json" --bootstrap-replicates 10000 > "$L/${m}_boot.log" 2>&1; echo "boot exit $?" >> "$F"
Rscript analysis/rel_text_dependency_glmm.R "$D" "$A/outputs/rel_annotations_integrated_366.csv" "$H/results/$m/rel_text_dependency" generation_correct 1 "$A/data/pilot_manifest.json" > "$L/${m}_textdep.log" 2>&1; echo "textdep exit $?" >> "$F"
Rscript analysis/rq2_glmm.R "$D" "$H/results/$m/rq2_generation_fast" generation_correct fast 1 > "$L/${m}_rq2.log" 2>&1; echo "rq2 exit $?" >> "$F"
echo ALL_DONE >> "$F"
