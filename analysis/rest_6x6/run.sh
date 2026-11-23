#!/bin/bash
# 6x6 frozen-rule analysis (paper_plan_20261112/분석규칙_v1.md, 6×6) for the four models without one, same code and settings as UI-TARS
# (analysis_uitars_6x6_20261116/run.sh); bash 3.2 compatible; repo code run as before (PYTHONPATH=src python3 -m): build table, RQ1 GLMM, core5 Thai-excluded 5x5, item bootstrap 10,000; RQ2 fast as supplementary.
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
H="${PROJECT_ROOT}/analysis_6x6_rest_20261122"; cd "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/code_01"
inp() { case "$1" in
 minicpm_v45) echo "${PROJECT_ROOT}/model_results_20261118/minicpm_6x6/unpacked/6x6_minicpm_v45/full_6x6.jsonl";;
 qwen3_vl_4b) echo "${PROJECT_ROOT}/model_results_20261118/qwen3_vl_4b_6x6/unpacked/6x6_qwen3_vl_4b/full_6x6.jsonl";;
 internvl35_4b) echo "${PROJECT_ROOT}/model_results_20261118/internvl35_4b_6x6/unpacked/6x6_internvl35_4b/full_6x6.jsonl";;
 internvl25_8b) echo "${PROJECT_ROOT}/model_results_20261118/internvl25_8b_6x6_4090/unpacked/6x6_internvl25_8b/full_6x6.jsonl";;
esac; }
for m in minicpm_v45 qwen3_vl_4b internvl35_4b internvl25_8b; do
  D="$H/derived/${m}_glmm.csv"
  PYTHONPATH=src python3 -m mpr_crosslocale.analysis.build_glmm_table --input "$(inp $m)" --output "$D" --validation-out "$H/derived/${m}_validation.json" > "$H/logs/${m}_build.log" 2>&1; echo "$m build exit $?" >> "$H/logs/done.txt"
  Rscript analysis/rq1_glmm.R "$D" "$H/results/$m/rq1_generation" generation_correct 1 > "$H/logs/${m}_rq1.log" 2>&1; echo "$m rq1 exit $?" >> "$H/logs/done.txt"
  Rscript analysis/core5_glmm.R "$D" "$H/results/$m/core5_rq1_generation" standard 1 rq1 > "$H/logs/${m}_core5.log" 2>&1; echo "$m core5 exit $?" >> "$H/logs/done.txt"
  PYTHONPATH=src python3 -m mpr_crosslocale.analysis.descriptive_glmm --table "$m=$D" --csv-out "$H/results/descriptive_bootstrap_$m.csv" --json-out "$H/results/descriptive_details_$m.json" --bootstrap-replicates 10000 > "$H/logs/${m}_boot.log" 2>&1; echo "$m boot exit $?" >> "$H/logs/done.txt"
  Rscript analysis/rq2_glmm.R "$D" "$H/results/$m/rq2_generation_fast" generation_correct fast 1 > "$H/logs/${m}_rq2.log" 2>&1; echo "$m rq2 exit $?" >> "$H/logs/done.txt"
done
echo ALL_DONE >> "$H/logs/done.txt"
