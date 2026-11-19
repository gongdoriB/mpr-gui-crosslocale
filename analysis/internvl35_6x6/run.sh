#!/bin/bash
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/code_01"
D="${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/derived/internvl35_glmm.csv"; P="${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/results/internvl35"
case $1 in
rq1) Rscript analysis/rq1_glmm.R "$D" "$P/rq1_generation" generation_correct 1;;
rq2) Rscript analysis/rq2_glmm.R "$D" "$P/rq2_generation_fast" generation_correct fast 1;;
slope) Rscript analysis/random_slope_sensitivity.R "$D" "$P/random_slope_generation_fast" generation_correct fast 1;;
core5) Rscript analysis/core5_glmm.R "$D" "$P/core5_rq1_generation" standard 1 rq1;;
textdep) Rscript analysis/rel_text_dependency_glmm.R "$D" "${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference/annotation/rel_text_dependency/outputs/rel_annotations_integrated_366.csv" "$P/rel_text_dependency" generation_correct 1 "${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference/annotation/rel_text_dependency/data/pilot_manifest.json";;
esac > "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/logs/$1.log" 2>&1
echo "$1 exit $?" >> "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/logs/done.txt"
