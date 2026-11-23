#!/bin/bash
# REL text-dependency GLMM (frozen script rel_text_dependency_glmm.R, same arguments as analysis_internvl35_6x6_20261112/run.sh textdep)
# for the five models that lack it; UI-TARS uses its existing GLMM table. 2026-11-22.
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/code_01"
A="${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference/annotation/rel_text_dependency"
for m in minicpm_v45 qwen3_vl_4b internvl35_4b internvl25_8b uitars; do
  if [ $m = uitars ]; then D="${PROJECT_ROOT}/analysis_uitars_6x6_20261116/derived/uitars_glmm.csv"; else D="${PROJECT_ROOT}/analysis_6x6_rest_20261122/derived/${m}_glmm.csv"; fi
  Rscript analysis/rel_text_dependency_glmm.R "$D" "$A/outputs/rel_annotations_integrated_366.csv" "${PROJECT_ROOT}/analysis_6x6_rest_20261122/results/$m/rel_text_dependency" generation_correct 1 "$A/data/pilot_manifest.json" > "${PROJECT_ROOT}/analysis_6x6_rest_20261122/logs/${m}_textdep.log" 2>&1
  echo "$m textdep exit $?" >> "${PROJECT_ROOT}/analysis_6x6_rest_20261122/logs/done_textdep.txt"
done
echo ALL_DONE >> "${PROJECT_ROOT}/analysis_6x6_rest_20261122/logs/done_textdep.txt"
