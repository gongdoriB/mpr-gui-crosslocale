#!/bin/bash
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd ${TMPDIR}/work
nice -n 10 Rscript analysis/core5_glmm.R ${TMPDIR}/work "${PROJECT_ROOT}/analysis_plan_followup/task7/core5_all_fast" fast 1 all > "${PROJECT_ROOT}/analysis_plan_followup/logs/t7b_core5_all.log" 2>&1
