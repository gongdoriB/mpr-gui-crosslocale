#!/bin/bash
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd ${TMPDIR}/work
D=${TMPDIR}/work
O="${PROJECT_ROOT}/analysis_plan_followup/task7"
nice -n 10 Rscript analysis/rq1_glmm.R $D "$O/rq1_scoring" scoring_correct 1 > "${PROJECT_ROOT}/analysis_plan_followup/logs/t7a_rq1_scoring.log" 2>&1
nice -n 10 Rscript analysis/core5_glmm.R $D "$O/core5_all_fast" fast 1 all > "${PROJECT_ROOT}/analysis_plan_followup/logs/t7b_core5_all.log" 2>&1
