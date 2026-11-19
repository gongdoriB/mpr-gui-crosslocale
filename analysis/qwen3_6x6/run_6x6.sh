#!/bin/bash
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd ${TMPDIR}/work
D=${TMPDIR}/work P=${TMPDIR}/work
case $1 in
rq1) Rscript analysis/rq1_glmm.R $D $P/rq1_generation generation_correct 1;;
rq2) Rscript analysis/rq2_glmm.R $D $P/rq2_generation_fast generation_correct fast 1;;
slope) Rscript analysis/random_slope_sensitivity.R $D $P/random_slope_generation_fast generation_correct fast 1;;
core5) Rscript analysis/core5_glmm.R $D $P/core5_rq1_generation standard 1 rq1;;
esac > ${TMPDIR}/work 2>&1
