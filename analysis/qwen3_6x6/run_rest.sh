#!/bin/bash
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
cd ${TMPDIR}/work
P=${TMPDIR}/work
case $1 in
rel4check) Rscript ${TMPDIR}/work ${TMPDIR}/work
rel4fit) Rscript ${TMPDIR}/work ${TMPDIR}/work ${TMPDIR}/work qwen3;;
sens4) Rscript ${TMPDIR}/work ${TMPDIR}/work $P/sens4lang_rq1_generation generation_correct 1;;
textdep) cd 01_code; Rscript analysis/rel_text_dependency_glmm.R ${TMPDIR}/work ${TMPDIR}/work $P/rel_text_dependency generation_correct 1 ${TMPDIR}/work
esac > ${TMPDIR}/work 2>&1
