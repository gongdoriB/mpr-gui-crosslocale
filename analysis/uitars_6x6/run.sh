#!/bin/bash
# UI-TARS-1.5-7B 6x6, frozen rule (paper_plan_20261112/분석규칙_v1.md, 6×6): RQ1 GLMM matched LRT (frozen rq1_glmm.R),
# Thai-excluded 5x5 with the same script, simple diff item bootstrap 10,000.
eval "$(conda shell.bash hook)"; conda activate mpr-r-analysis
H="${PROJECT_ROOT}/analysis_uitars_6x6_20261116"
cd "${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/code_01"
Rscript analysis/rq1_glmm.R "$H/derived/uitars_glmm.csv" "$H/results/rq1_generation" generation_correct 1 > "$H/logs/rq1.log" 2>&1; echo "rq1 exit $?" >> "$H/logs/done.txt"
Rscript analysis/rq1_glmm.R "$H/derived/uitars_glmm_noTH_5x5.csv" "$H/results/rq1_generation_noTH_5x5" generation_correct 1 > "$H/logs/rq1_noTH.log" 2>&1; echo "rq1_noTH exit $?" >> "$H/logs/done.txt"
PYTHONPATH=src python3 -m mpr_crosslocale.analysis.descriptive_glmm --table "uitars=$H/derived/uitars_glmm.csv" --csv-out "$H/results/descriptive_bootstrap_uitars.csv" --json-out "$H/results/descriptive_details_uitars.json" --bootstrap-replicates 10000 > "$H/logs/boot.log" 2>&1; echo "boot exit $?" >> "$H/logs/done.txt"
# 2026-11-16: the rq1_glmm.R call on the filtered noTH CSV above fails by design (script requires 36 rows/item).
# The frozen 5x5 procedure (as for the 8B models) is core5_glmm.R on the full table:
Rscript analysis/core5_glmm.R "$H/derived/uitars_glmm.csv" "$H/results/core5_rq1_generation" standard 1 rq1 > "$H/logs/core5.log" 2>&1; echo "core5 exit $?" >> "$H/logs/done.txt"
# 2026-11-16 (supplementary, not a frozen primary contrast): RQ2 dimension GLMM, same script/settings as 8B models,
# so the 5-1 table's REL row uses one estimand for all models.
# Rscript analysis/rq2_glmm.R "$H/derived/uitars_glmm.csv" "$H/results/rq2_generation_fast" generation_correct fast 1 > "$H/logs/rq2.log" 2>&1
