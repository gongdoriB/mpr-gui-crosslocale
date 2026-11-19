#!/usr/bin/env bash
# run test groups of run_q3_queue_tests.sh and write the output to <log>: run_group.sh <log> <test>...
O="${PROJECT_ROOT}/model_expansion_20261116/qwen3_8b_frja"
log=$1; shift
: > "$log"
for t in "$@"; do bash "$O/local_tests/run_q3_queue_tests.sh" "$t" >> "$log" 2>&1; done
echo "GROUP DONE $*" >> "$log"
