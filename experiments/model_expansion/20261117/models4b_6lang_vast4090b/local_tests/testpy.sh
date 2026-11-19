#!/bin/bash
# Stand-in for <repo>/.venv/bin/python in the local tests: the local test python (torch 2.9.1, transformers 4.57.6, CPU) with the repo's own source first and tiny stand-in modules for qwen_vl_utils/timm-less imports last.
# TESTPY_REPO = repository root (set by mk_repo in run_b3_queue_tests.sh, written into the generated wrapper).
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${TESTPY_REPO:+$TESTPY_REPO/01_code/src:}${PYTHONPATH:-}:$HERE/shimmods"
if [ "$1" = "-m" ] && [ "$2" = "mpr_crosslocale.data.download" ]; then
  echo "(test) fake download: copying $LOCAL_DATA_SRC to $PWD/data/raw/mpr_gui_bench"
  mkdir -p data/raw && cp -R "$LOCAL_DATA_SRC" data/raw/mpr_gui_bench && exit 0
fi
exec "${TEST_PY_REAL:-${TMPDIR}/work" "$@"
