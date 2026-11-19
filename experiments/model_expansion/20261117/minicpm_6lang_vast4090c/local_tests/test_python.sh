#!/bin/bash
# Q6_TEST_PY: the stub venv python. Needs LOCAL_DATA_SRC (a finished MPR-GUI-Bench download) and TEST_PY_REAL.
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PYTHONPATH:$HERE/shimmods"
if [ "$1" = "-m" ] && [ "$2" = "mpr_crosslocale.data.download" ]; then
  echo "(test) fake download: copying $LOCAL_DATA_SRC to $PWD/data/raw/mpr_gui_bench"
  mkdir -p data/raw && cp -R "$LOCAL_DATA_SRC" data/raw/mpr_gui_bench && exit 0
fi
exec "${TEST_PY_REAL:-${TMPDIR}/work" "$@"
