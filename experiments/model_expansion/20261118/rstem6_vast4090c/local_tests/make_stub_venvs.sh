#!/bin/bash
# Local tests: stub .venv in each test repo (python wrapper -> a venv with torch/transformers, PYTHONPATH = the repo's 01_code/src)
TESTROOT=${TESTROOT:?}; PY=${PY:?}
for f in q mc ivl; do
  R=$TESTROOT/$f; mkdir -p $R/.venv/bin
  printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec "%s" "$@"\n' "$R" "$PY" > $R/.venv/bin/python; chmod +x $R/.venv/bin/python
  printf 'export PATH="%s/.venv/bin:$PATH"\n' "$R" > $R/.venv/bin/activate
done
