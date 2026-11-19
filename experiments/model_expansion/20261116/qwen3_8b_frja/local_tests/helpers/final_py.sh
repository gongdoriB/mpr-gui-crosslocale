#!/usr/bin/env bash
# final run of the python tests in a FRESH workspace made from the template + the built addon (REL full mock first, as the tests need its predictions)
S=${TMPDIR}/work
O="${PROJECT_ROOT}/model_expansion_20261116/qwen3_8b_frja"
W=$S/q3frja/final_py_ws
rm -rf $W; mkdir -p $W; cp -Rl $S/q3frja/template $W/mpr-gui-crosslocale; R=$W/mpr-gui-crosslocale
rm $R/.venv/bin/python $R/.venv/bin/activate
printf '#!/bin/bash\nPYTHONPATH="%s/01_code/src" exec %s/mc/venv/bin/python "$@"\n' "$R" "$S" > $R/.venv/bin/python; chmod +x $R/.venv/bin/python
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$R" > $R/.venv/bin/activate
cd $R || exit 1
tar -xzf "$O/qwen3_8b_frja_addon_20261116.tar.gz"
mkdir -p followup_rel_6lang; cp -Rl "${PROJECT_ROOT}/followup_rel_6lang/assets" followup_rel_6lang/assets
export LOCAL_MOCK=1 EXPECT_TORCH=2.9.1
bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh smoke > /dev/null 2>&1 && bash followup_rel_6lang/q3_frja/run_rel6_qwen3.sh full > /dev/null 2>&1 || echo "REL mock run failed"
export Q3WS=$PWD Q3ADDON="$O/addon" Q3TMP=$S/q3frja/final_py_tmp
rm -rf $Q3TMP
$S/mc/venv/bin/python "$O/local_tests/test_q3_equivalence.py" > "$S/q3frja/logs/py_final.log" 2>&1
echo "PY DONE" >> "$S/q3frja/logs/py_final.log"
