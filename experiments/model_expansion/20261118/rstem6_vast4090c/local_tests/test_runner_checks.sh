#!/bin/bash
# Local: runner/gate negative tests and image checks in the test copy q (repository state 6664400 + addon).
# Env: TESTROOT, OUTLOG. Uses the stub venv python of q. Each test prints PASS/FAIL.
TESTROOT=${TESTROOT:?}; OUTLOG=${OUTLOG:?}; mkdir -p "$OUTLOG"
cd "$TESTROOT/q" || exit 2
PY=.venv/bin/python; X=followup_rel_6lang/rstem6; CFG=$X/models/qwen3_vl_8b_4090_channels_last.yaml
W=$OUTLOG/neg; rm -rf "$W"; mkdir -p "$W"
t() { local name=$1; shift; if "$@" > "$W/$name.log" 2>&1; then echo "PASS(exit 0 expected) $name"; else echo "FAIL $name (see $W/$name.log)"; fi; }
tn() { local name=$1; shift; if "$@" > "$W/$name.log" 2>&1; then echo "FAIL(error expected) $name"; else echo "PASS(failed as expected: $(tail -1 "$W/$name.log" | cut -c1-110)) $name"; fi; }
# 1 real image check over all rows: every image of the 4,740 pairs exists with the sha256 of the row
cat > "$W/imgcheck.py" <<'PY'
import sys
sys.path.insert(0, "followup_rel_6lang/rstem6"); sys.path.insert(0, "01_code/src")
import run_rstem6 as r
from pathlib import Path
rows, audit = r.load_rows(r.DEFAULT_INPUT, r.DEFAULT_LOCK, r.DEFAULT_FROZEN, r.REPO, None, True)
print("rows", len(rows), "images checked", audit["images_checked"], audit["equal_question_pairs"])
assert len(rows) == 18960 and audit["images_checked"] == 948
PY
t images_all_sha_ok $PY "$W/imgcheck.py"
# 2 tampered input: one option changed -> sha differs from the lock
cp $X/inputs/rstem6_conditions.jsonl "$W/bad_inputs.jsonl"; sed -i.bak '3s/The /Thx /' "$W/bad_inputs.jsonl"
tn tampered_input_sha $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/o1" --dry-run --inputs "$W/bad_inputs.jsonl"
# 3 consistent tamper (file and lock both changed, lock sha constant in the runner differs): R_stem not R stem + O options
$PY - "$W" <<'PY'
import json, sys, hashlib
w = sys.argv[1]
rows = [json.loads(l) for l in open("followup_rel_6lang/rstem6/inputs/rstem6_conditions.jsonl")]
for r in rows[:4]:
    if r["condition"] == "R_stem":
        r["options"] = dict(r["options"], A=r["options"]["A"] + "x"); r["question"] = r["question_stem"] + " " + " ".join(f"{k}: {r['options'][k]}" for k in "ABCD")
open(w + "/bad2.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
lock = json.load(open("followup_rel_6lang/rstem6/inputs/rstem6_lock.json")); lock["file_sha256"] = hashlib.sha256(open(w + "/bad2.jsonl", "rb").read()).hexdigest()
json.dump(lock, open(w + "/bad2_lock.json", "w"))
PY
# recombination check with the sha constant of the runner patched to the tampered file (PYTHONPATH trick: run in-process)
$PY - "$W" <<'PY' > "$W/recomb.log" 2>&1 && echo "PASS(failed as expected: $(tail -1 "$W/recomb.log" | cut -c1-100)) recombination_check" || echo "FAIL recombination_check"
import sys, json, hashlib
sys.path.insert(0, "followup_rel_6lang/rstem6"); sys.path.insert(0, "01_code/src")
import run_rstem6 as r
from pathlib import Path
w = Path(sys.argv[1])
r.EXPECTED_INPUT_SHA256 = hashlib.sha256((w / "bad2.jsonl").read_bytes()).hexdigest()
try:
    r.load_rows(w / "bad2.jsonl", w / "bad2_lock.json", r.DEFAULT_FROZEN, r.REPO, None, False)
except ValueError as e:
    print("ValueError:", e)
else:
    raise SystemExit("no error")
PY
tn lock_constant_guard $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/o2" --dry-run --inputs "$W/bad2.jsonl" --lock "$W/bad2_lock.json"
# 4 wrong batch size
tn wrong_batch $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/o3" --dry-run --batch-size 1
# 5 config of another model
tn wrong_config $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $X/models/uitars_1_5_7b_4090.yaml --output-dir "$W/o4" --dry-run
# 6 resume: mock run, cut predictions to the first 1,000 lines, rerun with --resume -> complete, no duplicate success per id
$PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/res" --mock-model --max-items 3 > "$W/res_first.log" 2>&1
head -n 509 "$W/res/predictions.jsonl" > "$W/res/p.tmp" && mv "$W/res/p.tmp" "$W/res/predictions.jsonl"   # 509 lines: ends in the middle of a pair
t resume_completes $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/res" --mock-model --max-items 3 --resume
$PY - "$W/res" <<'PY' > "$W/res_check.log" 2>&1 && echo "PASS resume: 360 rows, every pair complete" || echo "FAIL resume check"
import json, sys, collections
ids = [json.loads(l)["input_id"] for l in open(sys.argv[1] + "/predictions.jsonl")]
last = {}
for i in ids: last[i] = 1
by = collections.Counter(i.split("::", 1)[1].rsplit("::", 1)[0] for i in last)
assert len(last) == 3 * 30 * 4 and all(v == 4 for v in by.values()), (len(last), by.most_common(2))
print("ok", len(ids), "lines", len(last), "ids")
PY
tn resume_without_flag $PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/res" --mock-model --max-items 3
# 7 gate negatives on a finished mock smoke: drop one line -> incomplete; and a batch-size mismatch
$PY $X/run_rstem6.py --model qwen3_vl_8b --model-config $CFG --output-dir "$W/g" --mock-model --max-items 2 > "$W/g_first.log" 2>&1
tn gate_wrong_batch $PY $X/check_rstem6_gate.py smoke "$W/g" --model qwen3_vl_8b --batch 1 --mock
tn gate_full_n_rows $PY $X/check_rstem6_gate.py complete "$W/g" --model qwen3_vl_8b --batch 4 --expected-rows 18960 --mock
t gate_ok_smoke $PY $X/check_rstem6_gate.py smoke "$W/g" --model qwen3_vl_8b --batch 4 --mock
sed -i.bak '5d' "$W/g/predictions.jsonl"
tn gate_missing_row $PY $X/check_rstem6_gate.py smoke "$W/g" --model qwen3_vl_8b --batch 4 --mock
# 8 reference agreement logic of the gate (unit test with fabricated rows built from the reference file itself): identical labels -> agreement 1.0, every 4th label changed -> 0.75 (< 0.90 = problem)
cat > "$W/refunit.py" <<'PY'
import sys, json, gzip
sys.path.insert(0, "followup_rel_6lang/rstem6")
import check_rstem6_gate as g
from pathlib import Path
ref = Path("followup_rel_6lang/rstem6/reference/rstem4lang_qwen3_vl_8b_labels.jsonl.gz")
rows = [json.loads(l) for l in gzip.open(ref, "rt")][:400]
ok = [{"pair_id": r["pair_id"], "condition": r["condition"], "parsed_generated_label": r["label"], "generation_correct": r["correct"]} for r in rows]
a = g.reference_check(ok, ref, None)
assert a["label_agreement"] == 1.0 and a["common_rows"] == 400, a
for i, r in enumerate(ok):
    if i % 4 == 0: r["parsed_generated_label"] = "Z"
b = g.reference_check(ok, ref, None)
assert b["label_agreement"] == 0.75, b
print("agreement logic ok", a["label_agreement"], b["label_agreement"])
PY
t reference_agreement_unit $PY "$W/refunit.py"
