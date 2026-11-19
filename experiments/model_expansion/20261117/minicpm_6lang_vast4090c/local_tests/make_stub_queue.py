"""Writes a test copy of queue_minicpm_6lang.sh in which the heavy step functions (setup_env, setup_6lang, setup_6x6, model, smoke_rel, smoke_expa, consistency, full_rel,
full_expa, retry_expa) are stubs; everything else (GPU/net/disk/sha checks, Qwen-cache rule, step/timeout/ETA handling, failure rules, packaging, status, done marker) is the real script.
Usage: make_stub_queue.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
R = '$R/followup_rel_6lang/results'
def gate(path):
    return 'mkdir -p "$(dirname %s)"; echo \'{"status": "passed", "timing": {"eta_full_hours": 3.6}, "sanity": {"accuracy": 0.7}, "parse_failures": 0}\' > %s' % (path, path)
def one(name, body):
    return name + '() { echo %s >> "$W/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = %s ] && return 1; [ "${FAKE_FAIL_STEP:-}" = %s_eta ] && return 3; [ "${FAKE_FAIL_STEP:-}" = %s_sleep ] && sleep 30; %s; }' % (name, name, name, name, body)
multi = {
 "setup_env": 'setup_env() { echo setup_env >> "$W/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = setup_env ] && return 1; mkdir -p "$R/.cache/pip" "$R/.venv/bin"; touch "$R/.venv/bin/activate"; echo "key packages: 0 differences" > "$RES/freeze_compare.txt"; return 0; }',
 "setup_6lang": one("setup_6lang", ":").replace("}", "}", 1),
 "setup_6x6": one("setup_6x6", ":"),
 "model": 'model() { echo model >> "$W/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = model ] && return 1; mkdir -p "$R/.cache/huggingface"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download openbmb/MiniCPM-V-4_5 --revision 14a2004908140699317e2109e90573c1e6323c36; }',
}
single = {
 "smoke_6x6": one("smoke_6x6", 'mkdir -p "$R/followup_rel_4lang/results/6x6_minicpm_v45"; echo \'{"status": "passed", "eta_hours_full_at_median": 10}\' > "$R/followup_rel_4lang/results/6x6_minicpm_v45/gate_6x6_smoke.json"'),
 "smoke_rel": one("smoke_rel", gate(R + '/rel6new_minicpm_v45/smoke/gate_rel_smoke.json')),
 "smoke_expa": one("smoke_expa", gate(R + '/expA6_minicpm_v45/smoke/gate_smoke.json')),
 "consistency": one("consistency", 'mkdir -p %s/consistency_minicpm_v45; echo \'{"overall": {"label_agreement": 0.99, "old_accuracy": 0.7, "new_accuracy": 0.7}}\' > %s/consistency_minicpm_v45/consistency_agreement.json' % (R, R)),
 "full_rel": one("full_rel", 'mkdir -p %s/rel6new_minicpm_v45/full; echo "{}" > %s/rel6new_minicpm_v45/full/predictions.jsonl' % (R, R)),
 "full_expa": one("full_expa", 'mkdir -p %s/expA6_minicpm_v45/full; echo "{}" > %s/expA6_minicpm_v45/full/predictions.jsonl' % (R, R)),
 "retry_expa": one("retry_expa", 'mkdir -p %s/expA6_minicpm_v45/retry; echo "{}" > %s/expA6_minicpm_v45/retry/predictions.jsonl' % (R, R)),
}
for name, body in multi.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup_env", "setup_6lang", "setup_6x6", "model") and "\n" in "x" else rf"^{name}\(\)\s*\{{.*?^\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
for name, body in single.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{[^\n]*\}}\n", re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
a = 'bash "$W/$SIX_QUEUE" 2>&1'
assert src.count(a) == 1
src = src.replace(a, 'bash "$W/fake_6x6.sh" 2>&1')   # the real queue_6x6.sh is not run in the flow tests (its sha is still checked in preflight)
open(sys.argv[2], "w").write(src)
