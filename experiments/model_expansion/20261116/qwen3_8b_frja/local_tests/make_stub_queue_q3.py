"""Writes a test copy of queue_qwen3_8b_frja.sh in which setup, model, smoke_rel, smoke_expa, consistency, full_rel, full_expa and retry_expa are replaced by stubs; everything
else (GPU check, network check, retry, step/timeout handling, nonfatal rule, packaging, cache rule, trap, order) is the real script.
Usage: make_stub_queue_q3.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
R = '$R/followup_rel_6lang/results'
def gate(path, extra=""):
    return 'mkdir -p "$(dirname %s)"; echo \'{"status": "passed", "timing": {"eta_full_hours": 4.0}, "sanity": {"accuracy": 0.41}, "parse_failures": 0, "warnings": [], "problems": []%s}\' > %s' % (path, extra, path)
stubs = {
 "setup": 'setup() { echo STUB setup; mkdir -p "$R"; [ "${FAKE_FAIL_STEP:-}" = setup ] && return 1; return 0; }',
 "model": 'model() { echo STUB model; mkdir -p "$R/.cache/huggingface" "$R/.cache/pip"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; }',
}
def one(name, order, body):
    return name + '() { echo %s >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = %s ] && return 1; [ "${FAKE_FAIL_STEP:-}" = %s_eta ] && return 3; [ "${FAKE_FAIL_STEP:-}" = %s_sleep ] && sleep 30; %s; }' % (name, name, name, name, body)
stubs["smoke_rel"] = one("smoke_rel", 1, gate(R + '/rel6new_qwen3_vl_8b/smoke/gate_rel_smoke.json'))
stubs["smoke_expa"] = one("smoke_expa", 2, gate(R + '/expA6_qwen3_vl_8b/smoke/gate_smoke.json'))
stubs["consistency"] = one("consistency", 3, 'mkdir -p %s/consistency_qwen3_vl_8b; echo \'{"overall": {"label_agreement": 0.98, "old_accuracy": 0.7, "new_accuracy": 0.69}}\' > %s/consistency_qwen3_vl_8b/consistency_agreement.json' % (R, R))
stubs["full_rel"] = one("full_rel", 4, 'mkdir -p %s/rel6new_qwen3_vl_8b/full; echo "{}" > %s/rel6new_qwen3_vl_8b/full/predictions.jsonl' % (R, R))
stubs["full_expa"] = one("full_expa", 5, 'mkdir -p %s/expA6_qwen3_vl_8b/full; echo "{}" > %s/expA6_qwen3_vl_8b/full/predictions.jsonl' % (R, R))
stubs["retry_expa"] = one("retry_expa", 6, 'mkdir -p %s/expA6_qwen3_vl_8b/retry; echo "{}" > %s/expA6_qwen3_vl_8b/retry/predictions.jsonl' % (R, R))
for name, body in stubs.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup", "model") else rf"^{name}\(\)\s*\{{[^\n]*\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
