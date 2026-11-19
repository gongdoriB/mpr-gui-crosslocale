"""Writes a test copy of queue_internvl25_expA.sh in which setup, model, smoke_expa and full_expa are replaced by stubs; everything else
(waiting, other-process wait, GPU wait, network check, retry, step/timeout handling, cache rule, trap, packaging) is the real script.
Usage: make_stub_queue.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
stubs = {
 "setup": 'setup() { echo STUB setup; mkdir -p "$R"; [ "${FAKE_FAIL_STEP:-}" = setup ] && return 1; return 0; }',
 "model": 'model() { echo STUB model; mkdir -p "$R/.cache/huggingface" "$R/.cache/pip"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download OpenGVLab/InternVL2_5-8B --revision e9e4c0dc1db56bfab10458671519b7fa3dd29463; }',
 "smoke_expa": 'smoke_expa() { d="$R/followup_rel_4lang/results/expA_internvl25_8b/expa_smoke"; mkdir -p "$d"; [ "${FAKE_FAIL_STEP:-}" = smoke_expa ] && return 1; echo "{\\"timing\\": {\\"eta_full_hours\\": 0.5}, \\"sanity\\": {\\"accuracy\\": 0.41}, \\"warnings\\": [], \\"problems\\": []}" > "$d/gate_expa_smoke.json"; }',
 "full_expa": 'full_expa() { d="$R/followup_rel_4lang/results/expA_internvl25_8b/expa_full"; mkdir -p "$d"; echo STUB full_expa >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = full_expa ] && return 1; [ "${FAKE_FAIL_STEP:-}" = full_expa_eta ] && return 3; [ "${FAKE_FAIL_STEP:-}" = full_expa_sleep ] && sleep 30; echo \'{"status": "success"}\' > "$d/predictions.jsonl"; }',
}
for name, body in stubs.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup", "model") else rf"^{name}\(\)\s*\{{.*?\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
