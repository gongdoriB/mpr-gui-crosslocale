"""Writes a test copy of queue_internvl25_rel6.sh in which setup, model, smoke_rel6 and full_rel6 are replaced by stubs; everything else
(waiting, other-process wait, GPU wait, network check, retry, step/timeout handling, cache rule, trap, packaging) is the real script.
Usage: make_stub_queue.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
OUT = '$R/followup_rel_6lang/results/rel6_internvl25_8b'
stubs = {
 "setup": 'setup() { echo STUB setup; mkdir -p "$R"; [ "${FAKE_FAIL_STEP:-}" = setup ] && return 1; return 0; }',
 "model": 'model() { echo STUB model; mkdir -p "$R/.cache/huggingface" "$R/.cache/pip"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download OpenGVLab/InternVL2_5-8B --revision e9e4c0dc1db56bfab10458671519b7fa3dd29463; }',
 "smoke_rel6": 'smoke_rel6() { d="%s/smoke"; mkdir -p "$d"; [ "${FAKE_FAIL_STEP:-}" = smoke_rel6 ] && return 1; echo "{\\"timing\\": {\\"eta_full_hours\\": 4.0}, \\"sanity\\": {\\"accuracy\\": 0.41}, \\"parse_failures\\": 0, \\"warnings\\": [], \\"problems\\": []}" > "$d/gate_rel_smoke.json"; }' % OUT,
 "full_rel6": 'full_rel6() { d="%s/full"; mkdir -p "$d"; echo STUB full_rel6 >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = full_rel6 ] && return 1; [ "${FAKE_FAIL_STEP:-}" = full_rel6_eta ] && return 3; [ "${FAKE_FAIL_STEP:-}" = full_rel6_sleep ] && sleep 30; echo \'{"status": "success"}\' > "$d/predictions.jsonl"; }' % OUT,
}
for name, body in stubs.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup", "model") else rf"^{name}\(\)\s*\{{.*?\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
