"""Test copy of queue_b2_step1_ivl35_rel.sh with setup, model, smoke and full replaced by stubs (everything else is the real script). Usage: make_stub_step1.py <queue> <out>"""
import re, sys
src = open(sys.argv[1]).read()
OUT = '$R/followup_rel_6lang/results/rel6new_internvl35_8b'
stubs = {
 "setup": 'setup() { echo STUB setup; mkdir -p "$R"; [ "${FAKE_FAIL_STEP:-}" = setup ] && return 1; return 0; }',
 "model": 'model() { echo STUB model; mkdir -p "$R/.cache/huggingface"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download OpenGVLab/InternVL3_5-8B --revision 9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79; }',
 "smoke": 'smoke() { d="%s/smoke"; mkdir -p "$d"; echo smoke >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = smoke ] && return 1; [ "${FAKE_FAIL_STEP:-}" = smoke_sleep ] && sleep 30; echo "{\\"timing\\": {\\"eta_full_hours\\": 4.0}, \\"sanity\\": {\\"accuracy\\": 0.41}, \\"parse_failures\\": 0, \\"warnings\\": [], \\"problems\\": []}" > "$d/gate_rel_smoke.json"; }' % OUT,
 "full": 'full() { d="%s/full"; mkdir -p "$d"; echo full >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = full ] && return 1; [ "${FAKE_FAIL_STEP:-}" = full_eta ] && return 3; [ "${FAKE_FAIL_STEP:-}" = full_sleep ] && sleep 30; echo "{}" > "$d/predictions.jsonl"; }' % OUT,
}
for name, body in stubs.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
