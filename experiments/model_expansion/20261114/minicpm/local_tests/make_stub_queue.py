"""Writes a test copy of queue_minicpm.sh: the heavy functions (setup, model, smoke_*, full_*) are replaced by stubs, everything else
(waiting, network check, retry, ETA guard, cache rule, trap, packaging) is the real script. Usage: make_stub_queue.py <queue_minicpm.sh> <out>"""
import re, sys
src = open(sys.argv[1]).read()
stubs = {
 "setup": 'setup() { echo STUB setup; mkdir -p "$R"; [ "${FAKE_FAIL_STEP:-}" = setup ] && return 1; return 0; }',
 "model": 'model() { echo STUB model; mkdir -p "$R/.cache/huggingface" "$R/.cache/pip"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download openbmb/MiniCPM-V-4_5 --revision 14a2004908140699317e2109e90573c1e6323c36; }',
 "smoke_rel": 'smoke_rel() { d="$R/followup_rel_4lang/results/minicpm_rel/smoke"; mkdir -p "$d"; [ "${FAKE_FAIL_STEP:-}" = smoke_rel ] && return 1; echo "{\\"eta_full_seconds\\": ${FAKE_ETA_REL_S:-100}, \\"sanity\\": {\\"accuracy\\": 0.41}, \\"warnings\\": []}" > "$d/gate_smoke.json"; }',
 "smoke_expa": 'smoke_expa() { d="$R/followup_rel_4lang/results/expA_minicpm/smoke"; mkdir -p "$d"; echo "{\\"eta_full_seconds\\": ${FAKE_ETA_EXPA_S:-100}, \\"sanity\\": {\\"accuracy\\": 0.38}, \\"warnings\\": [\\"smoke accuracy 0.2 < 0.3\\"]}" > "$d/gate_smoke.json"; }',
 "full_rel": 'full_rel() { d="$R/followup_rel_4lang/results/minicpm_rel/full"; mkdir -p "$d"; echo STUB full_rel >> "$R/steps_run.txt"; [ "${FAKE_FAIL_STEP:-}" = full_rel ] && return 1; [ "${FAKE_FAIL_STEP:-}" = full_rel_sleep ] && sleep 30; echo \'{"status": "success"}\' > "$d/predictions.jsonl"; }',
 "full_expa": 'full_expa() { d="$R/followup_rel_4lang/results/expA_minicpm/full"; mkdir -p "$d"; echo STUB full_expa >> "$R/steps_run.txt"; echo \'{"status": "success"}\' > "$d/predictions.jsonl"; }',
}
for name, body in stubs.items():
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup", "model") else rf"^{name}\(\)\s*\{{.*?\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
