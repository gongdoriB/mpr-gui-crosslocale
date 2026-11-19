"""Writes a test copy of queue_rstem6_c.sh in which the heavy per-model step functions (setup, model, smoke, full) are stubs; everything else
(GPU/net/disk/sha checks, step/timeout/ETA handling, failure rules, packaging, cache deletion, status, done marker) is the real script.
Failure switches: FAKE_FAIL=<step>_<KEY> (step: setup, model, smoke, full), FAKE_FAIL=full_eta_<KEY> (exit 3), FAKE_FAIL=full_sleep_<KEY>, FAKE_FAIL=full_cuda_<KEY> (fails and prints a CUDA fatal text).
Usage: make_stub_queue.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
R = '$R/followup_rel_6lang/results/rstem6_$KEY'
def stub(name, body):
    return (name + '() { echo "' + name + '_$KEY" >> "$W/steps_run.txt"; f="${FAKE_FAIL:-}"; '
            '[ "$f" = ' + name + '_$KEY ] && return 1; [ "$f" = ' + name + '_eta_$KEY ] && return 3; [ "$f" = ' + name + '_sleep_$KEY ] && sleep 30; '
            '[ "$f" = ' + name + '_cuda_$KEY ] && { echo "RuntimeError: CUDA error: device-side assert triggered"; return 1; }; ' + body + '; }\n')
GATE = '{"status": "passed", "timing": {"eta_full_hours": 3.6, "median_s_per_inference": 0.6}, "sanity": {"accuracy": 0.7}, "parse_failures": 0, "by_condition(n,correct)": {"O": [10, 7]}}'
bodies = {
 "setup": 'mkdir -p "$R/.git" "$R/01_code" "$R/.venv/bin"; touch "$R/.venv/bin/activate"',
 "model": 'mkdir -p "$R/.cache/huggingface"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download "$MID" --revision "$REV"',
 "smoke": 'mkdir -p "' + R + '/smoke"; echo \'' + GATE + '\' > "' + R + '/smoke/gate_smoke.json"; echo "{}" > "' + R + '/smoke/predictions.jsonl"',
 "full": 'mkdir -p "' + R + '/full"; echo \'' + GATE + '\' > "' + R + '/full/gate_complete.json"; echo "{}" > "' + R + '/full/predictions.jsonl"',
}
import os
keep = os.environ.get("STUB_ONLY", "setup model smoke full").split()   # which step functions are stubbed (default all four)
for name, body in bodies.items():
    if name not in keep:
        continue
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if name in ("setup", "model") else rf"^{name}\(\)\s*\{{[^\n]*\}}\n", re.S | re.M if name in ("setup", "model") else re.M)
    src, n = pat.subn(lambda m: stub(name, body), src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
