"""Test copy of queue_b3_4b.sh in which setup, model, smoke_rel, full_rel, smoke_expa, full_expa, retry_expa, six_setup and six_run are replaced by stubs; everything else (wait, guards, step/timeout handling,
failure classification, packaging, cache rule, trap, order) is the real script. Stub controls (env): FAKE_FAIL="name:KEY ..." (exit 1), FAKE_ETA="name:KEY" (exit 3), FAKE_SLEEP="name:KEY" (sleep 30),
FAKE_CUDA="name:KEY" (prints a CUDA fatal error text and exits 1), FAKE_BUSY_AFTER=1 (a failing stub leaves the fake GPU busy). Usage: make_stub_b3.py <queue script> <out>"""
import re, sys
src = open(sys.argv[1]).read()
GATE = '{"status": "passed", "timing": {"eta_full_hours": 4.0}, "sanity": {"accuracy": 0.41}, "parse_failures": 0, "warnings": [], "problems": []}'
R = '$R/followup_rel_6lang/results'
def ctl(name):
    return (f'echo "{name} $KEY" >> "$W/steps_run.txt"; ' f'case " ${{FAKE_SLEEP:-}} " in *" {name}:$KEY "*) sleep 30;; esac; '
            f'case " ${{FAKE_CUDA:-}} " in *" {name}:$KEY "*) echo "RuntimeError: CUDA error: an illegal memory access was encountered"; return 1;; esac; '
            f'case " ${{FAKE_ETA:-}} " in *" {name}:$KEY "*) return 3;; esac; '
            f'case " ${{FAKE_FAIL:-}} " in *" {name}:$KEY "*) [ -n "${{FAKE_BUSY_AFTER:-}}" ] && touch "$FAKE_STATE/busy"; return 1;; esac; ')
def gate(path):
    return f"mkdir -p \"$(dirname {path})\"; echo '{GATE}' > {path}; "
SIX = ('mkdir -p "$W/queue_results/6x6_$KEY" "$R/followup_rel_4lang/results/6x6_$KEY"; echo "{}" > "$R/followup_rel_4lang/results/6x6_$KEY/full_6x6.jsonl"; '
       'tar -czf "$W/queue_results/6x6_$KEY/6x6_${KEY}_test.tar.gz" -C "$R/followup_rel_4lang/results" "6x6_$KEY"; (cd "$W/queue_results/6x6_$KEY" && sha256sum ./*.tar.gz > SHA256SUMS); '
       'touch "$W/queue_results/6x6_$KEY/COMPLETE"; ')
stubs = {
 "setup": 'setup() { ' + ctl("setup") + 'echo STUB setup; }',
 "model": 'model() { ' + ctl("model") + 'echo STUB model; mkdir -p "$R/.cache/huggingface"; export HF_HOME="$R/.cache/huggingface"; retry 3 hf download "$MID" --revision "$REV"; }',
 "smoke_rel": 'smoke_rel() { ' + ctl("smoke_rel") + gate(R + '/rel6new_$KEY/smoke/gate_rel_smoke.json') + '}',
 "full_rel": 'full_rel() { ' + ctl("full_rel") + gate(R + '/rel6new_$KEY/full/gate_rel_complete.json') + 'echo "{}" > "' + R + '/rel6new_$KEY/full/predictions.jsonl"; }',
 "smoke_expa": 'smoke_expa() { ' + ctl("smoke_expa") + gate(R + '/expA6_$KEY/smoke/gate_smoke.json') + '}',
 "full_expa": 'full_expa() { ' + ctl("full_expa") + gate(R + '/expA6_$KEY/full/gate_complete.json') + 'echo "{}" > "' + R + '/expA6_$KEY/full/predictions.jsonl"; }',
 "retry_expa": 'retry_expa() { ' + ctl("retry_expa") + gate(R + '/expA6_$KEY/retry/gate_complete.json') + '}',
 "six_setup": 'six_setup() { ' + ctl("six_setup") + 'echo STUB six_setup; }',
 "six_run": 'six_run() { ' + ctl("six_run") + SIX + '}',
}
for name, body in stubs.items():
    multi = name in ("setup", "model", "six_setup", "six_run")
    pat = re.compile(rf"^{name}\(\)\s*\{{.*?^\}}\n" if multi else rf"^{name}\(\)\s*\{{[^\n]*\}}\n", re.S | re.M)
    src, n = pat.subn(lambda m: body + "\n", src, count=1)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
