"""Unit checks for check_6x6_gate.py with fabricated outputs. Usage: python test_gate.py <gate script> <scratch dir>
Includes the sanity rules added 2026-11-14 (identical labels, A-D mass, token ranges per family, accuracy warning, ETA)."""
import json, math, os, subprocess, sys
from pathlib import Path

gate, tmp = sys.argv[1], Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
N = 216

def make(name, n=N, unparsed=0, batch=1, fail=0, drop=0, extra=0, dup_fail_then_ok=0, labels=None, mass=None, visual=None, prompt=None,
         acc=None, runtime_ms=600):
    ids = [f"id{i}" for i in range(n)]
    (tmp / f"{name}_plan.json").write_text(json.dumps({"input_ids": ids}))
    rows = []
    for i, k in enumerate(ids[: n - drop]):
        r = {"input_id": k, "status": "success", "parse_success": i >= unparsed, "batch_size": batch,
             "dimension": "wf" if i % 2 else "rel", "runtime_ms": runtime_ms}
        if labels is not None: r["parsed_generated_label"] = labels[i % len(labels)]
        if mass is not None: r["label_logprobs"] = {k: math.log(mass) for k in "ABCD"}
        if visual is not None: r["visual_token_count"] = visual[i % len(visual)] if isinstance(visual, list) else visual
        if prompt is not None: r["prompt_token_count"] = prompt
        if acc is not None: r["correct"] = (i % 100) < acc * 100
        if i < fail: r = {"input_id": k, "status": "failed"}
        if i < dup_fail_then_ok: rows.append({"input_id": k, "status": "failed"})
        rows.append(r)
    for j in range(extra): rows.append({"input_id": f"x{j}", "status": "success", "parse_success": True, "batch_size": 1, "runtime_ms": 1})
    (tmp / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

def run(name, mode="smoke", expected=N, model=None):
    cmd = [sys.executable, gate, mode, str(tmp / f"{name}.jsonl"), str(tmp / f"{name}_plan.json"), str(tmp / f"{name}_gate.json"), "--expected", str(expected)]
    if model: cmd += ["--model", model]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, [l for l in p.stdout.strip().splitlines() if l.startswith("GATE")][-1] + "".join(" | " + l for l in p.stdout.splitlines() if l.startswith("WARNING"))

def gj(name): return json.loads((tmp / f"{name}_gate.json").read_text())

cases = [
    ("ok", {}, 0), ("ok_with_resume_retry", dict(dup_fail_then_ok=5), 0),
    ("unparsed_10pct", dict(unparsed=21), 0), ("unparsed_50pct_exactly", dict(unparsed=108), 0),
    ("unparsed_61pct", dict(unparsed=132), 1), ("one_failed_row", dict(fail=1), 1),
    ("missing_rows", dict(drop=3), 1), ("batch3", dict(batch=3), 1), ("extra_ids", dict(extra=2), 1),
]
bad = 0
for name, kw, want in cases:
    make(name, **kw); rc, line = run(name)
    print(("PASS" if rc == want else "FAIL"), name, "exit", rc, "|", line); bad += rc != want
make("wrong_size", n=100); rc, line = run("wrong_size"); print(("PASS" if rc == 1 else "FAIL"), "wrong_size (plan 100 vs expected 216) exit", rc, "|", line); bad += rc != 1
make("full_ok", n=79020); rc, line = run("full_ok", "complete", 79020); print(("PASS" if rc == 0 else "FAIL"), "complete 79020 ok exit", rc, "|", line); bad += rc != 0
make("full_short", n=79020, drop=10); rc, line = run("full_short", "complete", 79020); print(("PASS" if rc == 1 else "FAIL"), "complete 79020 with 10 missing exit", rc, "|", line); bad += rc != 1

# ---- sanity rules (2026-11-14)
def rule(label, name, model, want, expect_text=None, **kw):
    global bad
    make(name, **kw); rc, line = run(name, model=model); g = gj(name)
    ok = rc == want and (expect_text is None or any(expect_text in x for x in g["problems"] + g["warnings"]))
    print(("PASS" if ok else "FAIL"), label, "exit", rc, "|", line, "|", "; ".join(g["problems"])); bad += not ok
    return g
rule("baseline Qwen3 (prompt tokens only, mixed labels)", "r_base", "qwen3_vl_4b", 0, labels=list("ABCD"), mass=0.25, prompt=2100)
rule("all labels identical fails", "r_same", "qwen3_vl_4b", 1, "parsed labels are 'B'", labels=["B"])
rule("two labels pass", "r_two", "qwen3_vl_4b", 0, labels=["A", "B"])
rule("median A-D mass 0.2 fails", "r_mass", "qwen3_vl_4b", 1, "A-D probability mass", labels=list("ABCD"), mass=0.05)
g = rule("label_logprobs missing: reported, passes", "r_nolp", "qwen3_vl_4b", 0, labels=list("ABCD"))
if "not evaluated" not in str(g["sanity"]["median_abcd_mass"]): print("FAIL missing logprobs not reported"); bad += 1
rule("Qwen3 visual 2049 fails", "r_q_vis", "qwen3_vl_2b", 1, "outside 0..2048", visual=2049)
rule("Qwen3 prompt 3000 fails", "r_q_pr", "qwen3_vl_2b", 1, "prompt_token_count outside 800..2700", prompt=3000)
rule("UI-TARS visual 2674 passes", "r_ut_ok", "uitars_15_7b", 0, visual=2674)
rule("UI-TARS visual 2675 fails", "r_ut_bad", "uitars_15_7b", 1, "outside 0..2674", visual=2675)
rule("InternVL 1792/2304 pass", "r_iv_ok", "internvl35_4b", 0, visual=[1792, 2304])
rule("InternVL 1536 fails", "r_iv_bad", "internvl35_4b", 1, "outside 1792..2304", visual=[1792, 1536])
rule("MiniCPM 576/640 pass", "r_mc_ok", "minicpm_v45", 0, visual=[576, 640])
rule("MiniCPM 704 fails", "r_mc_bad", "minicpm_v45", 1, "outside 576..640", visual=704)
g = rule("InternVL visual field missing: reported, passes", "r_iv_novis", "internvl35_1b", 0)
if "not evaluated" not in str(g["sanity"].get("token_range_rule", "")): print("FAIL missing visual not reported"); bad += 1
rule("smoke accuracy 0.10 warns, passes", "r_acc", "qwen3_vl_4b", 0, "smoke accuracy", acc=0.10)
g = rule("smoke accuracy 0.50 no warning", "r_acc2", "qwen3_vl_4b", 0, acc=0.5)
if g["warnings"]: print("FAIL unexpected warning"); bad += 1
g = rule("ETA: 2.0 s x 79,020 = 158,040 s", "r_eta", "qwen3_vl_4b", 0, runtime_ms=2000)
ok = g["eta_full_seconds"] == 158040 and g["eta_hours_full_at_median"] == 43.9
print(("PASS" if ok else "FAIL"), "ETA fields", g["eta_full_seconds"], g["eta_hours_full_at_median"]); bad += not ok
os.environ["GATE_VISUAL_RANGE"] = "500,700"
rule("GATE_VISUAL_RANGE=500,700 override accepts MiniCPM 660", "r_override", "minicpm_v45", 0, visual=660)
sys.exit(1 if bad else 0)
