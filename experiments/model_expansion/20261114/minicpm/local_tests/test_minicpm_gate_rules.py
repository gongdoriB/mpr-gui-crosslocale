"""Unit tests for the sanity rules of check_minicpm_gates.py (REL) and check_T_gate.py generic mode (Exp A), fabricated outputs.
Usage: python -I test_minicpm_gate_rules.py <runners dir> <scratch dir>
"""
import json, math, os, subprocess, sys
from pathlib import Path

runners, tmp = Path(sys.argv[1]), Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
N = 80
bad = 0

def make(name, n=N, labels=None, mass=0.25, visual=576, acc_rate=0.5, drop_lp=False, runtime_ms=600, parse_fail=0, drop_visual=False):
    d = tmp / name; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        lab = labels[i % len(labels)] if labels else "ABCD"[i % 4]
        r = {"input_id": f"x::{i}", "status": "success", "parse_success": i >= parse_fail, "batch_size": 1, "batch_id": f"b{i}",
             "parsed_generated_label": lab if i >= parse_fail else None,
             "generation_correct": (i % 100) < acc_rate * 100, "inference_runtime_ms": runtime_ms, "prompt_token_count": 700,
             "label_scoring_method": "next_token_single_label", "generation_scoring_disagreement": False,
             "condition": "OTR"[i % 3], "pair_id": f"p{i // 3}", "rendered_prompt": f"q{i}"}
        if not drop_lp: r["label_logprobs"] = {k: math.log(mass) for k in "ABCD"}
        if not drop_visual: r["visual_token_count"] = visual[i % len(visual)] if isinstance(visual, list) else visual
        rows.append(r)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "plan.json").write_text(json.dumps({"actual_inferences": n, "condition_rows": n}))
    (d / "summary.json").write_text(json.dumps({"statuses": {"success": n}, "complete": True, "accuracy": {}, "primary_descriptive_summary": None}))
    return d

def run(d, which, mode="smoke"):
    if which == "rel": cmd = [sys.executable, "-I", str(runners / "check_minicpm_gates.py"), "rel", mode, str(d)]
    else: cmd = [sys.executable, "-I", str(runners / "check_T_gate.py"), "generic", mode, str(d)]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **({"GATE_VISUAL_RANGE": os.environ["TEST_RANGE"]} if os.environ.get("TEST_RANGE") else {})})
    try: res = json.loads((d / f"gate_{mode}.json").read_text())
    except Exception: res = {"problems": [p.stderr[-300:]], "warnings": []}
    return p.returncode, res

def check(name, rc, want, res, must=None, warn=None):
    global bad
    ok = rc == want
    if must: ok = ok and any(must in x for x in res["problems"])
    if warn: ok = ok and any(warn in x for x in res["warnings"])
    if warn == "": ok = ok and not res["warnings"]
    print(("PASS" if ok else "FAIL"), name, "exit", rc, "|", "; ".join(res["problems"] + ["W:" + w for w in res["warnings"]]) or "-")
    bad += not ok

for which in ("rel", "expa"):
    print(f"--- {which}")
    def go(tag, mode="smoke", **kw):
        d = make(f"{which}_{tag}", **kw); return run(d, which, mode)
    rc, res = go("base"); check("baseline (visual 576) passes", rc, 0, res, warn="")
    rc, res = go("vis640", visual=[576, 640]); check("visual 576/640 pass", rc, 0, res)
    rc, res = go("vis575", visual=[576, 575]); check("visual 575 fails", rc, 1, res, "outside 576..640")
    rc, res = go("vis641", visual=641); check("visual 641 fails", rc, 1, res, "outside 576..640")
    rc, res = go("novis", drop_visual=True); check("visual_token_count missing: reported, not failed", rc, 0, res)
    if "not evaluated" not in str(res["sanity"].get("token_range_rule", "")): print("FAIL missing visual field not reported"); bad += 1
    rc, res = go("same", labels=["C"]); check("all labels identical fails", rc, 1, res, "parsed labels are 'C'")
    rc, res = go("lowmass", mass=0.1); check("median A-D mass 0.4 fails", rc, 1, res, "A-D probability mass")
    rc, res = go("nolp", drop_lp=True); check("label_logprobs missing: reported, not failed", rc, 0, res)
    if "not evaluated" not in str(res["sanity"]["median_abcd_mass"]): print("FAIL missing logprobs not reported"); bad += 1
    rc, res = go("lowacc", acc_rate=0.1); check("smoke accuracy < 0.30 warns, passes", rc, 0, res, warn="smoke accuracy")
    rc, res = go("lowacc_c", mode="complete", n=18432 if which == "rel" else 5688, acc_rate=0.1); check("complete mode no accuracy warning", rc, 0, res, warn="")
    rc, res = go("eta", runtime_ms=2000)
    want = 36864 if which == "rel" else 11376
    ok = res.get("eta_full_seconds") == want
    print(("PASS" if ok else "FAIL"), f"ETA {which}: 2.0 s x full = {res.get('eta_full_seconds')} (want {want})"); bad += not ok
    rc, res = go("parsefail", parse_fail=60); check("parse failure 75% still fails", rc, 1, res)
os.environ["TEST_RANGE"] = "500,700"
d = make("rel_override", visual=660); rc, res = run(d, "rel"); check("GATE_VISUAL_RANGE=500,700 override accepts 660", rc, 0, res)
sys.exit(1 if bad else 0)
