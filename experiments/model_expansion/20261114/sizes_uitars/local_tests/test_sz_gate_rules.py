"""Unit tests for the sanity rules of check_generic_gate.py (fabricated outputs).
Usage: python -I test_sz_gate_rules.py <check_generic_gate.py> <scratch dir>
"""
import json, math, subprocess, sys
from pathlib import Path

gate, tmp = sys.argv[1], Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
N = 80

def lp(mass_each):  # four equal logprobs whose probabilities sum to 4*mass_each
    return {k: math.log(mass_each) for k in "ABCD"}

def make(name, kind="rel", n=N, labels=None, mass=0.25, visual=None, prompt=None, acc_rate=0.5, drop_lp=False,
         runtime_ms=600, parse_fail=0):
    d = tmp / name; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        lab = labels[i % len(labels)] if labels else "ABCD"[i % 4]
        r = {"input_id": f"{kind}::{i}", "status": "success", "parse_success": i >= parse_fail, "batch_size": 1, "batch_id": f"b{i}",
             "parsed_generated_label": lab if i >= parse_fail else None,
             "generation_correct": (i % 100) < acc_rate * 100, "inference_runtime_ms": runtime_ms,
             "condition": "OTR"[i % 3]}
        if not drop_lp:
            r["label_logprobs"] = lp(mass)
        if visual is not None:
            r["visual_token_count"] = visual[i % len(visual)] if isinstance(visual, list) else visual
        if prompt is not None:
            r["prompt_token_count"] = prompt[i % len(prompt)] if isinstance(prompt, list) else prompt
        rows.append(r)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    plan = {"actual_inferences": n} if kind == "rel" else {"inferences": n}
    (d / "plan.json").write_text(json.dumps(plan))
    return d

def run(d, kind, mode, key, expected=None):
    cmd = [sys.executable, "-I", gate, kind, mode, str(d)] + ([str(expected)] if expected else []) + ["--model", key]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        res = json.loads((d / f"gate_{kind}_{mode}.json").read_text())
    except Exception:
        res = {"problems": [p.stderr[-300:]], "warnings": []}
    return p.returncode, res

bad = 0
def check(name, rc, want, res, must_contain=None, warn_contain=None):
    global bad
    ok = rc == want
    if must_contain: ok = ok and any(must_contain in x for x in res["problems"])
    if warn_contain: ok = ok and any(warn_contain in x for x in res["warnings"])
    if warn_contain == "": ok = ok and not res["warnings"]
    print(("PASS" if ok else "FAIL"), name, "exit", rc, "|", "; ".join(res["problems"] + ["W:" + w for w in res["warnings"]]) or "-")
    bad += not ok

# baseline
rc, res = run(make("base_q", visual=None, prompt=2100), "rel", "smoke", "qwen3_vl_4b", N); check("baseline Qwen3 passes (prompt tokens only)", rc, 0, res, warn_contain="")
# rule 1: identical labels
rc, res = run(make("same", labels=["B"]), "rel", "smoke", "qwen3_vl_4b", N); check("all labels identical fails", rc, 1, res, "parsed labels are 'B'")
rc, res = run(make("same_small", n=10, labels=["B"]), "rel", "smoke", "qwen3_vl_4b", 10); check("10 identical labels: rule not evaluated (needs 20)", rc, 0, res)
rc, res = run(make("two_labels", labels=["A", "B"]), "rel", "smoke", "qwen3_vl_4b", N); check("two distinct labels pass", rc, 0, res)
# rule 2: A-D mass
rc, res = run(make("lowmass", mass=0.05), "rel", "smoke", "qwen3_vl_4b", N); check("median A-D mass 0.2 fails", rc, 1, res, "A-D probability mass")
rc, res = run(make("mass_edge", mass=0.125), "rel", "smoke", "qwen3_vl_4b", N); check("median mass exactly 0.5 passes", rc, 0, res)
rc, res = run(make("nolp", drop_lp=True), "rel", "smoke", "qwen3_vl_4b", N); check("label_logprobs missing: reported, not failed", rc, 0, res)
print("   sanity info (nolp):", res["sanity"]["median_abcd_mass"])
if "not evaluated" not in str(res["sanity"]["median_abcd_mass"]): print("FAIL missing field not reported"); bad += 1
# rule 3: tokens
rc, res = run(make("q_vis_ok", visual=2048), "rel", "smoke", "qwen3_vl_2b", N); check("Qwen3 visual 2048 passes", rc, 0, res)
rc, res = run(make("q_vis_bad", visual=2049), "rel", "smoke", "qwen3_vl_2b", N); check("Qwen3 visual 2049 fails", rc, 1, res, "visual_token_count outside")
rc, res = run(make("q_prompt_bad", prompt=3000), "rel", "smoke", "qwen3_vl_4b", N); check("Qwen3 prompt 3000 fails (no visual field)", rc, 1, res, "prompt_token_count outside")
rc, res = run(make("q_prompt_low", prompt=300), "rel", "smoke", "qwen3_vl_4b", N); check("Qwen3 prompt 300 (image not fed) fails", rc, 1, res, "prompt_token_count outside")
rc, res = run(make("ut_ok", visual=2674), "rel", "smoke", "uitars_15_7b", N); check("UI-TARS visual 2674 passes", rc, 0, res)
rc, res = run(make("ut_bad", visual=2675), "rel", "smoke", "uitars_15_7b", N); check("UI-TARS visual 2675 fails", rc, 1, res, "visual_token_count outside")
rc, res = run(make("ivl_ok", visual=[1792, 2304]), "rel", "smoke", "internvl35_2b", N); check("InternVL 1792/2304 pass", rc, 0, res)
rc, res = run(make("ivl_low", visual=[1792, 2304, 1536]), "rel", "smoke", "internvl35_2b", N); check("InternVL 1536 fails", rc, 1, res, "outside 1792..2304")
rc, res = run(make("ivl_hi", visual=2560), "rel", "smoke", "internvl35_1b", N); check("InternVL 2560 fails", rc, 1, res, "outside 1792..2304")
rc, res = run(make("ivl_novis"), "rel", "smoke", "internvl35_4b", N); check("InternVL without visual field: reported, not failed", rc, 0, res)
if "not evaluated" not in str(res["sanity"].get("token_range_rule", "")): print("FAIL missing visual field not reported"); bad += 1
# accuracy warning
rc, res = run(make("lowacc", acc_rate=0.10), "rel", "smoke", "qwen3_vl_4b", N); check("smoke accuracy 0.10 warns, still passes", rc, 0, res, warn_contain="smoke accuracy")
rc, res = run(make("acc030", acc_rate=0.30), "rel", "smoke", "qwen3_vl_4b", N); check("smoke accuracy 0.30 no warning", rc, 0, res, warn_contain="")
rc, res = run(make("lowacc_full", acc_rate=0.10), "rel", "complete", "qwen3_vl_4b", N); check("complete mode: low accuracy does not warn", rc, 0, res, warn_contain="")
# eta
d = make("eta", runtime_ms=2000); rc, res = run(d, "rel", "smoke", "qwen3_vl_4b", N)
t = res["timing"]; ok = t["median_s_per_inference"] == 2.0 and t["eta_full_seconds"] == 36864 and t["full_inferences"] == 18432
print(("PASS" if ok else "FAIL"), "ETA rel: 2.0 s x 18432 =", t.get("eta_full_seconds")); bad += not ok
d = make("eta_expa", kind="expa", runtime_ms=1000); rc, res = run(d, "expa", "smoke", "qwen3_vl_4b", N)
t = res["timing"]; ok = t["eta_full_seconds"] == 5688 and t["full_inferences"] == 5688
print(("PASS" if ok else "FAIL"), "ETA expa: 1.0 s x 5688 =", t.get("eta_full_seconds")); bad += not ok
# old rules still fire
rc, res = run(make("parsefail", parse_fail=60), "rel", "smoke", "qwen3_vl_4b", N); check("parse failure rate 75% still fails", rc, 1, res, "parse failure rate")
rc, res = run(make("short"), "rel", "smoke", "qwen3_vl_4b", N + 5); check("missing rows still fail", rc, 1, res, "expected")
rc, res = run(make("nokey"), "rel", "smoke", "someothermodel", N); check("unknown model key: token rule not evaluated, passes", rc, 0, res)
sys.exit(1 if bad else 0)
