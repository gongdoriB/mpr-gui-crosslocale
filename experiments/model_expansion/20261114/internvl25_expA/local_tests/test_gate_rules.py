"""Unit tests of check_generic_gate.py (internvl25 rules) with synthetic prediction files.
usage: python -I test_gate_rules.py <check_generic_gate.py> <tmp dir>"""
import json, subprocess, sys
from pathlib import Path

gate, tmp = Path(sys.argv[1]), Path(sys.argv[2])
SYS = "<|im_start|>system\nYou are a GUI reasoning assistant. Inspect the GUI image.<|im_end|>\n<|im_start|>user\n<image>\nQ"
ZH = "<|im_start|>system\n你是书生·万象，英文名是InternVL<|im_end|>\n<|im_start|>user\n<image>\nQ"
def rows(n=108, visual=768, prompt=SYS, labels="ABCD", mass=0.99, parse=True):
    out = []
    for i in range(n):
        cond = "OTR"[i % 3]; lab = labels[i % len(labels)]
        lp = {k: -50.0 for k in "ABCD"}; lp[lab] = __import__("math").log(mass)
        out.append({"input_id": f"rel_expA::p{i//3}::{cond}", "status": "success", "parse_success": parse, "generation_correct": lab == "A",
                    "parsed_generated_label": lab, "batch_size": 1, "label_logprobs": lp, "visual_token_count": visual,
                    "prompt_token_count": 900, "rendered_prompt": prompt, "inference_runtime_ms": 300.0, "condition": cond})
    return out
def run(name, rs, expect_pass, must_contain=None):
    d = tmp / name; d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    (d / "plan.json").write_text(json.dumps({"inferences": len(rs)}))
    p = subprocess.run([sys.executable, "-I", str(gate), "expa", "smoke", str(d), str(len(rs)), "--model", "internvl25_8b"], capture_output=True, text=True)
    g = json.loads((d / "gate_expa_smoke.json").read_text())
    ok = (p.returncode == 0) == expect_pass and (must_contain is None or any(must_contain in x for x in g["problems"]))
    print(("PASS " if ok else "FAIL ") + name, "| status", g["status"], "|", g["problems"][:1], "| eta_h", g["timing"]["eta_full_hours"])
    return ok
res = [
    run("good_768", rows(), True),
    run("good_1792", rows(visual=1792), True),
    run("visual_too_small", rows(visual=256), False, "visual_token_count outside 768..1792"),
    run("visual_too_big", rows(visual=2304), False, "visual_token_count outside"),
    run("chinese_default_system", rows(prompt=ZH), False, "default Chinese system message"),
    run("all_same_label", rows(labels="A"), False, "parsed labels are"),
    run("low_mass", rows(mass=0.2), False, "probability mass"),
]
print("failed:", res.count(False)); sys.exit(0 if all(res) else 1)
