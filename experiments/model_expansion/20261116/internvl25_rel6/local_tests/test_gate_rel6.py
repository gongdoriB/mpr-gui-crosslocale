"""Unit tests of check_rel6_gate.py with synthetic prediction files (rel kind, 6-language input ids).
usage: python -I test_gate_rel6.py <check_rel6_gate.py> <tmp dir>"""
import json, math, subprocess, sys
from pathlib import Path

gate, tmp = Path(sys.argv[1]), Path(sys.argv[2])
SYS = "<|im_start|>system\nYou are a GUI reasoning assistant. Inspect the GUI image.<|im_end|>\n<|im_start|>user\n<image>\nQ"
ZH = "<|im_start|>system\n你是书生·万象，英文名是InternVL<|im_end|>\n<|im_start|>user\n<image>\nQ"
LANGS = ["en", "zh", "th", "ru", "fr", "ja"]
def rows(n=200, visual=768, prompt=SYS, labels="ABCD", mass=0.99, parse=True, batch=1):
    out = []
    for i in range(n):
        q, g, cond = LANGS[i % 6], LANGS[(i // 6) % 6], "ORCF"[i % 4]
        lab = labels[i % len(labels)]
        lp = {k: -50.0 for k in "ABCD"}; lp[lab] = math.log(mass)
        out.append({"input_id": f"rel_orcf::rel::2/x{i}.png::{q}::{g}::{cond}", "status": "success", "parse_success": parse, "generation_correct": lab == "A",
                    "parsed_generated_label": lab, "batch_size": batch, "label_logprobs": lp, "visual_token_count": visual,
                    "prompt_token_count": 900, "rendered_prompt": prompt, "inference_runtime_ms": 326.0})
    return out
def run(name, rs, expect_pass, mode="smoke", expected=None, full_n=None, full_rows=None, plan_rows=None, summary=None, must_contain=None, check=None):
    d = tmp / name; d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    (d / "plan.json").write_text(json.dumps({"actual_inferences": len(rs), "condition_rows": plan_rows if plan_rows is not None else len(rs) + 10}))
    (d / "summary.json").write_text(json.dumps(summary if summary is not None else {"condition_rows": len(rs) + 10, "statuses": {"success": len(rs) + 10}}))
    cmd = [sys.executable, "-I", str(gate), "rel", mode, str(d)] + ([str(expected)] if expected else []) + ["--model", "internvl25_8b"]
    if full_n: cmd += ["--full-n", str(full_n)]
    if full_rows: cmd += ["--full-rows", str(full_rows)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    g = json.loads((d / f"gate_rel_{mode}.json").read_text())
    ok = (p.returncode == 0) == expect_pass and (must_contain is None or any(must_contain in x for x in g["problems"])) and (check(g) if check else True)
    print(("PASS " if ok else "FAIL ") + name, "| status", g["status"], "|", g["problems"][:1], "| eta_h", g["timing"]["eta_full_hours"])
    return ok
res = [
    run("good_smoke_eta_uses_full_n", rows(), True, full_n=44489, check=lambda g: abs(g["timing"]["eta_full_hours"] - 44489 * 0.326 / 3600) < 0.01),
    run("per_language_report_has_fr_ja", rows(), True, check=lambda g: {"fr", "ja"} <= set(g["by_language(n,correct,unparsed)"]["query"]) and {"fr", "ja"} <= set(g["by_language(n,correct,unparsed)"]["gui"])),
    run("complete_ok", rows(300), True, mode="complete", expected=300, full_n=300, full_rows=310, plan_rows=310, summary={"condition_rows": 310, "statuses": {"success": 310}}),
    run("complete_wrong_condition_rows", rows(300), False, mode="complete", expected=300, full_rows=52572, plan_rows=310, summary={"condition_rows": 310, "statuses": {"success": 310}}, must_contain="condition rows"),
    run("complete_not_all_success", rows(300), False, mode="complete", expected=300, full_rows=310, plan_rows=310, summary={"condition_rows": 310, "statuses": {"success": 305, "failed": 5}}, must_contain="statuses"),
    run("complete_row_count_differs_from_expected", rows(299), False, mode="complete", expected=300, must_contain="expected 300"),
    run("batch_size_not_1", rows(batch=2), False, must_contain="batch sizes"),
    run("visual_too_small", rows(visual=256), False, must_contain="visual_token_count outside 768..1792"),
    run("chinese_default_system", rows(prompt=ZH), False, must_contain="default Chinese system message"),
    run("all_same_label", rows(labels="A"), False, must_contain="parsed labels are"),
    run("low_mass", rows(mass=0.2), False, must_contain="probability mass"),
    run("unparsed_over_half", rows(parse=False), False, must_contain="parse failure rate"),
]
print("failed:", res.count(False)); sys.exit(0 if all(res) else 1)
