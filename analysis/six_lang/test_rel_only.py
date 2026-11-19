#!/usr/bin/env python3
"""REL-only mode == full mode for every REL output (exact equality). Cases: synthetic merged 6-lang, synthetic full 6-lang,
and the two real 4-language equivalence cases (MiniCPM-V-4.5, Qwen3-VL-8B). Needs synthetic/ (python3 -I make_synthetic.py synthetic).
Full-mode references are re-run here with the current driver. Writes tests/rel_only_report.json."""
import json, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = Path("${PROJECT_ROOT}")
S = HERE / "synthetic"; O = HERE / "tests/rel_only"; shutil.rmtree(O, ignore_errors=True); O.mkdir(parents=True)
DRV = str(HERE / "run_6lang_analysis.py"); rep = []

def run(args):
    r = subprocess.run([sys.executable, "-I", DRV, *args], capture_output=True, text=True); return r
def eq(a, b, path, d):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): d.append(f"{path} keys {sorted(set(a) ^ set(b))}")
        for k in a:
            if k in b: eq(a[k], b[k], f"{path}/{k}", d)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): d.append(f"{path} len")
        for i, (x, y) in enumerate(zip(a, b)): eq(x, y, f"{path}[{i}]", d)
    elif a != b and not (a != a and b != b): d.append(f"{path}: {a!r} != {b!r}")
def relonly(lst): return [x for x in lst if x.get("exp") == "REL"]
def compare(name, fdir, rdir):
    F = json.loads((fdir / "results.json").read_text()); Rr = json.loads((rdir / "results.json").read_text()); d = []
    ok_flags = Rr["rel_only"] is True and Rr["expa_status"].startswith("SKIPPED") and F["rel_only"] is False
    noexp = not any((k.startswith("expa") and k != "expa_status") or k == "expA_analyze" for k in Rr) and not (rdir / "expA_analyze").exists() and not (rdir / "provenance_expa_rows.csv").exists()
    for k in ("rel", "rel_counts", "rel_new_rows_only_exploratory", "rel_glmm", "thai_indicator_a", "thai_flag_a", "four_lang_count_consistency", "validated_summaries"):
        if k in F or k in Rr: eq(F.get(k), Rr.get(k), k, d)
    if "four_lang" in F:
        eq({k: v for k, v in F["four_lang"].items() if not k.startswith("expa")}, Rr["four_lang"], "four_lang", d)
    eq([x for x in F["parse_failures"] if x["experiment"] == "REL"], Rr["parse_failures"], "parse_failures", d)
    eq({k: v for k, v in F["parse_failure_totals"].items() if k.startswith("REL")}, Rr["parse_failure_totals"], "pf_totals", d)
    eq(F["agreement"]["REL(condition rows)"], Rr["agreement"]["REL(condition rows)"], "agreement", d)
    eq(F["tokens"]["REL"], Rr["tokens"]["REL"], "tokens", d)
    for k in ("replication", "thai_exclusion_changes", "scoring_vs_generation_decisions"): eq(relonly(F[k]), Rr[k], k, d)
    if "holm_decision_four_vs_six" in F:
        eq(relonly(F["holm_decision_four_vs_six"]["rows"]), Rr["holm_decision_four_vs_six"]["rows"], "holm4v6", d)
        eq([x for x in F["new_rows_only_sign_agreement"] if x["exp"] == "REL"], Rr["new_rows_only_sign_agreement"], "signagree", d)
    fc = [c for c in F["precheck"]["checks"] if not c["name"].startswith("expA") and not c["name"].startswith("inputs:") or c["name"].startswith("inputs: ") and "pairs" not in c["name"] and "collision" not in c["name"] and "sub2" not in c["name"] and "source-issue" not in c["name"] and "SHA" not in c["name"] and "first_spans" not in c["name"] and "excluded" not in c["name"] and "T conditions" not in c["name"]]
    fc = [c for c in F["precheck"]["checks"] if c["name"].startswith("results") or c["name"].startswith("input: ")]
    rc = [c for c in Rr["precheck"]["checks"]]
    eq(fc, rc, "precheck(REL part)", d)
    prov = (fdir / "provenance_rel_rows.csv").read_bytes() == (rdir / "provenance_rel_rows.csv").read_bytes()
    ok = ok_flags and noexp and not d and prov
    rep.append(dict(case=name, ok=ok, flags=ok_flags, no_expa_outputs=noexp, provenance_csv_identical=prov, n_differences=len(d), first=d[:4], rel_checks=len(rc)))
    print("PASS" if ok else "FAIL", name, "diffs", len(d), "prov", prov, "rel_checks", len(rc), d[:3])

# synthetic merged
SY = ["--model", "synth", "--mode", "6"]
ex = O / "ex_old"; assert run(["--model", "synth", "--mode", "4", "--out", str(ex), "--rel-old", str(S / "rel_old.jsonl"), "--expa-old", str(S / "expa_old.jsonl")]).returncode == 0
relargs = ["--rel-old", str(S / "rel_old.jsonl"), "--rel-new", str(S / "rel_new.jsonl"), "--existing-json", str(ex / "results.json")]
f = O / "m_full"; assert run(SY + ["--out", str(f), *relargs, "--expa-old", str(S / "expa_old.jsonl"), "--expa-new", str(S / "expa_new.jsonl"), "--existing-expa-csv", str(ex / "expA_analyze/contrasts.csv")]).returncode == 0
r = O / "m_rel"; assert run(SY + ["--out", str(r), *relargs]).returncode == 0
compare("synthetic merged 6-lang (old+new)", f, r)
f = O / "f_full"; assert run(SY + ["--out", str(f), "--rel-full", str(S / "rel_full6.jsonl"), "--expa-full", str(S / "expa_full6.jsonl")]).returncode == 0
r = O / "f_rel"; assert run(SY + ["--out", str(r), "--rel-full", str(S / "rel_full6.jsonl")]).returncode == 0
compare("synthetic full 6-lang single run (InternVL2.5 layout)", f, r)
# real 4-language cases
M = BASE / "model_results_20261116/minicpm/adapted"
for nm, model, relf, ea, extra in (("MiniCPM-V-4.5 (4-lang)", "minicpm_v45", M / "rel_full/results.jsonl", M / "expa_full/predictions.jsonl", ["--rel-old-summary", str(M / "rel_full/summary.json")]),
        ("Qwen3-VL-8B (4-lang)", "qwen3", BASE / "analysis_qwen3_20261108/raw/full_rel_results.jsonl", BASE / "expA_results_20261114/qwen3/expA_qwen3/full/predictions.jsonl", [])):
    f = O / (model + "_full"); assert run(["--model", model, "--mode", "4", "--out", str(f), "--rel-old", str(relf), "--expa-old", str(ea), *extra]).returncode == 0
    r = O / (model + "_rel"); assert run(["--model", model, "--mode", "4", "--out", str(r), "--rel-old", str(relf), *extra]).returncode == 0
    compare(nm, f, r)
# broken input must still be rejected in REL-only mode
rows = [json.loads(l) for l in open(S / "rel_new.jsonl")]
j = next(i for i, x in enumerate(rows) if x["primary_analysis_eligible"]); rows[j]["primary_analysis_eligible"] = False
bp = O / "broken_rel_new.jsonl"; bp.write_text("".join(json.dumps(x) + "\n" for x in rows))
b = O / "broken_out"; rr = run(SY + ["--out", str(b), "--rel-old", str(S / "rel_old.jsonl"), "--rel-new", str(bp), "--existing-json", str(ex / "results.json")])
ok = rr.returncode == 3 and not (b / "results.json").exists(); bp.unlink()
rep.append(dict(case="REL-only: broken eligible flag rejected", ok=ok, exit=rr.returncode)); print("PASS" if ok else "FAIL", "REL-only broken input rejected, exit", rr.returncode)
(HERE / "tests/rel_only_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
sys.exit(0 if all(x["ok"] for x in rep) else 1)
