#!/usr/bin/env python3
"""Equivalence test: run the 6-language driver in 4-language mode on EXISTING 4-language results and compare, number by
number (exact equality, no tolerance), with the frozen outputs of the old driver / analyze_expA.py.
Models: MiniCPM-V-4.5 (reference analysis_newmodels_20261114/minicpm_v45/results.json) and Qwen3-VL-8B
(reference: context_8b block of analysis_newmodels_20261114/uitars_15_7b/results.json + expA_results_20261114/analysis_qwen3).
Read-only on all inputs. Writes tests/equivalence_4lang_report.json.
"""
import hashlib, json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path("${PROJECT_ROOT}")
OUT = HERE / "tests" / "equivalence_4lang"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def deep_eq(a, b, path="", diffs=None, counter=None):
    """exact equality; NaN == NaN. counter['n'] counts compared scalar leaves."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
        for k in a:
            if k in b:
                deep_eq(a[k], b[k], f"{path}/{k}", diffs, counter)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_eq(x, y, f"{path}[{i}]", diffs, counter)
    else:
        counter["n"] += 1
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")


def run(cmd):
    r = subprocess.run([sys.executable, "-I", *cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {r.stderr[-800:]}")
    return r


def compare(name, new, ref_rel, ref_expa, ref_thai, extra, ref_csv_dir, new_dir):
    rep = {"model": name, "compared": {}}
    counter = {"n": 0}
    alld = []

    def one(label, a, b):
        d = []; c = {"n": 0}
        deep_eq(a, b, "", d, c)
        rep["compared"][label] = dict(leaves=c["n"], diffs=len(d), first_diffs=d[:5])
        counter["n"] += c["n"]; alld.extend(d)

    one("rel (generation+scoring variants)", new["rel"], ref_rel)
    one("expa variants", new["expa"], ref_expa)
    one("thai_indicator_a", new["thai_indicator_a"], ref_thai)
    for k, (a, b) in extra.items():
        one(k, a, b)
    for fn in ("contrasts.csv", "glmm_rows.csv"):
        same = (new_dir / "expA_analyze" / fn).read_bytes() == (ref_csv_dir / fn).read_bytes()
        rep["compared"][f"expA_analyze/{fn} byte-identical"] = same
        if not same:
            alld.append(fn)
    s_new = json.loads((new_dir / "expA_analyze/summary.json").read_text())["sets"]
    s_ref = json.loads((ref_csv_dir / "summary.json").read_text())["sets"]
    one("expA_analyze/summary.json sets (all analysis sets)", s_new, s_ref)
    rep["total_scalar_leaves_compared"] = counter["n"]
    rep["n_differences"] = len(alld)
    rep["identical"] = len(alld) == 0
    return rep


reports = []
# ---------------- MiniCPM-V-4.5
M = BASE / "model_results_20261116/minicpm/adapted"
ref = json.loads((BASE / "analysis_newmodels_20261114/minicpm_v45/results.json").read_text())
out = OUT / "minicpm_v45"
args = ["--model", "minicpm_v45", "--mode", "4", "--out", str(out), "--rel-old", str(M / "rel_full/results.jsonl"), "--rel-old-summary", str(M / "rel_full/summary.json"),
        "--expa-old", str(M / "expa_full/predictions.jsonl")]
run([str(HERE / "run_6lang_analysis.py"), *args] if not out.exists() else [str(HERE / "run_6lang_analysis.py"), *args])
new = json.loads((out / "results.json").read_text())
rel_inputs_match = (sha(M / "rel_full/results.jsonl") == ref["provenance"]["inputs"]["rel_full/results.jsonl"] and
                    sha(M / "expa_full/predictions.jsonl") == ref["provenance"]["inputs"]["expa_full/predictions.jsonl"])
extra = {
    "expa_complete_pairs": (new["expa_complete_pairs"], ref["expa_complete_pairs"]),
    "parse_failures": (new["parse_failures"], ref["parse_failures"]),
    "parse_failure_totals REL/ALL vs REL": (new["parse_failure_totals"]["REL/ALL"], ref["parse_failure_totals"]["REL"]),
    "parse_failure_totals ExpA/ALL vs ExpA": (new["parse_failure_totals"]["ExpA/ALL"], ref["parse_failure_totals"]["ExpA"]),
    "agreement": (new["agreement"], {k: ref["agreement"][k] for k in new["agreement"]}),
    "tokens (visual/prompt/batch_sizes)": ({e: {k: v[k] for k in ("visual_token_count_available", "visual", "prompt", "batch_sizes")} for e, v in new["tokens"].items()},
                                           {e: {k: v[k] for k in ("visual_token_count_available", "visual", "prompt", "batch_sizes")} for e, v in ref["tokens"].items()}),
    "thai_flag_a": (new["thai_flag_a"], {k: ref["thai_flag_a"][k] for k in ("ratio_headline", "flag_weak_thai_reading", "ratio_all_rows", "flag_if_all_rows")} | {"indicator_b": "6x6 not part of this driver", "question_languages_used": ["en", "zh", "th", "ru"]}),
    "replication verdicts": (new["replication"], ref["replication"]),
    "thai_exclusion_changes": (new["thai_exclusion_changes"], ref["thai_exclusion_changes"]),
    "scoring_vs_generation_decisions": (new["scoring_vs_generation_decisions"], ref["scoring_vs_generation_decisions"]),
    "rel_counts (rows, eligible quartets)": ({"condition_rows": new["rel_counts"]["condition_rows"], "eligible_quartets": new["rel_counts"]["eligible_quartets"]},
                                             {k: ref["rel_counts"][k] for k in ("condition_rows", "eligible_quartets")}),
}
rep = compare("MiniCPM-V-4.5", new, ref["rel"], ref["expa"], ref["thai_indicator_a"], extra, BASE / "analysis_newmodels_20261114/minicpm_v45/expA_analyze", out)
rep["input_sha_equals_reference_provenance"] = rel_inputs_match
rep["precheck"] = dict(n_checks=new["precheck"]["n_checks"], n_failed=new["precheck"]["n_failed"])
reports.append(rep)

# ---------------- Qwen3-VL-8B
ctx = json.loads((BASE / "analysis_newmodels_20261114/uitars_15_7b/results.json").read_text())["context_8b"]["Qwen3-VL-8B"]
out = OUT / "qwen3_vl_8b"
run([str(HERE / "run_6lang_analysis.py"), "--model", "qwen3", "--mode", "4", "--out", str(out),
     "--rel-old", str(BASE / "analysis_qwen3_20261108/raw/full_rel_results.jsonl"),
     "--expa-old", str(BASE / "expA_results_20261114/qwen3/expA_qwen3/full/predictions.jsonl")])
new = json.loads((out / "results.json").read_text())
rep = compare("Qwen3-VL-8B", new, ctx["rel"], ctx["expa"], ctx["thai_indicator_a"], {}, BASE / "expA_results_20261114/analysis_qwen3", out)
# ctx["rel"] holds only generation variants; compare against our generation block explicitly
d = []; c = {"n": 0}
deep_eq(new["rel"]["generation_correct"], ctx["rel"], "", d, c)
rep["compared"]["rel generation_correct vs context_8b rel"] = dict(leaves=c["n"], diffs=len(d), first_diffs=d[:5])
# the generic "rel" comparison above compared {generation,scoring} vs ctx rel -> keys differ by construction; replace by the explicit one
rep["compared"].pop("rel (generation+scoring variants)", None)
rep["n_differences"] = sum(v["diffs"] if isinstance(v, dict) else (0 if v else 1) for v in rep["compared"].values())
rep["identical"] = rep["n_differences"] == 0
rep["precheck"] = dict(n_checks=new["precheck"]["n_checks"], n_failed=new["precheck"]["n_failed"])
reports.append(rep)

(HERE / "tests").mkdir(exist_ok=True)
(HERE / "tests/equivalence_4lang_report.json").write_text(json.dumps(reports, indent=1, ensure_ascii=False) + "\n")
for r in reports:
    print(r["model"], "IDENTICAL" if r["identical"] else "DIFFERENT", "differences:", r["n_differences"])
    for k, v in r["compared"].items():
        print("  ", k, v if not isinstance(v, dict) else f"leaves={v['leaves']} diffs={v['diffs']}")
sys.exit(0 if all(r["identical"] for r in reports) else 1)
