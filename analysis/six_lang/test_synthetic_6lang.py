#!/usr/bin/env python3
"""Tests on SYNTHETIC 6-language data (synthetic/ made by make_synthetic.py; no real result file is read).
 P*: positive tests (all code paths run, numbers consistent between layouts)
 B*: deliberately broken inputs; each must be rejected by the pre-opening checks (exit code 3, results.json not written)
Writes tests/synthetic_6lang_report.json. Run: python3 -I test_synthetic_6lang.py
"""
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "synthetic"
O = HERE / "tests/synth"
DRV = str(HERE / "run_6lang_analysis.py")
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
rep = []


def run(args, expect=0):
    r = subprocess.run([sys.executable, "-I", DRV, *args], capture_output=True, text=True)
    return r


def record(name, ok, detail=""):
    rep.append(dict(test=name, ok=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + ("  | " + str(detail)[:230] if detail else ""))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def deep_eq(a, b, path="", diffs=None):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in a:
            if k in b:
                deep_eq(a[k], b[k], f"{path}/{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_eq(x, y, f"{path}[{i}]", diffs)
    elif a != b and not (a != a and b != b):
        diffs.append(f"{path}: {a!r} != {b!r}")


MERGED = ["--model", "synth_merged", "--mode", "6", "--rel-old", str(S / "rel_old.jsonl"), "--rel-new", str(S / "rel_new.jsonl"),
          "--expa-old", str(S / "expa_old.jsonl"), "--expa-new", str(S / "expa_new.jsonl")]


def merged_args(out, **over):
    a = list(MERGED)
    for k, v in over.items():
        i = a.index(k); a[i + 1] = str(v)
    return a + ["--out", str(out)]


# ---------------------------------------------------------------- positive tests (outputs from the earlier runs of this folder)
ex = json.loads((O / "existing_old/results.json").read_text())
mg = json.loads((O / "merged6/results.json").read_text())
record("P1 merged 6-language run: pre-opening checks passed", mg["precheck"]["passed"], f"{mg['precheck']['n_checks']} checks")
record("P1 REL rows 52,572; eligible quartets 10,714 (old 4,292 + new 6,422)", mg["rel_counts"]["condition_rows"] == 52572 and mg["rel_counts"]["eligible_quartets"] == 10714 and
       mg["rel_counts"]["eligible_quartets_old"] == 4292 and mg["rel_counts"]["eligible_quartets_new"] == 6422, mg["rel_counts"])
d = mg["rel"]["generation_correct"]["all6"]["dependent"]["R_minus_O"]
record("P1 REL dependent: 8,922 quartets, 301 items", d["quartets"] == 8922 and d["items"] == 301, (d["quartets"], d["items"]))
record("P1 Exp A: 4,740 complete pairs, primary 4,736, sub2 1,747, collision-included 4,739, source-issue-dropped 4,726",
       mg["expa_complete_pairs"] == 4740 and mg["expa"]["all6"]["generation_correct"]["pairs"] == 4736 and
       mg["expa_sets_summary"]["sub2_meaning_preserved|generation_correct"]["pairs"] == 1747 and
       mg["expa_sensitivity"]["collision_pairs_included"]["all6"]["generation_correct"]["pairs"] == 4739 and
       mg["expa_sensitivity"]["calendar5_fr_source_issue_pairs_excluded"]["all6"]["generation_correct"]["pairs"] == 4726,
       (mg["expa_complete_pairs"], mg["expa"]["all6"]["generation_correct"]["pairs"]))
record("P1 variants all6 / noTH_both / noTH_screen present in REL and Exp A", all(v in mg["rel"]["generation_correct"] and v in mg["expa"] for v in ("all6", "noTH_both", "noTH_screen")))
record("P1 T_retry sensitivity computed", isinstance(mg["expa_sensitivity"]["T_retry"], dict) and mg["expa_sensitivity"]["T_retry"]["primary_pairs_with_retry"] == 4736,
       {k: v for k, v in mg["expa_sensitivity"]["T_retry"].items() if k != "results" and k != "note"})
record("P1 new-rows-only (REL, Exp A) and sign-agreement table present", "rel_new_rows_only_exploratory" in mg and "expa_new_rows_only_exploratory" in mg and len(mg["new_rows_only_sign_agreement"]) == 6)
pr = [l.split(",") for l in (O / "merged6/provenance_rel_rows.csv").read_text().splitlines()[1:]]
record("P1 per-row provenance: 52,572 REL rows with fingerprint/server/batch, both sources", len(pr) == 52572 and {r[2] for r in pr} == {"old", "new"} and
       {r[4] for r in pr} == {"SYNTH-B200", "SYNTH-4090"} and all(r[3] and r[5] for r in pr))
dd = []; deep_eq(mg["four_lang"]["rel_generation_correct"], ex["rel"]["generation_correct"], "", dd)
dd2 = []; deep_eq(mg["four_lang"]["expa"], ex["expa"], "", dd2)
record("P2 merged model: 4-language values are the copied existing values (REL, Exp A) exactly", not dd and not dd2, (dd + dd2)[:3])
record("P2 copied Exp A contrasts.csv byte-identical to the existing one",
       (O / "merged6/four_lang_copied_expA_contrasts.csv").read_bytes() == (O / "existing_old/expA_analyze/contrasts.csv").read_bytes())
dd = []; deep_eq(mg["thai_indicator_a"], ex["thai_indicator_a"], "", dd)
record("P3 Thai indicator (a): value of the 6-language merged data equals the old-only value (question languages en/zh/th/ru only)", not dd, dd[:2])

# full 6-language single-run layout (InternVL2.5-8B style); model label equal so contrasts.csv is comparable byte for byte
out = O / "full6"
shutil.rmtree(out, ignore_errors=True)
r = run(["--model", "synth_merged", "--mode", "6", "--out", str(out), "--rel-full", str(S / "rel_full6.jsonl"), "--expa-full", str(S / "expa_full6.jsonl")])
record("P4 full 6-language layout runs", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
fl = json.loads((out / "results.json").read_text())
dd = []; deep_eq(fl["rel"], mg["rel"], "", dd)
record("P4 full layout == merged layout for REL primary (same rows)", not dd, dd[:2])
dd = []; deep_eq(fl["expa"], mg["expa"], "", dd)
record("P4 full layout == merged layout for Exp A primary (same rows)", not dd, dd[:2])
dd = []; deep_eq(fl["four_lang"]["rel_generation_correct"], ex["rel"]["generation_correct"], "", dd)
deep_eq(fl["four_lang"]["rel_scoring_correct"], ex["rel"]["scoring_correct"], "", dd)
record("P5 full layout: 4-language REL values recomputed from the 4-lang rows == independent 4-language run (generation + scoring)", not dd, dd[:2])
dd = []; deep_eq(fl["four_lang"]["expa"], ex["expa"], "", dd)
record("P5 full layout: 4-language Exp A recomputed == independent 4-language run", not dd, dd[:2])
dd = []; deep_eq(fl["four_lang"]["thai_indicator_a"], ex["thai_indicator_a"], "", dd)
record("P5 full layout: 4-language Thai indicator recomputed == 4-language run", not dd, dd[:2])
record("P5 full layout: recomputed 4-language Exp A contrasts.csv byte-identical to the independent 4-language run (all sets incl. sub2=708)",
       (out / "expA_analyze_four_lang_recomputed/contrasts.csv").read_bytes() == (O / "existing_old/expA_analyze/contrasts.csv").read_bytes())

# outcome-blind check: result files WITHOUT any outcome field must pass the pre-opening checks
B = O / "broken"; shutil.rmtree(B, ignore_errors=True); B.mkdir(parents=True)
STRIP = ("generation_correct", "correct", "scoring_correct", "parse_success", "parsed_generated_label", "scored_predicted_label", "generated_text")
for nm in ("rel_old", "rel_new", "expa_old", "expa_new"):
    wjl(B / f"{nm}_nooutcome.jsonl", [{k: v for k, v in x.items() if k not in STRIP} for x in jl(S / f"{nm}.jsonl")])
r = run(merged_args(B / "o_nooutcome", **{"--rel-old": B / "rel_old_nooutcome.jsonl", "--rel-new": B / "rel_new_nooutcome.jsonl",
                                          "--expa-old": B / "expa_old_nooutcome.jsonl", "--expa-new": B / "expa_new_nooutcome.jsonl"}) + ["--checks-only"])
record("P6 pre-opening checks pass on files that contain NO outcome fields (checks never read correctness)", r.returncode == 0, r.stderr[-200:])
for nm in ("rel_old", "rel_new", "expa_old", "expa_new"):
    (B / f"{nm}_nooutcome.jsonl").unlink()

# ---------------------------------------------------------------- broken inputs
relnew, relold, eanew, eaold = jl(S / "rel_new.jsonl"), jl(S / "rel_old.jsonl"), jl(S / "expa_new.jsonl"), jl(S / "expa_old.jsonl")


def broken(name, expect_sub, files, extra_args=(), exit_code=3):
    """files: {arg: rows}. Must exit with exit_code and write no results.json."""
    over = {}
    for arg, rows in files.items():
        p = B / f"{name}_{arg.strip('-')}.jsonl"; wjl(p, rows); over[arg] = p
    out = B / f"o_{name}"
    r = run(merged_args(out, **over) + list(extra_args))
    pc = out / "precheck.json"
    failed = [c["name"] for c in json.loads(pc.read_text())["checks"] if not c["ok"]] if pc.exists() else []
    ok = r.returncode == exit_code and not (out / "results.json").exists() and (not expect_sub or any(expect_sub in n for n in failed) or expect_sub in r.stderr)
    record(f"{name}: rejected (exit {r.returncode}, no results.json)", ok, (failed[:3] or r.stderr[-200:]))
    for p in over.values():
        Path(p).unlink()


def first_idx(rows, f):
    return next(i for i, x in enumerate(rows) if f(x))


i = first_idx(relnew, lambda x: x["condition"] == "F")
broken("B01_rel_row_missing", "row count", {"--rel-new": relnew[:i] + relnew[i + 1:]})
j = first_idx(relnew, lambda x: x["primary_analysis_eligible"] and x["text_dependency"] == "dependent")
x = [dict(r) for r in relnew]; x[j]["primary_analysis_eligible"] = False
broken("B02_rel_eligible_flag_flipped", "primary_analysis_eligible ==", {"--rel-new": x})
x = [dict(r) for r in relnew]; x[5]["status"] = "error"
broken("B03_rel_status_not_success", "status == success", {"--rel-new": x})
broken("B04_rel_duplicate_row", "unique", {"--rel-new": relnew + [relnew[7]]})
k = first_idx(relnew, lambda x: x["condition"] == "O")
broken("B05_rel_old_contains_fr_row", "contains no fr/ja", {"--rel-old": relold + [dict(relnew[k], input_id="rel_orcf::dup")]})
x = [dict(r) for r in relnew]; x[j]["text_dependency"] = "independent"
broken("B06_rel_text_dependency_changed", "text_dependency", {"--rel-new": x})
x = [dict(r) for r in relnew]
for r in x:  # a source-issue pair counted as eligible everywhere (eligible_without_source_adjudication and primary_analysis_eligible both true)
    if r["query_language"] != r["gui_language"] and not r["eligible_without_source_adjudication"]:
        r["eligible_without_source_adjudication"] = True; r["primary_analysis_eligible"] = True; r["source_issue"] = None
broken("B07_rel_exclusions_dropped", "dependent excluded", {"--rel-new": x})
broken("B08_rel_new_missing_entirely_wrong_scope", "", {"--rel-new": relold[:1000]})
tt = first_idx(eanew, lambda x: x["condition"] == "T" and x["split"] == "main")
broken("B09_expa_T_row_missing", "complete main pairs", {"--expa-new": eanew[:tt] + eanew[tt + 1:]})
x = [dict(r) for r in eanew]; x[first_idx(x, lambda r: r["split"] == "main")]["status"] = "error"
broken("B10_expa_status_not_success", "status == success", {"--expa-new": x})
x = [dict(r) for r in eanew]
mm = first_idx(x, lambda r: r["split"] == "main")
for r in x[mm:mm + 3]:
    r["pair_id"] = "rel::9/bogus.png::en::fr"
broken("B11_expa_unknown_pair", "pair set", {"--expa-new": x})
broken("B12_expa_old_has_new_pair_rows", "main pair set", {"--expa-old": eaold + [dict(eanew[mm], input_id="rel_expA::dupe")]}, exit_code=3)

# existing 4-language values inconsistent with the old rows (only quartet counts are compared) -> stop (assertion, exit 1)
bad = json.loads((O / "existing_old/results.json").read_text())
bad["rel"]["generation_correct"]["all4"]["dependent"]["R_minus_O"]["quartets"] = 3500
(B / "existing_bad.json").write_text(json.dumps(bad))
out = B / "o_B13"
r = run(merged_args(out) + ["--existing-json", str(B / "existing_bad.json")])
record("B13 copied 4-language values inconsistent with old rows: stopped (no results.json)", r.returncode != 0 and not (out / "results.json").exists(), r.stderr[-160:].strip())
r = run(merged_args(B / "o_B14") + [])  # merged mode without --existing-json
record("B14 merged 6-language mode without existing 4-language values: refuses", r.returncode != 0 and not (B / "o_B14/results.json").exists(), r.stderr[-120:].strip())

# in-process: wrong EXCLUDED set and wrong expected counts are caught by the Exp A checks
spec = importlib.util.spec_from_file_location("A", HERE / "analyze_expA_6lang.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import preopen_checks as PC
A.configure("6"); A.EXCLUDED = {A.FOREST9}
C = PC.Checks(); PC.check_expa_inputs(A, C)
record("B15 EXCLUDED = forest_9 only (collision pairs not excluded) is caught by the input checks", any(not c["ok"] for c in C.items), [c["name"] for c in C.failed()][:3])
A.configure("6"); A.EXPECTED_COMPLETE = 1896
C = PC.Checks(); PC.check_expa_results(A, {"old": S / "expa_old.jsonl", "new": S / "expa_new.jsonl"}, "6", C)
record("B16 wrong expected complete-pair count (1,896) is caught", any(not c["ok"] for c in C.items), [c["name"] for c in C.failed()][:3])

rep_path = HERE / "tests/synthetic_6lang_report.json"
rep_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
n_fail = sum(not r["ok"] for r in rep)
print(f"\n{len(rep)} tests, {n_fail} failed")
sys.exit(1 if n_fail else 0)
