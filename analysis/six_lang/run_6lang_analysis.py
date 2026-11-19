#!/usr/bin/env python3
"""6-language analysis driver (rules: 분석규칙_v1.md + 부록1 + 부록2 [FROZEN]). Local, CPU only, no network, no R.

Derived from analysis_newmodels_20261114/run_frozen_analysis.py. Statistical functions (rel_boot, holm, thai_indicator,
expa bootstrap via analyze_expA_6lang) are the original logic. Allowed changes only (appendix 2 sec.5): input paths,
language lists, merging old 4-language rows with new fr/ja rows, expected counts (1,896 -> 4,740), EXCLUDED set,
variant label all4 -> all6, hard-coded REL row checks.

Usage
  merged model (old 4-lang run + new fr/ja run), 6-language mode:
    run_6lang_analysis.py --model KEY --mode 6 --out DIR --rel-old R_OLD --rel-new R_NEW --expa-old E_OLD --expa-new E_NEW
        --existing-json OLD_RESULTS.json [--existing-key context_8b.Qwen3-VL-8B] --existing-expa-csv OLD_contrasts.csv
  one full 6-language run (InternVL2.5-8B):
    run_6lang_analysis.py --model KEY --mode 6 --out DIR --rel-full R --expa-full E
  4-language equivalence mode (reproduces the frozen 4-language driver):
    run_6lang_analysis.py --model KEY --mode 4 --out DIR --rel-old R --expa-old E [--rel-old-summary S]
Always runs the pre-opening checks first (precheck.json) and stops (exit 3) if any fails. --checks-only stops after them.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, importlib.util, json, shutil, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preopen_checks as PC  # noqa: E402

BASE = Path("${PROJECT_ROOT}")
PLAN = BASE / "paper_plan_20261112"
RULES = {"v1": PLAN / "분석규칙_v1.md", "app1": PLAN / "분석규칙_v1_부록1.md", "app2": PLAN / "분석규칙_v1_부록2_6개언어_초안.md"}
RULES_FROZEN = {"v1": PLAN / "분석규칙_v1.FROZEN.txt", "app1": PLAN / "분석규칙_v1_부록1.FROZEN.txt", "app2": PLAN / "분석규칙_v1_부록2.FROZEN.txt"}
B_REL, SEED_REL = 10_000, 42
REL_CON = ["R_minus_O", "C_minus_O", "F_minus_C", "F_minus_R"]
REF_DIRECTION = {"R_minus_O": ">0", "F_minus_C": ">0", "R_minus_T": ">0"}
RETRY_LABELS = ("T_retry", "T_gemma_retry")
TH_Q_LANGS = ("en", "zh", "th", "ru")   # appendix 2 sec.6: indicator (a) uses question languages en/zh/th/ru only


def sha(p):
    return PC.sha256_file(p)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def holm(p):
    p = np.array(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def variants_for(label):
    return {label: lambda q, g: True,
            "noTH_screen": lambda q, g: g != "th",
            "noTH_both": lambda q, g: q != "th" and g != "th"}


# ---------------------------------------------------------------- REL (identical logic to run_frozen_analysis.py)
def rel_boot(qrows, cols=REL_CON, B=B_REL, seed=SEED_REL):
    ids = sorted({r["parallel_id"] for r in qrows}); ix = {p: i for i, p in enumerate(ids)}; k = len(ids)
    n = np.zeros(k); sums = {c: np.zeros(k) for c in cols}
    for r in qrows:
        i = ix[r["parallel_id"]]; n[i] += 1
        for c in cols:
            sums[c][i] += r[c]
    rng = np.random.default_rng(seed); idx = rng.integers(0, k, size=(B, k))
    out = {}
    for c in cols:
        d = 100 * sums[c][idx].sum(1) / n[idx].sum(1); est = 100 * sums[c].sum() / n.sum()
        s = np.sort(d)
        le, ge = int((d <= 0).sum()), int((d >= 0).sum())
        p = min(1.0, 2 * min((1 + le) / (B + 1), (1 + ge) / (B + 1)))
        out[c] = dict(est_pp=est, lo=s[int(0.025 * (B - 1))], hi=s[int(0.975 * (B - 1))], p_raw=p, items=k, quartets=int(n.sum()))
    return out


def rel_analysis(quartets, variants):
    res = {}
    for vn, f in variants.items():
        res[vn] = {}
        sel = [q for q in quartets if f(q["query_language"], q["gui_language"])]
        for st, rows in (("dependent", [q for q in sel if q["text_dependency"] == "dependent"]),
                         ("independent", [q for q in sel if q["text_dependency"] == "independent"]),
                         ("all", sel)):
            if len({r["parallel_id"] for r in rows}) < 5:
                continue
            r = rel_boot(rows)
            if st == "dependent":
                h = holm([r[c]["p_raw"] for c in REL_CON])
                for c, a in zip(REL_CON, h):
                    r[c]["p_holm"] = float(a)
            res[vn][st] = r
    return res


def make_quartets(prep, model_key, raw, metric="generation_correct"):
    use = raw
    if metric == "scoring_correct":
        use = []
        for r in raw:
            r = dict(r); r["generation_correct"] = bool(r["scoring_correct"]); r["parse_success"] = True; use.append(r)
    proj = [prep.project_row(model_key, r) for r in use if r["primary_analysis_eligible"]]
    return prep.quartet_table(proj)


def pfail_table(rows, exp, cond_key="condition", gui_key="gui_language"):
    c = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (exp, r[cond_key], r[gui_key]); c[k][0] += 1; c[k][1] += (not r["parse_success"]); c[k][2] += (r["status"] != "success")
    return [dict(experiment=k[0], condition=k[1], screen=k[2], n=v[0], parse_fail=v[1], non_success=v[2]) for k, v in sorted(c.items())]


def agreement(rows):
    n = len(rows)
    same_corr = sum(bool(r["generation_correct"]) == bool(r["scoring_correct"]) for r in rows)
    same_lab = sum(r["parsed_generated_label"] == r["scored_predicted_label"] for r in rows)
    return dict(n=n, correctness_agree=same_corr / n, label_agree=same_lab / n, acc_generation=sum(bool(r["generation_correct"]) for r in rows) / n,
                acc_scoring=sum(bool(r["scoring_correct"]) for r in rows) / n)


def thai_indicator(rows):
    """(a) REL R-condition accuracy, Thai screen / English screen. Appendix 2 sec.6: only rows with question language en/zh/th/ru."""
    rows = [r for r in rows if r["query_language"] in TH_Q_LANGS]

    def acc(sel):
        s = [bool(r["generation_correct"]) for r in sel]; return sum(s) / len(s), len(s)
    out = {}
    for name, f in (("all_rows", lambda r: True), ("eligible_only", lambda r: r.get("primary_analysis_eligible"))):
        R = [r for r in rows if r["condition"] == "R" and f(r)]
        th, nth = acc([r for r in R if r["gui_language"] == "th"]); en, nen = acc([r for r in R if r["gui_language"] == "en"])
        out[name] = dict(acc_th=th, n_th=nth, acc_en=en, n_en=nen, ratio=th / en)
    return out


# ---------------------------------------------------------------- Exp A
def expa_analysis(A, rows, prim_pairs, variants, complete=None, pair_filter=None):
    """Same as the original expa_analysis but on given rows and primary pair set. Returns ({variant:{outcome:stats}}, n_complete)."""
    complete = complete if complete is not None else A.complete_pairs(rows)
    rows = [r for r in rows if r["pair_id"] in complete]
    prim = set(prim_pairs) & complete
    if pair_filter:
        prim = {p for p in prim if pair_filter(p)}
    res = {}
    for vn, f in variants.items():
        P = {p for p in prim if f(p.split("::")[-2], p.split("::")[-1])}
        res[vn] = {}
        for outcome in ("generation_correct", "scoring_correct"):
            items, m = A.item_matrix(rows, outcome, P)
            if len(items) < 10:
                res[vn][outcome] = dict(pairs=len(P), items=len(items), note="fewer than 10 items; not computed")
                continue
            b = A.bootstrap(m)
            res[vn][outcome] = dict(pairs=len(P), items=len(items), acc=dict(zip(A.CONDS, (100 * m.mean(0)).tolist())), **b)
    return res, len(complete)


def verdict(contrast, est, p_holm):
    ref = REF_DIRECTION.get(contrast)
    if ref is None:
        return "양측 보고만"
    if est > 0 and p_holm <= 0.05:
        return "재현"
    if est > 0:
        return "부호 일치 (유의 아님)"
    return "부호 반대" if est < 0 else "0"


def sidecar(path):
    """gate / contract json next to a results file (provenance only)."""
    d = Path(path).parent
    out = {}
    for n in ("gate_rel_complete.json", "gate_expa_complete.json", "run_contract.json", "summary.json"):
        if (d / n).exists():
            out[n] = sha(d / n)
    rc = d / "run_contract.json"
    if rc.exists():
        c = json.loads(rc.read_text())
        out["contract"] = dict(model_config=c.get("model_config"), batch_size=c.get("batch_size"), dataset_sha256=c.get("dataset_sha256"),
                               seed=c.get("seed"))
    return out


def prov_rows(rows, exp, server_by_tag, writer):
    for r in rows:
        hw = (r.get("hardware") or {}).get("gpus") or [{}]
        writer.writerow([exp, r["input_id"], r["_src"], r.get("experiment_fingerprint"), server_by_tag.get(r["_src"]) or hw[0].get("name"),
                         r.get("batch_size"), r.get("code_commit")])


def pick(d, path):
    for k in path.split(".") if path else []:
        d = d[k]
    return d


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--mode", choices=["4", "6"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rel-old"); ap.add_argument("--rel-new"); ap.add_argument("--rel-full")
    ap.add_argument("--rel-old-summary"); ap.add_argument("--rel-new-summary"); ap.add_argument("--rel-full-summary")
    ap.add_argument("--expa-old"); ap.add_argument("--expa-new"); ap.add_argument("--expa-full")
    ap.add_argument("--expa-retry", action="append", default=[], help="T_retry predictions (sensitivity); condition label T_retry or T_gemma_retry")
    ap.add_argument("--existing-json"); ap.add_argument("--existing-key", default=""); ap.add_argument("--existing-expa-csv")
    ap.add_argument("--server-old", default=None); ap.add_argument("--server-new", default=None)
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    mode = a.mode
    HAS_EXPA = bool(a.expa_full or a.expa_old)   # REL-only mode: no Exp A file given -> every Exp A output is skipped
    assert HAS_EXPA or not (a.expa_retry or a.existing_expa_csv), "--expa-retry / --existing-expa-csv need Exp A"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    VA = "all4" if mode == "4" else "all6"
    VARS = variants_for(VA)

    # ---- layout
    full = bool(a.rel_full)
    if full:
        assert not (a.rel_old or a.rel_new or a.expa_old or a.expa_new), "use either *-full or *-old/*-new"
        rel_src = {"full": a.rel_full}; expa_src = {"full": a.expa_full} if a.expa_full else {}
    else:
        assert a.rel_old
        rel_src = {"old": a.rel_old}; expa_src = {"old": a.expa_old} if a.expa_old else {}
        if mode == "6":
            assert a.rel_new, "6-language merged mode needs --rel-new"
            rel_src["new"] = a.rel_new
            assert bool(a.expa_old) == bool(a.expa_new), "Exp A needs both --expa-old and --expa-new (or neither: REL-only mode)"
            if a.expa_new:
                expa_src["new"] = a.expa_new
        else:
            assert not (a.rel_new or a.expa_new or full), "4-language mode takes only old results"
    summaries = {"old": a.rel_old_summary, "new": a.rel_new_summary, "full": a.rel_full_summary}
    server = {"old": a.server_old, "new": a.server_new, "full": a.server_old}

    # ---- 0. frozen rule files
    R = {"model_key": a.model, "mode": mode, "variant_label": VA, "rel_only": not HAS_EXPA,
         "expa_status": "computed" if HAS_EXPA else "SKIPPED: REL-only mode (no Exp A predictions given); no Exp A output in this file", "layout": "full_single_run" if full else "merged_old_new" if mode == "6" else "old_4lang_only"}
    prov = {}
    for k in RULES:
        h = sha(RULES[k]); prov[f"rules_{k}_sha256"] = h
        prov[f"rules_{k}_frozen_ok"] = h in RULES_FROZEN[k].read_text()
        assert prov[f"rules_{k}_frozen_ok"], f"rules {k} sha mismatch -> stop"
    prep = load_mod(HERE / "prepare_glmm_6lang.py", "prep_glmm_6lang")
    A = load_mod(HERE / "analyze_expA_6lang.py", "analyze_expA_6lang")
    A.configure(mode)
    if mode == "4":
        prep.LANGUAGES = {"en", "zh", "th", "ru"}
    prov.update(driver_sha256=sha(Path(__file__).resolve()), prepare_glmm_6lang_sha256=sha(HERE / "prepare_glmm_6lang.py"),
                analyze_expA_6lang_sha256=sha(HERE / "analyze_expA_6lang.py"), preopen_checks_sha256=sha(HERE / "preopen_checks.py"),
                inputs={f"rel_{t}": sha(p) for t, p in rel_src.items()} | {f"expa_{t}": sha(p) for t, p in expa_src.items()},
                sidecars={f"rel_{t}": sidecar(p) for t, p in rel_src.items()} | {f"expa_{t}": sidecar(p) for t, p in expa_src.items()})
    R["provenance"] = prov

    # ---- 1. pre-opening checks (ids / counts / eligibility only). Stops here on failure.
    C = PC.Checks()
    PC.check_rel(rel_src, mode, C)
    if HAS_EXPA:
        if mode == "6":
            PC.check_expa_inputs(A, C)
        PC.check_expa_results(A, expa_src, mode, C)
    try:
        R["precheck"] = PC.finish(C, out / "precheck.json")
    except PC.PrecheckFailed as e:
        print("PRECHECK FAILED -> stop:", e, file=sys.stderr)
        sys.exit(3)
    print(f"precheck ok: {len(C.items)} checks")
    if a.checks_only:
        return

    # ---- 2. REL
    raw = []
    for t, p in rel_src.items():
        for r in jl(p):
            r["_src"] = t; raw.append(r)
    assert all(r["status"] == "success" for r in raw)
    for t, sp in summaries.items():     # optional per-run summary.json cross-check (the original validate_summary)
        if sp and t in rel_src:
            part = [r for r in raw if r["_src"] == t]
            prep.validate_summary(make_quartets(prep, a.model, part), {a.model: json.loads(Path(sp).read_text())})
            R.setdefault("validated_summaries", []).append(t)
    quartets = make_quartets(prep, a.model, raw)
    qs = make_quartets(prep, a.model, raw, "scoring_correct")
    is_new = lambda q: PC.is_new_pair(q["query_language"], q["gui_language"])
    R["rel_counts"] = dict(condition_rows=len(raw), eligible_quartets=len(quartets),
                           eligible_quartets_old=sum(not is_new(q) for q in quartets), eligible_quartets_new=sum(is_new(q) for q in quartets),
                           by_source=dict(Counter(r["_src"] for r in raw)))
    R["rel"] = {"generation_correct": rel_analysis(quartets, VARS), "scoring_correct": rel_analysis(qs, VARS)}
    R["rel_glmm"] = "skipped: Rscript not available (GLMM Wald p is a sensitivity analysis; no substitute computed)"
    four = {}
    if mode == "6":
        R["rel_new_rows_only_exploratory"] = {"generation_correct": rel_analysis([q for q in quartets if is_new(q)], VARS),
                                              "note": "rows with fr or ja in question or screen language; exploratory (appendix 2 sec.2/7)"}
        if full:   # InternVL2.5-8B: 4-language values recomputed from the new run's 4-language rows (appendix 2 sec.2)
            VARS4 = variants_for("all4")
            q4 = [q for q in quartets if not is_new(q)]
            four["rel_generation_correct"] = rel_analysis(q4, VARS4)
            four["rel_scoring_correct"] = rel_analysis([q for q in qs if not is_new(q)], VARS4)
            four["source"] = "recomputed from the 4-language rows of the new run (not copied)"
            four["thai_indicator_a"] = thai_indicator([r for r in raw if not is_new_pair_row(r)])
        elif a.existing_json:   # merged models: copy existing frozen 4-language outputs, never recompute
            ex = json.loads(Path(a.existing_json).read_text())
            sub = pick(ex, a.existing_key)
            relx = sub["rel"]
            four["rel_generation_correct"] = relx["generation_correct"] if "generation_correct" in relx else relx
            four["rel_scoring_correct"] = relx.get("scoring_correct")
            if HAS_EXPA:
                four["expa"] = sub["expa"]
            four["thai_indicator_a"] = sub["thai_indicator_a"]
            four["source"] = dict(copied_from=str(a.existing_json), sha256=sha(a.existing_json), key=a.existing_key or "(top level)",
                                  note="copied, NOT recomputed (appendix 2 sec.1)")
            if a.existing_expa_csv:
                dst = out / "four_lang_copied_expA_contrasts.csv"
                shutil.copyfile(a.existing_expa_csv, dst)
                four["expa_contrasts_csv"] = dict(copied_from=str(a.existing_expa_csv), sha256=sha(a.existing_expa_csv), copy=str(dst.name))
            # consistency of counts only (no outcomes): the old rows of the merged data must have the same quartet counts as the copied values
            cp = four["rel_generation_correct"]["all4"]["dependent"]["R_minus_O"]
            mine = rel_boot([q for q in quartets if not is_new(q) and q["text_dependency"] == "dependent"], cols=["R_minus_O"], B=100)["R_minus_O"]
            R["four_lang_count_consistency"] = dict(copied_quartets=cp["quartets"], merged_old_quartets=mine["quartets"], ok=cp["quartets"] == mine["quartets"])
            assert R["four_lang_count_consistency"]["ok"], "old rows of merged data do not match the copied 4-language quartet count -> stop"
        else:
            raise SystemExit("6-language merged mode needs --existing-json (4-language values are copied, not recomputed)")
    # Thai indicator (a): question languages en/zh/th/ru only
    R["thai_indicator_a"] = thai_indicator(raw)
    ti = R["thai_indicator_a"]["eligible_only"]["ratio"]
    R["thai_flag_a"] = dict(ratio_headline=ti, flag_weak_thai_reading=ti < 0.7, ratio_all_rows=R["thai_indicator_a"]["all_rows"]["ratio"],
                            flag_if_all_rows=R["thai_indicator_a"]["all_rows"]["ratio"] < 0.7, indicator_b="6x6 not part of this driver",
                            question_languages_used=list(TH_Q_LANGS))

    # ---- 3. Exp A
    rows = []
    if HAS_EXPA:
        rows = A.load_predictions_many(list(expa_src.values()))
        tags = {}
        for t, p in expa_src.items():
            for r in A.load_predictions(Path(p)):
                tags[r["input_id"]] = t
        for r in rows:
            r["_src"] = tags[r["input_id"]]
        sets = A.pair_sets()
        complete = A.complete_pairs(rows)
        R["expa_complete_pairs"] = len(complete)
        R["expa"], _ = expa_analysis(A, rows, sets["primary_full"], VARS, complete)
        ea_dir = out / "expA_analyze"
        summ = A.run_sets(a.model, rows, ";".join(expa_src.values()), ea_dir, sets, complete)
        ps = summ["sets"]["primary_full|generation_correct"]
        chk = R["expa"][VA]["generation_correct"]
        assert abs(chk["T_minus_O"]["pp"] - ps["T_minus_O"]["pp"]) < 1e-9 and abs(chk["R_minus_T"]["p_holm_primary2"] - ps["R_minus_T"]["p_holm_primary2"]) < 1e-12
        R["expa_sets_summary"] = {k: dict(pairs=v["pairs"], items=v["items"]) for k, v in summ["sets"].items()}
        if mode == "6":
            newf = lambda p: A.new_pair_flag(p)
            R["expa_new_rows_only_exploratory"], _ = expa_analysis(A, rows, sets["primary_full"], VARS, complete, pair_filter=newf)
            R["expa_sensitivity"] = {}
            for nm, key in (("collision_pairs_included", "sens_include_collision_pairs"), ("calendar5_fr_source_issue_pairs_excluded", "sens_drop_source_issue_pairs")):
                R["expa_sensitivity"][nm], _ = expa_analysis(A, rows, sets[key], VARS, complete)
            if a.expa_retry:
                rr = A.load_predictions_many(a.expa_retry)
                assert all(r["condition"] in RETRY_LABELS for r in rr), f"retry rows must have condition in {RETRY_LABELS}"
                ret = []
                for r in rr:
                    r = dict(r); r["condition"] = "T"; ret.append(r)
                rows_r = [r for r in rows if r["condition"] != "T"] + ret
                ret_pairs = {r["pair_id"] for r in ret}
                comp_r = A.complete_pairs(rows_r)
                res_r, _ = expa_analysis(A, rows_r, sets["primary_full"], VARS, comp_r)
                R["expa_sensitivity"]["T_retry"] = dict(results=res_r, retry_pairs=len(ret_pairs), primary_pairs_with_retry=len(ret_pairs & sets["primary_full"]),
                                                        complete_pairs=len(comp_r),
                                                        note="T condition replaced by T_retry predictions where present; label of retry rows in prediction files is unverified (확인 필요)")
            else:
                R["expa_sensitivity"]["T_retry"] = "not computed: no --expa-retry predictions given"
            if full:   # 4-language recompute from the new run (InternVL2.5-8B)
                A.configure("4")
                rows4 = [r for r in rows if not A.new_pair_flag(r["pair_id"])]
                sets4 = A.pair_sets(); comp4 = A.complete_pairs(rows4)
                assert len(comp4) == 1896, f"4-language part has {len(comp4)} complete pairs"
                four["expa"], _ = expa_analysis(A, rows4, sets4["primary_full"], variants_for("all4"), comp4)
                A.run_sets(a.model, rows4, "4-lang rows of the new run", out / "expA_analyze_four_lang_recomputed", sets4, comp4)
                A.configure("6")
    if four:
        R["four_lang"] = four

    # ---- 4. parse failures, agreement, tokens (also split by run source: gate / 2% rule applies to the new fr/ja run separately)
    expa_rows = rows
    SETS = (("REL", raw), ("ExpA", expa_rows)) if HAS_EXPA else (("REL", raw),)
    pf = pfail_table(raw, "REL") + (pfail_table(expa_rows, "ExpA") if HAS_EXPA else [])
    R["parse_failures"] = pf
    totals = {}
    for e, rs in SETS:
        for t in sorted({r["_src"] for r in rs}) + ["ALL"]:
            sel = [r for r in rs if t == "ALL" or r["_src"] == t]
            n = len(sel); f = sum(not r["parse_success"] for r in sel)
            totals[f"{e}/{t}"] = dict(n=n, parse_fail=f, rate=f / n, over_2pct=f / n > 0.02)
    R["parse_failure_totals"] = totals
    R["agreement"] = {"REL(condition rows)": agreement(raw), **({"ExpA": agreement(expa_rows)} if HAS_EXPA else {})}
    tok = {}
    for e, rs in SETS:
        v = [r["visual_token_count"] for r in rs if r.get("visual_token_count") is not None]
        p = [r["prompt_token_count"] for r in rs if r.get("prompt_token_count") is not None]
        tok[e] = dict(visual_token_count_available=bool(v), visual=(dict(min=min(v), median=statistics.median(v), max=max(v)) if v else None),
                      prompt=(dict(min=min(p), median=statistics.median(p), max=max(p)) if p else None),
                      batch_sizes=dict(Counter(r["batch_size"] for r in rs)),
                      batch_sizes_by_source={t: dict(Counter(r["batch_size"] for r in rs if r["_src"] == t)) for t in sorted({r["_src"] for r in rs})})
    R["tokens"] = tok

    # ---- 5. verdicts and 4 vs 6 comparison
    V = []
    for c in REL_CON:
        e = R["rel"]["generation_correct"][VA]["dependent"][c]
        V.append(dict(exp="REL", contrast=c, est_pp=e["est_pp"], p_holm=e["p_holm"], verdict=verdict(c, e["est_pp"], e["p_holm"])))
    for c in (("T_minus_O", "R_minus_T") if HAS_EXPA else ()):
        e = R["expa"][VA]["generation_correct"][c]
        V.append(dict(exp="ExpA", contrast=c, est_pp=e["pp"], p_holm=e["p_holm_primary2"], verdict=verdict(c, e["pp"], e["p_holm_primary2"])))
    R["replication"] = V
    R["thai_exclusion_changes"] = thai_changes(R, VA, HAS_EXPA)
    sc = []
    for c in REL_CON:
        x, y = R["rel"]["generation_correct"][VA]["dependent"][c], R["rel"]["scoring_correct"][VA]["dependent"][c]
        sc.append(dict(exp="REL", contrast=c, sign_same=(x["est_pp"] > 0) == (y["est_pp"] > 0), holm_same=(x["p_holm"] <= .05) == (y["p_holm"] <= .05)))
    for c in (("T_minus_O", "R_minus_T") if HAS_EXPA else ()):
        x, y = R["expa"][VA]["generation_correct"][c], R["expa"][VA]["scoring_correct"][c]
        sc.append(dict(exp="ExpA", contrast=c, sign_same=(x["pp"] > 0) == (y["pp"] > 0), holm_same=(x["p_holm_primary2"] <= .05) == (y["p_holm_primary2"] <= .05)))
    R["scoring_vs_generation_decisions"] = sc
    if mode == "6" and four:
        R["holm_decision_four_vs_six"] = four_vs_six(R, four)
        R["new_rows_only_sign_agreement"] = sign_agreement(R)

    # ---- 6. per-row provenance
    with open(out / "provenance_rel_rows.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["experiment", "input_id", "source", "experiment_fingerprint", "server_gpu", "batch_size", "code_commit"])
        prov_rows(raw, "REL", server, w)
    R["provenance_rows_files"] = ["provenance_rel_rows.csv"]
    if HAS_EXPA:
        with open(out / "provenance_expa_rows.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["experiment", "input_id", "source", "experiment_fingerprint", "server_gpu", "batch_size", "code_commit"])
            prov_rows(expa_rows, "ExpA", server, w)
        R["provenance_rows_files"].append("provenance_expa_rows.csv")

    (out / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float) + "\n")
    write_tables(R, out, VA)
    print("done:", out)


def is_new_pair_row(r):
    return PC.is_new_pair(r["query_language"], r["gui_language"])


def thai_changes(R, VA, has_expa=True):
    chg = []
    for vn in ("noTH_screen", "noTH_both"):
        for c in REL_CON:
            x, y = R["rel"]["generation_correct"][VA]["dependent"][c], R["rel"]["generation_correct"][vn]["dependent"][c]
            chg.append(dict(exp="REL", variant=vn, contrast=c, sign_same=(x["est_pp"] > 0) == (y["est_pp"] > 0), sig_all=x["p_holm"] <= .05, sig_var=y["p_holm"] <= .05))
        for c in (("T_minus_O", "R_minus_T") if has_expa else ()):
            x, y = R["expa"][VA]["generation_correct"][c], R["expa"][vn]["generation_correct"][c]
            chg.append(dict(exp="ExpA", variant=vn, contrast=c, sign_same=(x["pp"] > 0) == (y["pp"] > 0), sig_all=x["p_holm_primary2"] <= .05, sig_var=y["p_holm_primary2"] <= .05))
    return chg


def four_vs_six(R, four):
    out = []
    g4 = four["rel_generation_correct"]["all4"]["dependent"]; g6 = R["rel"]["generation_correct"]["all6"]["dependent"]
    for c in REL_CON:
        out.append(dict(exp="REL", contrast=c, est4=g4[c]["est_pp"], holm4=g4[c]["p_holm"], est6=g6[c]["est_pp"], holm6=g6[c]["p_holm"],
                        sign_same=bool((g4[c]["est_pp"] > 0) == (g6[c]["est_pp"] > 0)), holm_decision_differs=bool((g4[c]["p_holm"] <= .05) != (g6[c]["p_holm"] <= .05))))
    e4 = four["expa"]["all4"]["generation_correct"] if "expa" in R else None; e6 = R["expa"]["all6"]["generation_correct"] if "expa" in R else None
    for c in (("T_minus_O", "R_minus_T") if "expa" in R else ()):
        out.append(dict(exp="ExpA", contrast=c, est4=e4[c]["pp"], holm4=e4[c]["p_holm_primary2"], est6=e6[c]["pp"], holm6=e6[c]["p_holm_primary2"],
                        sign_same=bool((e4[c]["pp"] > 0) == (e6[c]["pp"] > 0)), holm_decision_differs=bool((e4[c]["p_holm_primary2"] <= .05) != (e6[c]["p_holm_primary2"] <= .05))))
    return dict(rows=out, any_decision_differs=any(r["holm_decision_differs"] for r in out),
                note="if any_decision_differs: record internally and tell the user (appendix 2 sec.1)")


def sign_agreement(R):
    out = []
    n = R["rel_new_rows_only_exploratory"]["generation_correct"]["all6"]["dependent"]; p = R["rel"]["generation_correct"]["all6"]["dependent"]
    for c in REL_CON:
        out.append(dict(exp="REL", contrast=c, est_new_only=n[c]["est_pp"], est_primary=p[c]["est_pp"], same_sign=bool((n[c]["est_pp"] > 0) == (p[c]["est_pp"] > 0))))
    n = R["expa_new_rows_only_exploratory"]["all6"]["generation_correct"] if "expa" in R else None; p = R["expa"]["all6"]["generation_correct"] if "expa" in R else None
    for c in (("T_minus_O", "R_minus_T") if "expa" in R else ()):
        out.append(dict(exp="ExpA", contrast=c, est_new_only=n[c]["pp"], est_primary=p[c]["pp"], same_sign=bool((n[c]["pp"] > 0) == (p[c]["pp"] > 0))))
    return out


def fmt_ci(e, d=1):
    return f"{e['est_pp']:+.{d}f} [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}]"


def write_tables(R, out, VA):
    L = []
    g = R["rel"]["generation_correct"]; e = R.get("expa", {})
    L.append(f"# {R['model_key']} (mode {R['mode']}, layout {R['layout']})\n")
    L.append("## REL 의존 문항 층, 생성 답 정오 (문항 bootstrap 10,000회, seed 42, Holm 4개)\n")
    L.append("| 언어 범위 | 문항/quartet | 대비 | 차이 %p [95% CI] | 원시 p | Holm p |\n|---|---|---|---|---|---|")
    for vn in g:
        for c in REL_CON:
            s = g[vn]["dependent"][c]
            L.append(f"| {vn} | {s['items']}/{s['quartets']} | {c} | {fmt_ci(s)} | {s['p_raw']:.4f} | {s['p_holm']:.4f} |")
    if not e:
        L.append("\n## 실험 A: 건너뜀 (REL-only 모드, Exp A 예측 없음)\n")
    else:
      L.append("\n## 실험 A 주 분석 (생성 답 정오, seed 20261113, Holm 2개)\n")
    if e:
      L.append("| 언어 범위 | 쌍/문항 | O | T | R | T-O [CI] | R-T [CI] | R-O [CI] | Holm p (T-O / R-T) |\n|---|---|---|---|---|---|---|---|---|")
    for vn in e:
        s = e[vn]["generation_correct"]
        f = lambda k: f"{s[k]['pp']:+.1f} [{s[k]['lo']:+.1f}, {s[k]['hi']:+.1f}]"
        L.append(f"| {vn} | {s['pairs']}/{s['items']} | {s['acc']['O']:.1f} | {s['acc']['T']:.1f} | {s['acc']['R']:.1f} | {f('T_minus_O')} | {f('R_minus_T')} | {f('R_minus_O')} | {s['T_minus_O']['p_holm_primary2']:.4f} / {s['R_minus_T']['p_holm_primary2']:.4f} |")
    if "holm_decision_four_vs_six" in R:
        L.append("\n## 4개 언어 vs 6개 언어 (Holm 판정이 갈리는 대비 표시)\n")
        L.append("| 실험 | 대비 | 4언어 추정/Holm | 6언어 추정/Holm | 부호 같음 | 판정 갈림 |\n|---|---|---|---|---|---|")
        for r in R["holm_decision_four_vs_six"]["rows"]:
            L.append(f"| {r['exp']} | {r['contrast']} | {r['est4']:+.2f} / {r['holm4']:.4f} | {r['est6']:+.2f} / {r['holm6']:.4f} | {r['sign_same']} | {r['holm_decision_differs']} |")
    L.append("\n## 태국어 읽기 지표 (a) (질문 언어 en/zh/th/ru만)\n")
    L.append("| 정의 | 태국어 | 영어 | 비율 |\n|---|---|---|---|")
    for df_, v in R["thai_indicator_a"].items():
        L.append(f"| {df_} | {100*v['acc_th']:.1f}% (n={v['n_th']}) | {100*v['acc_en']:.1f}% (n={v['n_en']}) | {v['ratio']:.3f} |")
    L.append(f"\n표시(<0.7, eligible_only): {R['thai_flag_a']['flag_weak_thai_reading']}")
    L.append("\n## 파싱 실패 합계 (실행 묶음별; 2% 규칙은 새 fr/ja 묶음에 따로 적용)\n")
    L.append("; ".join(f"{k} {v['parse_fail']}/{v['n']} ({100*v['rate']:.2f}%, 2% 초과={v['over_2pct']})" for k, v in R["parse_failure_totals"].items()))
    L.append("\n## 재현 판정 (모델 안 Holm, 8B 기준 방향 대비)\n")
    L.append("| 실험 | 대비 | 추정 %p | Holm p | 판정 |\n|---|---|---|---|---|")
    for v in R["replication"]:
        L.append(f"| {v['exp']} | {v['contrast']} | {v['est_pp']:+.2f} | {v['p_holm']:.4f} | {v['verdict']} |")
    (out / "tables_KO.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
