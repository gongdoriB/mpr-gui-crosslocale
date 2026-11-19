"""Pre-opening checks (appendix 2 sec.4). They read ONLY counts, IDs and eligibility fields of result files; correctness
fields (generation_correct, scoring_correct, parsed labels, probabilities ...) are dropped at parse time and never
reach any check. Any failed check raises PrecheckFailed; the driver then stops before computing a single contrast.

Expected numbers are hard-coded from the frozen rule (appendix 2 sec.2/4/5) and ALSO recomputed from the input files
(conditions_6lang.jsonl, T_gemma_*.jsonl), so an input that disagrees with the rule is caught too.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("${PROJECT_ROOT}")
REL_COND = BASE / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl"
REL_COND_NEW = BASE / "followup_rel_6lang/data/rel_all_366/conditions_6lang_new_rows_only.jsonl"
EXPA_NEW = BASE / "expA_translation_6lang_20261115"
EXPA_OLD = BASE / "expA_translation_20261112"

LANGS4 = ("en", "zh", "th", "ru")
NEW_LANGS = ("fr", "ja")

# ---- hard-coded expectations (appendix 2) ----
SPEC6 = dict(
    rel_dependent_cross_quartets=9030, rel_dependent_eligible=8922, rel_dependent_eligible_old=3576, rel_dependent_eligible_new=5346,
    rel_dependent_excluded=108, rel_dependent_excluded_old=48, rel_dependent_excluded_new=60, rel_dependent_items=301,
    rel_condition_rows_total=52572, rel_condition_rows_old=23424, rel_condition_rows_new=29148,
    rel_independent_eligible=1792, rel_eligible_rows=42856,
    expa_complete_pairs=4740, expa_complete_old=1896, expa_complete_new=2844, expa_primary=4736, expa_primary_old=1895, expa_primary_new=2841,
    expa_sub2_primary=1747, expa_sub2_old=708, expa_sub2_new=1039, expa_collision_included=4739, expa_source_issue_main_pairs=10)
SPEC4 = dict(rel_condition_rows_total=23424, rel_eligible_rows=17168, rel_dependent_eligible=3576, rel_independent_eligible=716,
             rel_dependent_excluded=48, expa_complete_pairs=1896, expa_primary=1895, expa_sub2_primary=708)

REL_ID_FIELDS = ("input_id", "pair_id", "parallel_id", "condition", "status", "query_language", "gui_language", "text_dependency",
                 "primary_analysis", "primary_analysis_eligible", "eligible_without_source_adjudication", "source_issue")
EXPA_ID_FIELDS = ("input_id", "pair_id", "parallel_id", "split", "condition", "status", "query_language", "gui_language")


class PrecheckFailed(Exception):
    pass


def read_idfields(path, fields):
    """Parse a jsonl but keep only whitelisted non-outcome fields. source_issue is reduced to is-not-null."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = {k: r.get(k) for k in fields if k != "source_issue"}
            if "source_issue" in fields:
                d["source_issue"] = r.get("source_issue") is not None
            out.append(d)
    return out


def _jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def is_new_pair(q, g):
    return q in NEW_LANGS or g in NEW_LANGS


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail):
        self.items.append(dict(name=name, ok=bool(ok), detail=detail))

    def eq(self, name, got, want):
        self.add(name, got == want, f"got {got} expected {want}")

    def failed(self):
        return [c for c in self.items if not c["ok"]]


# ------------------------------------------------------------------ REL
def rel_input_expectations(langs):
    """From the INPUT conditions file (no results). Returns counts for the language set `langs`."""
    rows = [r for r in _jl(REL_COND) if r["query_language"] in langs and r["gui_language"] in langs]
    q = defaultdict(dict)
    for r in rows:
        q[r["pair_id"]][r["condition"]] = r
    e = Counter()
    items = set()
    for pid, v in q.items():
        o = v["O"]
        cross = o["query_language"] != o["gui_language"]
        new = is_new_pair(o["query_language"], o["gui_language"])
        e["rows"] += len(v)
        e["rows_new" if new else "rows_old"] += len(v)
        if not cross:
            continue
        dep = o["text_dependency"] == "dependent"
        el = bool(o["eligible_without_source_adjudication"])
        key = ("dep" if dep else "ind") + ("_el" if el else "_ex") + ("_new" if new else "_old")
        e[key] += 1
        if dep:
            e["dep_cross"] += 1
        if dep and el:
            items.add(o["parallel_id"])
        if el:
            e["eligible_rows"] += len(v)
    e["dep_items_eligible"] = len(items)
    return e


def check_rel(sources, mode, C=None):
    """sources: {tag: path}. tag in old/new (merged) or full (single run). Returns Checks."""
    C = C or Checks()
    langs = set(LANGS4) if mode == "4" else set(LANGS4) | set(NEW_LANGS)
    exp = rel_input_expectations(langs)
    spec = SPEC4 if mode == "4" else SPEC6
    # (1) input file agrees with the frozen numbers
    if mode == "6":
        C.eq("input: dependent cross quartets", exp["dep_cross"], spec["rel_dependent_cross_quartets"])
        C.eq("input: dependent eligible old", exp["dep_el_old"], spec["rel_dependent_eligible_old"])
        C.eq("input: dependent eligible new", exp["dep_el_new"], spec["rel_dependent_eligible_new"])
        C.eq("input: dependent excluded old", exp["dep_ex_old"], spec["rel_dependent_excluded_old"])
        C.eq("input: dependent excluded new", exp["dep_ex_new"], spec["rel_dependent_excluded_new"])
        C.eq("input: dependent eligible total", exp["dep_el_old"] + exp["dep_el_new"], spec["rel_dependent_eligible"])
        C.eq("input: dependent excluded total", exp["dep_ex_old"] + exp["dep_ex_new"], spec["rel_dependent_excluded"])
        C.eq("input: dependent items with >=1 eligible quartet", exp["dep_items_eligible"], spec["rel_dependent_items"])
        C.eq("input: condition rows total", exp["rows"], spec["rel_condition_rows_total"])
        C.eq("input: condition rows old", exp["rows_old"], spec["rel_condition_rows_old"])
        C.eq("input: condition rows new", exp["rows_new"], spec["rel_condition_rows_new"])
        C.eq("input: independent eligible quartets", exp["ind_el_old"] + exp["ind_el_new"], spec["rel_independent_eligible"])
        C.eq("input: eligible condition rows", exp["eligible_rows"], spec["rel_eligible_rows"])
        new_only = _jl(REL_COND_NEW)
        C.eq("input: conditions_6lang_new_rows_only rows", len(new_only), spec["rel_condition_rows_new"])
    else:
        C.eq("input: eligible condition rows (4-lang)", exp["eligible_rows"], spec["rel_eligible_rows"])
        C.eq("input: condition rows (4-lang)", exp["rows"], spec["rel_condition_rows_total"])
        C.eq("input: dependent eligible (4-lang)", exp["dep_el_old"], spec["rel_dependent_eligible"])
        C.eq("input: independent eligible (4-lang)", exp["ind_el_old"], spec["rel_independent_eligible"])
        C.eq("input: dependent excluded (4-lang)", exp["dep_ex_old"], spec["rel_dependent_excluded"])
    # (2) results: ids / eligibility only
    inp = {(r["pair_id"], r["condition"]): r for r in _jl(REL_COND) if r["query_language"] in langs and r["gui_language"] in langs}
    allrows = []
    for tag, path in sources.items():
        rows = read_idfields(path, REL_ID_FIELDS)
        allrows += [dict(r, _src=tag) for r in rows]
        C.add(f"results[{tag}]: input_id unique", len({r["input_id"] for r in rows}) == len(rows), f"{len(rows)} rows")
        C.add(f"results[{tag}]: all status == success", all(r["status"] == "success" for r in rows),
              dict(Counter(r["status"] for r in rows)))
        keyset = {(r["pair_id"], r["condition"]) for r in rows}
        C.add(f"results[{tag}]: (pair_id, condition) unique", len(keyset) == len(rows), f"{len(keyset)} unique of {len(rows)}")
        if tag == "old":
            want = {k for k, v in inp.items() if not is_new_pair(v["query_language"], v["gui_language"])}
        elif tag == "new":
            want = {k for k, v in inp.items() if is_new_pair(v["query_language"], v["gui_language"])}
        else:
            want = set(inp)
        C.add(f"results[{tag}]: (pair_id, condition) set equals input set", keyset == want,
              f"missing {len(want - keyset)}, unexpected {len(keyset - want)}")
        C.eq(f"results[{tag}]: row count", len(rows), len(want))
        if tag == "old":
            C.add("results[old]: contains no fr/ja language", not any(is_new_pair(r["query_language"], r["gui_language"]) for r in rows), "")
        if tag == "new":
            C.add("results[new]: every row has fr or ja", all(is_new_pair(r["query_language"], r["gui_language"]) for r in rows), "")
    if "old" in sources and "new" in sources:
        ids_old = {r["input_id"] for r in allrows if r["_src"] == "old"}
        ids_new = {r["input_id"] for r in allrows if r["_src"] == "new"}
        C.add("results: old and new share no input_id", not (ids_old & ids_new), f"overlap {len(ids_old & ids_new)}")
    # per-row agreement with the input
    bad_dep = bad_el = bad_prim = bad_lang = 0
    for r in allrows:
        i = inp.get((r["pair_id"], r["condition"]))
        if i is None:
            continue
        cross = r["query_language"] != r["gui_language"]
        bad_lang += (r["query_language"], r["gui_language"], r["parallel_id"]) != (i["query_language"], i["gui_language"], i["parallel_id"])
        bad_dep += r["text_dependency"] != i["text_dependency"]
        if cross:
            # primary_analysis_eligible must coincide with eligible_without_source_adjudication (appendix 2 sec.4)
            bad_el += not (r["primary_analysis_eligible"] == r["eligible_without_source_adjudication"] == bool(i["eligible_without_source_adjudication"]))
            bad_prim += not (r["primary_analysis"] is True and r["source_issue"] == (not r["primary_analysis_eligible"]))
        else:
            bad_prim += not (r["primary_analysis"] is False and r["primary_analysis_eligible"] is False)
    C.eq("results: language/parallel_id equal input", bad_lang, 0)
    C.eq("results: text_dependency equals input", bad_dep, 0)
    C.eq("results: primary_analysis_eligible == eligible_without_source_adjudication == input (cross-language rows)", bad_el, 0)
    C.eq("results: primary_analysis / source_issue consistent (cross: eligible == not source_issue; matched: false)", bad_prim, 0)
    # (3) counts from results
    q = defaultdict(dict)
    for r in allrows:
        q[r["pair_id"]][r["condition"]] = r
    C.add("results: every pair has exactly O,R,C,F", all(set(v) == {"O", "R", "C", "F"} for v in q.values()), f"{len(q)} pairs")
    cnt = Counter()
    items = set()
    for pid, v in q.items():
        o = v["O"]
        if o["query_language"] == o["gui_language"]:
            continue
        new = is_new_pair(o["query_language"], o["gui_language"])
        dep = o["text_dependency"] == "dependent"
        el = bool(o["primary_analysis_eligible"])
        cnt[("dep" if dep else "ind") + ("_el" if el else "_ex") + ("_new" if new else "_old")] += 1
        if dep:
            cnt["dep_cross"] += 1
        if dep and el:
            items.add(o["parallel_id"])
    elig_rows = sum(1 for r in allrows if r["primary_analysis_eligible"])
    if mode == "6":
        C.eq("results: dependent cross-language quartets", cnt["dep_cross"], spec["rel_dependent_cross_quartets"])
        C.eq("results: dependent eligible quartets", cnt["dep_el_old"] + cnt["dep_el_new"], spec["rel_dependent_eligible"])
        C.eq("results: dependent eligible old (4-lang part)", cnt["dep_el_old"], spec["rel_dependent_eligible_old"])
        C.eq("results: dependent eligible new (fr/ja part)", cnt["dep_el_new"], spec["rel_dependent_eligible_new"])
        C.eq("results: dependent excluded quartets", cnt["dep_ex_old"] + cnt["dep_ex_new"], spec["rel_dependent_excluded"])
        C.eq("results: dependent items with eligible quartet", len(items), spec["rel_dependent_items"])
        C.eq("results: independent eligible quartets", cnt["ind_el_old"] + cnt["ind_el_new"], spec["rel_independent_eligible"])
        C.eq("results: eligible condition rows", elig_rows, spec["rel_eligible_rows"])
        C.eq("results: total condition rows", len(allrows), spec["rel_condition_rows_total"])
    else:
        C.eq("results: dependent eligible quartets (4-lang)", cnt["dep_el_old"], spec["rel_dependent_eligible"])
        C.eq("results: dependent excluded quartets (4-lang)", cnt["dep_ex_old"], spec["rel_dependent_excluded"])
        C.eq("results: independent eligible quartets (4-lang)", cnt["ind_el_old"], spec["rel_independent_eligible"])
        C.eq("results: eligible condition rows (4-lang; was hard-coded 17,168)", elig_rows, spec["rel_eligible_rows"])
        C.eq("results: total condition rows (4-lang)", len(allrows), spec["rel_condition_rows_total"])
    return C


# ------------------------------------------------------------------ Exp A
def check_expa_inputs(A, C):
    """Input-only checks (6-language mode). A = configured analyze_expA_6lang module."""
    sets = A.pair_sets()
    nw, od = EXPA_NEW / "work", EXPA_OLD / "work"
    # SHA256SUMS of the 6-language inputs
    bad = []
    for line in (EXPA_NEW / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            if sha256_file(EXPA_NEW / rel.strip()) != h:
                bad.append(rel.strip())
    C.add("inputs: SHA256SUMS of expA_translation_6lang verified", not bad, f"mismatch {bad}")
    C.add("inputs: T_gemma_first_spans.jsonl byte-identical to T_gemma_spans.jsonl (first_spans is not in SHA256SUMS)",
          sha256_file(nw / "T_gemma_first_spans.jsonl") == sha256_file(nw / "T_gemma_spans.jsonl"), "")
    main_pairs = A.all_main_pairs()
    old_main = {p for p in main_pairs if not A.new_pair_flag(p)}
    new_main = main_pairs - old_main
    C.eq("inputs: main pairs old", len(old_main), SPEC6["expa_complete_old"])
    C.eq("inputs: main pairs new", len(new_main), SPEC6["expa_complete_new"])
    t_new = {r["pair_id"] for r in _jl(nw / "T_gemma_conditions.jsonl") if r["split"] == "main"}
    C.add("inputs: new T conditions main pairs == new span pairs", t_new == new_main, f"{len(t_new)} vs {len(new_main)}")
    prim = sets["primary_full"]
    C.eq("inputs: primary pairs", len(prim), SPEC6["expa_primary"])
    C.eq("inputs: primary old", len({p for p in prim if p in old_main}), SPEC6["expa_primary_old"])
    C.eq("inputs: primary new", len({p for p in prim if p in new_main}), SPEC6["expa_primary_new"])
    s2 = sets["sub2_meaning_preserved"]
    C.eq("inputs: sub2 in primary", len(s2), SPEC6["expa_sub2_primary"])
    C.eq("inputs: sub2 old", len(s2 & old_main), SPEC6["expa_sub2_old"])
    C.eq("inputs: sub2 new", len(s2 & new_main), SPEC6["expa_sub2_new"])
    C.eq("inputs: collision-included set", len(sets["sens_include_collision_pairs"]), SPEC6["expa_collision_included"])
    # collision pairs re-derived from inputs: O options all distinct but T options not (new main pairs only)
    def dist(r):
        v = list(r["options"].values()); return len(set(v)) == len(v)
    odist = {r["pair_id"]: dist(r) for r in _jl(nw / "OR_gemma_conditions.jsonl") if r["condition"] == "O"}
    derived = {r["pair_id"] for r in _jl(nw / "T_gemma_conditions.jsonl") if r["split"] == "main" and not dist(r) and odist.get(r["pair_id"])}
    C.add("inputs: collision pairs re-derived from inputs == the 3 pairs of appendix 2 sec.4", derived == set(A.COLLISION_PAIRS), f"derived {sorted(derived)}")
    C.add("inputs: excluded set in primary logic == forest_9 + 3 collision pairs",
          A.EXCLUDED == {A.FOREST9} | set(A.COLLISION_PAIRS) and not (A.EXCLUDED & prim), "")
    si = A.source_issue_pairs()
    C.eq("inputs: source-issue main pairs (calendar_5 fr)", len(si), SPEC6["expa_source_issue_main_pairs"])
    C.add("inputs: source-issue pairs are all calendar_5 with fr", all("calendar_5.jpg" in p and "fr" in p.split("::")[-2:] for p in si), sorted(si)[:2])
    return C


def check_expa_results(A, sources, mode, C):
    """sources: {tag: path}. Reads only id/status fields."""
    sets = A.pair_sets()
    prim, s2 = sets["primary_full"], sets["sub2_meaning_preserved"]
    allrows = []
    main_all = A.all_main_pairs()
    for tag, path in sources.items():
        rows = [r for r in read_idfields(path, EXPA_ID_FIELDS)]
        C.add(f"expA results[{tag}]: input_id unique", len({r["input_id"] for r in rows}) == len(rows), f"{len(rows)} rows")
        main = [r for r in rows if r["split"] == "main"]
        C.add(f"expA results[{tag}]: all main rows status == success", all(r["status"] == "success" for r in main), dict(Counter(r["status"] for r in main)))
        C.add(f"expA results[{tag}]: (pair_id, condition) unique among main", len({(r["pair_id"], r["condition"]) for r in main}) == len(main), "")
        pairs = {r["pair_id"] for r in main}
        if tag == "old":
            want = {p for p in main_all if not A.new_pair_flag(p)}
        elif tag == "new":
            want = {p for p in main_all if A.new_pair_flag(p)}
        else:
            want = set(main_all)
        C.add(f"expA results[{tag}]: main pair set equals input pair set", pairs == want, f"missing {len(want - pairs)}, unexpected {len(pairs - want)}")
        cs = defaultdict(set)
        for r in main:
            cs[r["pair_id"]].add(r["condition"])
        C.add(f"expA results[{tag}]: every main pair has O,T,R", all(v == {"O", "T", "R"} for v in cs.values()), "")
        allrows += [dict(r, _src=tag) for r in main]
    if "old" in sources and "new" in sources:
        C.add("expA results: old and new share no input_id", not ({r["input_id"] for r in allrows if r["_src"] == "old"} & {r["input_id"] for r in allrows if r["_src"] == "new"}), "")
    complete = A.complete_pairs(allrows)
    C.eq("expA: complete main pairs", len(complete), A.EXPECTED_COMPLETE)
    C.eq("expA: primary pairs (complete & not excluded)", len(prim & complete), A.EXPECTED_PRIMARY)
    if mode == "6":
        C.eq("expA: sub2 pairs in the complete primary set", len(s2 & complete), SPEC6["expa_sub2_primary"])
    else:
        C.eq("expA: sub2 pairs (4-lang)", len(s2 & complete), SPEC4["expa_sub2_primary"])
    return C


def finish(C, out_json=None):
    res = dict(passed=not C.failed(), n_checks=len(C.items), n_failed=len(C.failed()), checks=C.items)
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n")
    if C.failed():
        raise PrecheckFailed("; ".join(f"{c['name']}: {c['detail']}" for c in C.failed()[:6]) + f" ... ({len(C.failed())} failed)")
    return res
