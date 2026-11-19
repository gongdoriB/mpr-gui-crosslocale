"""6-language copy of expA_translation_20261112/analysis/analyze_expA.py (rules: 분석규칙_v1_부록2 sec.5).

Only the changes allowed by appendix 2 sec.5 were made (input paths, language lists, merging old 4-language and new
fr/ja files, expected complete-pair count, EXCLUDED, variant label). Statistics (load_predictions, item_matrix,
contrasts, bootstrap) are byte-for-byte the original logic. MODE "4" reproduces the original behaviour exactly
(old inputs, EXCLUDED = forest_9 zh->ru, expected 1,896); MODE "6" is the appendix-2 configuration.

Usage (CLI, same as the original): python analyze_expA_6lang.py <model> <predictions.jsonl[.gz]>... <out_dir> [--mode 4|6]
The first-class entry point for the driver is configure() + pair_sets() + run_sets().
Unit of resampling: item (parallel_id). Item score = mean over its remaining pairs.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

BASE = Path("${PROJECT_ROOT}")
OLD_WORK = BASE / "expA_translation_20261112/work"
NEW_WORK = BASE / "expA_translation_6lang_20261115/work"
CONDS = ("O", "T", "R")
SEED, B = 20261113, 10_000
SIM_MIN = 0.7
LANGS4 = ("en", "zh", "th", "ru")
LANGS6 = ("en", "zh", "th", "ru", "fr", "ja")
NEW_LANGS = ("fr", "ja")
FOREST9 = "rel::5/forest_9.jpg::zh::ru"  # T options not distinct (pre-registered, 4-language rule)
# appendix 2 sec.4 (ㄱ): main pairs whose O options were all distinct but whose T options collide (user decision 2026-11-16)
COLLISION_PAIRS = frozenset({"rel::2/calendar_6.png::fr::zh", "rel::3/clock_9.jpg::th::ja", "rel::6/phone_5.jpg::fr::ru"})

# ---- configuration (set by configure()) ----
MODE = "6"
LANGS = LANGS6
EXCLUDED = {FOREST9} | set(COLLISION_PAIRS)
EXPECTED_COMPLETE = 4740
EXPECTED_PRIMARY = 4736
VARIANT_ALL = "all6"


def configure(mode: str):
    global MODE, LANGS, EXCLUDED, EXPECTED_COMPLETE, EXPECTED_PRIMARY, VARIANT_ALL
    assert mode in ("4", "6")
    MODE = mode
    if mode == "4":
        LANGS, EXCLUDED, EXPECTED_COMPLETE, EXPECTED_PRIMARY, VARIANT_ALL = LANGS4, {FOREST9}, 1896, 1895, "all4"
    else:
        LANGS, EXCLUDED, EXPECTED_COMPLETE, EXPECTED_PRIMARY, VARIANT_ALL = LANGS6, {FOREST9} | set(COLLISION_PAIRS), 4740, 4736, "all6"


def _jl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_predictions(path: Path) -> list[dict]:
    path = Path(path)
    text = gzip.open(path, "rt", encoding="utf-8").read() if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["input_id"]] = r
    return [r for r in rows.values() if r["split"] == "main"]


def load_predictions_many(paths) -> list[dict]:
    """Merge several prediction files (old 4-language rows + new fr/ja rows). input_id must not repeat across files."""
    out, seen = [], {}
    for p in paths:
        for r in load_predictions(Path(p)):
            assert r["input_id"] not in seen, f"input_id {r['input_id']} in both {seen[r['input_id']]} and {p}"
            seen[r["input_id"]] = str(p)
            out.append(r)
    return out


def span_sources():
    """(spans path, similarity path, tag) in merge order. Old files are the 4-language T_gemma_first_* (as in the original)."""
    src = [(OLD_WORK / "T_gemma_first_spans.jsonl", OLD_WORK / "T_gemma_first_similarity.jsonl", "old")]
    if MODE == "6":
        src.append((NEW_WORK / "T_gemma_first_spans.jsonl", NEW_WORK / "T_gemma_similarity.jsonl", "new"))
    return src


def new_pair_flag(pair_id: str) -> bool:
    q, g = pair_id.split("::")[-2:]
    return q in NEW_LANGS or g in NEW_LANGS


def source_issue_pairs() -> set:
    """Main pairs whose T conditions carry eligible_without_source_adjudication == false (6-language files only)."""
    if MODE != "6":
        return set()
    return {r["pair_id"] for r in _jl(NEW_WORK / "T_gemma_conditions.jsonl")
            if r["split"] == "main" and not r["eligible_without_source_adjudication"]}


def pair_sets() -> dict[str, set]:
    spans, sim = [], {}
    for sp, sm, tag in span_sources():
        spans += _jl(sp)
        sim.update({(r["pair_id"], r["field"], r["ref_id"]): r for r in _jl(sm)})
    # mapping audit exists for the old 4-language pairs only (appendix 2 sec.5): audited_* and annotation_artifact sets apply to them only
    aff = json.loads((OLD_WORK / "mapping_audit_affected_pairs.json").read_text())
    artifact_refs = {(r["parallel_id"], r["ref_id"]) for r in csv.DictReader((OLD_WORK / "mapping_audit_review.csv").open(encoding="utf-8-sig"))
                     if r["verdict"] == "annotation_artifact"}
    by = defaultdict(list)
    for s in spans:
        if s["split"] == "main":
            by[s["pair_id"]].append(s)
    allp = set(by) - EXCLUDED
    frac = {p: sum(s["match"] != "different" for s in by[p]) / len(by[p]) for p in allp}
    screen_names = {p: {s["screen_text"] for s in by[p]} for p in allp}
    S = {
        "primary_full": allp,
        "audited_drop_mapping_error": allp - set(aff["mapping_error"]),
        "audited_drop_error_uncertain": allp - set(aff["mapping_error"]) - set(aff["uncertain"]),
        "sub1_zero_screen_match": {p for p in allp if frac[p] == 0},
        "sub2_meaning_preserved": {p for p in allp if any(s["match"] == "different" for s in by[p]) and
                                   all(sim[(s["pair_id"], s["field"], s["ref_id"])]["sim_T_screen"] >= SIM_MIN
                                       for s in by[p] if s["match"] == "different")},
        "sub3_stem_name_different": {p for p in allp if any(s["match"] == "different" and s["field"] == "question_stem" for s in by[p])},
        "sub4_frac_0": {p for p in allp if frac[p] == 0},
        "sub4_frac_0_to_half": {p for p in allp if 0 < frac[p] < 0.5},
        "sub4_frac_half_to_1": {p for p in allp if 0.5 <= frac[p] < 1},
        "sub4_frac_1": {p for p in allp if frac[p] == 1},
        # new pairs have no mapping audit: they are always kept (a (parallel_id, ref_id) key shared with an old pair must not drop them)
        "sens_drop_annotation_artifact": {p for p in allp if (MODE == "6" and new_pair_flag(p)) or
                                          not any((s["parallel_id"], s["ref_id"]) in artifact_refs for s in by[p])},
        "sens_drop_digit_mismatch": {p for p in allp if all(re.findall(r"\d+", s["source_text"]) == re.findall(r"\d+", s["screen_text"]) for s in by[p])},
        "sens_drop_untranslated": {p for p in allp if not any(s["untranslated"] for s in by[p])},
        "sens_drop_translation_equals_other_screen_name": {p for p in allp if not any(
            s["translation"] in screen_names[p] and s["translation"] != s["screen_text"] for s in by[p])},
    }
    for g in LANGS:
        S[f"sub5_gui_{g}"] = {p for p in allp if by[p][0]["gui_language"] == g}
    if MODE == "6":  # appendix 2 sec.4 sensitivity sets (6-language only)
        S["sens_include_collision_pairs"] = (set(by) - {FOREST9})
        S["sens_drop_source_issue_pairs"] = allp - source_issue_pairs()
    return S


def all_main_pairs() -> set:
    """All main pair_ids present in the span inputs (before EXCLUDED)."""
    out = set()
    for sp, _, _ in span_sources():
        out |= {s["pair_id"] for s in _jl(sp) if s["split"] == "main"}
    return out


def item_matrix(rows, outcome, pairs):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["pair_id"] in pairs:
            acc[r["parallel_id"]][r["condition"]].append(float(bool(r[outcome])))
    items = sorted(acc)
    return items, np.array([[np.mean(acc[i][c]) for c in CONDS] for i in items])


def contrasts(m):
    o, t, r = m[..., 0], m[..., 1], m[..., 2]
    return {"T_minus_O": t - o, "R_minus_T": r - t, "R_minus_O": r - o, "TO_minus_RT": (t - o) - (r - t)}


def bootstrap(m):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(m), size=(B, len(m)))
    point = {k: v.mean() for k, v in contrasts(m).items()}
    draws = {k: v.mean(axis=1) for k, v in contrasts(m[idx]).items()}
    out = {}
    for k, d in draws.items():
        d = np.round(d, 12)
        out[k] = {"pp": 100 * point[k], "lo": 100 * np.percentile(d, 2.5), "hi": 100 * np.percentile(d, 97.5),
                  "p_boot": min(1.0, 2 * min((1 + (d <= 0).sum()) / (B + 1), (1 + (d >= 0).sum()) / (B + 1)))}
    order = sorted(("T_minus_O", "R_minus_T"), key=lambda k: out[k]["p_boot"])
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (2 - rank) * out[k]["p_boot"]))
        out[k]["p_holm_primary2"] = run
    out["corr_TO_RT_boot"] = float(np.corrcoef(draws["T_minus_O"], draws["R_minus_T"])[0, 1])
    ro = draws["R_minus_O"]
    if out["R_minus_O"]["lo"] > 0:
        ratio = draws["T_minus_O"] / ro
        out["recovery_ratio_TO_over_RO"] = {"est": point["T_minus_O"] / point["R_minus_O"],
                                            "lo": float(np.percentile(ratio, 2.5)), "hi": float(np.percentile(ratio, 97.5))}
    return out


def complete_pairs(rows):
    ok = defaultdict(set)
    for r in rows:
        if r["status"] == "success":
            ok[r["pair_id"]].add(r["condition"])
    return {p for p, c in ok.items() if c == set(CONDS)}


def run_sets(model, rows, pred_label, out: Path, sets=None, complete=None):
    """The original main() body: writes contrasts.csv / glmm_rows.csv / summary.json. Returns summary."""
    out.mkdir(parents=True, exist_ok=True)
    complete = complete if complete is not None else complete_pairs(rows)
    if len(complete) != EXPECTED_COMPLETE:
        print(f"WARNING: {len(complete)} complete main pairs (expected {EXPECTED_COMPLETE})", file=sys.stderr)
    rows = [r for r in rows if r["pair_id"] in complete]
    sets = sets if sets is not None else pair_sets()
    summary = {"model": model, "predictions": pred_label, "seed": SEED, "B": B, "sets": {}}
    with (out / "contrasts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "set", "outcome", "pairs", "items", "acc_O", "acc_T", "acc_R", "contrast", "pp", "ci_lo", "ci_hi",
                    "p_boot", "p_holm_primary2"])
        for name, P in sets.items():
            P = P & complete
            for outcome in (("generation_correct", "scoring_correct") if name == "primary_full" else ("generation_correct",)):
                items, m = item_matrix(rows, outcome, P)
                if len(items) < 10:
                    continue
                res = bootstrap(m)
                acc = 100 * m.mean(axis=0)
                summary["sets"][f"{name}|{outcome}"] = {"pairs": len(P), "items": len(items), "acc": dict(zip(CONDS, acc.tolist())), **res}
                for k in ("T_minus_O", "R_minus_T", "R_minus_O", "TO_minus_RT"):
                    v = res[k]
                    w.writerow([model, name, outcome, len(P), len(items), *[round(a, 2) for a in acc], k, round(v["pp"], 3),
                                round(v["lo"], 3), round(v["hi"], 3), round(v["p_boot"], 5),
                                round(v["p_holm_primary2"], 5) if "p_holm_primary2" in v else ""])
    with (out / "glmm_rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["correct", "condition", "item", "pair", "query_language", "gui_language"])
        for r in rows:
            if r["pair_id"] in sets["primary_full"]:
                w.writerow([int(bool(r["generation_correct"])), r["condition"], r["parallel_id"], r["pair_id"],
                            r["query_language"], r["gui_language"]])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = "6"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        args = [a for a in args if a != mode]
    configure(mode)
    model, out, preds = args[0], Path(args[-1]), [Path(a) for a in args[1:-1]]
    rows = load_predictions_many(preds)
    summary = run_sets(model, rows, ";".join(map(str, preds)), out)
    p = summary["sets"]["primary_full|generation_correct"]
    print(json.dumps({k: p[k] for k in ("pairs", "items", "acc", "T_minus_O", "R_minus_T", "R_minus_O", "corr_TO_RT_boot")}, indent=1, default=float))


if __name__ == "__main__":
    main()
