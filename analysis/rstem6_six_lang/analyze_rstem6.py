"""6-language R_stem/R_opts analysis (rules: rstem_ropts_qwen3_20261109/ANALYSIS_PLAN_KO.md
+ model_expansion_20261118/rstem6_vast4090c/ANALYSIS_PLAN_ADDENDUM_6LANG_KO.md).
Bootstrap/contrast code copied unchanged from rstem_ropts_qwen3_20261109/analysis/analyze_rstem.py.

Usage: python -I analyze_rstem6.py <predictions.jsonl> <rstem6_conditions.jsonl> <out_dir> [<ref_4lang_predictions.jsonl>]
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CONDS = ("O", "R_stem", "R_opts", "R")
SEED, B = 20261109, 10_000
LANG4 = {"en", "zh", "th", "ru"}


def contrasts(m):
    o, s, p, r = (m[:, k] for k in range(4))
    return {"R_minus_O": r - o, "Rstem_minus_O": s - o, "Ropts_minus_O": p - o,
            "stem_effect_S": ((s - o) + (r - p)) / 2, "option_effect_P": ((p - o) + (r - s)) / 2,
            "interaction_I": r - s - p + o,
            "S_minus_P": ((s - o) + (r - p)) / 2 - ((p - o) + (r - s)) / 2}


def item_matrix(rows, outcome, subset=None):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] != "success" or (subset and not subset(r)):
            continue
        acc[r["parallel_id"]][r["condition"]].append(float(bool(r[outcome])))
    items = sorted(acc)
    for i in items:
        if set(acc[i]) != set(CONDS):
            raise ValueError(f"Item {i} lacks a condition")
    return items, np.array([[np.mean(acc[i][c]) for c in CONDS] for i in items])


def bootstrap(m):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(m), size=(B, len(m)))
    point = {k: v.mean() for k, v in contrasts(m).items()}
    draws = {k: np.empty(B) for k in point}
    for b in range(B):
        for k, v in contrasts(m[idx[b]]).items():
            draws[k][b] = v.mean()
    out = {}
    for k, d in draws.items():
        d = np.round(d, 12)
        p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[k] = {"pp": 100 * point[k], "lo": 100 * np.percentile(d, 2.5),
                  "hi": 100 * np.percentile(d, 97.5), "p_boot": min(1.0, p_two)}
    primary = ["stem_effect_S", "option_effect_P", "interaction_I"]
    order = sorted(primary, key=lambda k: out[k]["p_boot"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * out[k]["p_boot"]))
        out[k]["p_holm"] = running
    return out


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); rows[r["input_id"]] = r
    return [r for r in rows.values() if r.get("split", "main") == "main"]


def main():
    pred, cond, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ref = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    rows = load(pred)
    # identical-question pairs (addendum): two conditions of a pair share the same question text
    q = defaultdict(dict)
    for line in cond.read_text(encoding="utf-8").splitlines():
        if line.strip():
            x = json.loads(line); q[x["pair_id"]][x["condition"]] = x["question"]
    ident = {p for p, d in q.items() if len(set(d.values())) < len(d)}
    ok = defaultdict(set)
    for r in rows:
        if r["status"] == "success":
            ok[r["pair_id"]].add(r["condition"])
    complete = {p for p, c in ok.items() if c == set(CONDS)}
    n_pairs_all = len({r["pair_id"] for r in rows})
    rows = [r for r in rows if r["pair_id"] in complete]
    checks = {"rows": len(rows), "pairs": len(complete), "dropped_incomplete_pairs": n_pairs_all - len(complete),
              "identical_question_pairs": len(ident),
              "identical_question_items": len({p.rsplit("::", 2)[0] for p in ident}),
              "unparsed": sum(not r.get("parse_success", True) for r in rows)}
    # determinism: in identical pairs, same-question conditions give same answer
    lab = {(r["pair_id"], r["condition"]): r["parsed_generated_label"] for r in rows}
    same = tot = 0
    for p in ident:
        d = q[p]; cs = list(d)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if d[cs[i]] == d[cs[j]] and (p, cs[i]) in lab and (p, cs[j]) in lab:
                    tot += 1; same += lab[(p, cs[i])] == lab[(p, cs[j])]
    checks["identical_question_answer_agreement"] = {"same": same, "comparisons": tot}
    subsets = {
        "all": None,
        "excl_identical": lambda r: r["pair_id"] not in ident,
        "noTH_both": lambda r: r["query_language"] != "th" and r["gui_language"] != "th",
        "lang4": lambda r: r["query_language"] in LANG4 and r["gui_language"] in LANG4,
    }
    summary = {"checks": checks}
    with (out / "contrasts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "outcome", "contrast", "pp", "ci95_lo", "ci95_hi", "p_boot", "p_holm_primary3", "items", "rows"])
        for sname, sub in subsets.items():
            for outcome in ("generation_correct", "scoring_correct"):
                items, m = item_matrix(rows, outcome, sub)
                nrows = sum(1 for r in rows if not sub or sub(r))
                res = bootstrap(m)
                summary.setdefault(sname, {})[outcome] = {
                    "items": len(items), "rows": nrows,
                    "accuracy_pct": {c: 100 * m[:, k].mean() for k, c in enumerate(CONDS)}, "contrasts": res}
                for k, v in res.items():
                    w.writerow([sname, outcome, k, round(v["pp"], 3), round(v["lo"], 3), round(v["hi"], 3),
                                round(v["p_boot"], 5), round(v["p_holm"], 5) if "p_holm" in v else "", len(items), nrows])
    with (out / "by_gui.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gui_language", "contrast", "pp", "ci95_lo", "ci95_hi", "items"])
        for g in ("en", "zh", "th", "ru", "fr", "ja"):
            items, m = item_matrix(rows, "generation_correct", lambda r, g=g: r["gui_language"] == g)
            boot = bootstrap(m)
            for k in ("stem_effect_S", "option_effect_P", "interaction_I", "R_minus_O"):
                v = boot[k]
                w.writerow([g, k, round(v["pp"], 3), round(v["lo"], 3), round(v["hi"], 3), len(items)])
    if ref:
        rr = {(x["pair_id"], x["condition"]): x for x in load(ref)}
        agree = n = 0
        for r in rows:
            k = (r["pair_id"], r["condition"])
            if k in rr and rr[k]["status"] == "success":
                n += 1; agree += r["parsed_generated_label"] == rr[k]["parsed_generated_label"]
        summary["reference_4lang"] = {"rows_matched": n, "label_agreement": agree / n if n else None}
    with (out / "glmm_rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["correct", "scoring_correct", "condition", "stem_gui", "opts_gui", "item", "pair",
                    "query_language", "gui_language", "identical_question_pair"])
        for r in rows:
            w.writerow([int(bool(r["generation_correct"])), int(bool(r["scoring_correct"])), r["condition"],
                        int(r["condition"] in ("R_stem", "R")), int(r["condition"] in ("R_opts", "R")),
                        r["parallel_id"], r["pair_id"], r["query_language"], r["gui_language"], int(r["pair_id"] in ident)])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(checks, indent=1)); print(json.dumps(summary.get("reference_4lang"), default=float))
    for s in subsets:
        g = summary[s]["generation_correct"]
        print(s, g["items"], g["rows"], {c: round(v, 1) for c, v in g["accuracy_pct"].items()})
        for k in ("R_minus_O", "stem_effect_S", "option_effect_P", "interaction_I", "S_minus_P"):
            v = g["contrasts"][k]
            print(f"  {k:16s} {v['pp']:+6.1f} [{v['lo']:+5.1f}, {v['hi']:+5.1f}] holm={v.get('p_holm', '')}")


if __name__ == "__main__":
    main()
