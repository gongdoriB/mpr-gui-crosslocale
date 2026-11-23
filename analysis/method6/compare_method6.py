"""Descriptive comparison of method conditions (M_S, M_H) with rstem6 conditions (O, R_stem, R_opts, R) for one model (2026-11-20).
Preliminary: no pre-registered analysis plan for the method run yet. Pairs are joined on pair_id; item-cluster bootstrap over parallel_id
(seed 20261109, B 10,000, as analyze_rstem6.py).
Usage: python -I compare_method6.py <method_predictions.jsonl> <rstem6_predictions.jsonl> <out_dir>
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SEED, B = 20261109, 10_000
CONDS = ("O", "R_stem", "R_opts", "R", "M_S", "M_H")
DIFFS = [("M_S", "O"), ("M_H", "O"), ("M_S", "R_stem"), ("M_H", "R"), ("M_H", "M_S"), ("R_stem", "O"), ("R", "O")]


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split", "main") != "main":
            continue
        out[(r["pair_id"], r["condition"])] = r
    return out


def boot(rows, sub=None):
    acc = defaultdict(lambda: defaultdict(list))
    for (pid, c), r in rows.items():
        if sub and not sub(r):
            continue
        acc[r["parallel_id"]][c].append(float(bool(r["correct"])))
    items = [i for i in sorted(acc) if set(acc[i]) == set(CONDS)]
    if not items:
        return {"n_items": 0}
    m = np.array([[np.mean(acc[i][c]) for c in CONDS] for i in items])
    k = {c: j for j, c in enumerate(CONDS)}
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(m), size=(B, len(m)))
    res = {"n_items": len(items), "acc": {c: round(100 * m[:, k[c]].mean(), 2) for c in CONDS}}
    for a, b in DIFFS:
        d = m[:, k[a]] - m[:, k[b]]
        draws = np.round(d[idx].mean(axis=1), 12)
        res[f"{a}-{b}"] = {"pp": round(100 * d.mean(), 2), "lo": round(100 * np.percentile(draws, 2.5), 2),
                          "hi": round(100 * np.percentile(draws, 97.5), 2)}
    return res


meth, rs = load(sys.argv[1]), load(sys.argv[2])
rows = dict(rs); rows.update(meth)
pairs = {p for p, c in meth}
rows = {k: v for k, v in rows.items() if k[0] in pairs}
cnt = defaultdict(int)
for p, c in rows: cnt[p] += 1
missing = [p for p in pairs if cnt[p] != len(CONDS)]
unparsed = {c: sum(1 for (p, cc), r in rows.items() if cc == c and not r.get("parse_success", True)) for c in CONDS}
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
summary = {"pairs": len(pairs), "pairs_missing_condition": len(missing), "unparsed": unparsed, "all": boot(rows),
           "matched": boot(rows, lambda r: r["query_language"] == r["gui_language"]),
           "mismatched": boot(rows, lambda r: r["query_language"] != r["gui_language"])}
for g in ("en", "zh", "th", "ru", "fr", "ja"):
    summary[f"gui_{g}"] = boot(rows, lambda r, g=g: r["gui_language"] == g)
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
with open(out / "table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subset", "n_items"] + list(CONDS) + [f"{a}-{b}" for a, b in DIFFS])
    for s, v in summary.items():
        if isinstance(v, dict) and "acc" in v:  # empty subsets skipped
            w.writerow([s, v["n_items"]] + [v["acc"][c] for c in CONDS] + [f'{v[f"{a}-{b}"]["pp"]} [{v[f"{a}-{b}"]["lo"]}, {v[f"{a}-{b}"]["hi"]}]' for a, b in DIFFS])
print(open(out / "table.csv").read()); print("pairs", len(pairs), "missing", len(missing), "unparsed", unparsed)
