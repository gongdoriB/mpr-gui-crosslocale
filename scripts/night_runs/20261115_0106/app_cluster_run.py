"""App-cluster bootstrap for Exp A and REL, 4 models. Exploratory (post hoc). Writes only to this folder.
Reuses (import, no edit): analyze_expA.py, run_frozen_analysis.py (rel_quartets, rel_boot, VARIANTS, REL_CON, CONTEXT, PREP).
Cluster resampling logic copied from robustness_free_20261114/task1_task3_expA.py (cluster_boot).
Usage: python3 -I app_cluster_run.py [repro|expA|rel|all]   (all = repro first, stop if it fails)"""
import sys
sys.dont_write_bytecode = True
import csv, re, json, importlib.util
from pathlib import Path
import numpy as np

BASE = Path("${PROJECT_ROOT}")
OUT = BASE / "night_runs/20261115_0106"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


F = load_mod(BASE / "analysis_newmodels_20261114/run_frozen_analysis.py", "frozen")
A = load_mod(BASE / "expA_translation_20261112/analysis/analyze_expA.py", "ea")
prep = F.load_mod(F.PREP, "prep_glmm")
B = 10_000
SEED_A = 20261113
APP_RE = re.compile(r"^rel::\d+/(.+)_\d+\.[A-Za-z]+$")


def app_of(pid):
    m = APP_RE.match(pid)
    assert m, pid
    return m.group(1)


MODELS = {
    "qwen3_8b": ["Qwen3-VL-8B", F.CONTEXT["Qwen3-VL-8B"]["expa"], F.CONTEXT["Qwen3-VL-8B"]["rel"]],
    "internvl35_8b": ["InternVL3.5-8B", F.CONTEXT["InternVL3.5-8B"]["expa"], F.CONTEXT["InternVL3.5-8B"]["rel"]],
    "qwen3_vl_4b": ["Qwen3-VL-4B", None, None],
    "internvl35_4b": ["InternVL3.5-4B", None, None],
}
for k in ("qwen3_vl_4b", "internvl35_4b"):
    un = BASE / ("model_results_20261114/%s/unpacked/sizes_%s" % (k, k))
    MODELS[k][1] = un / "expa_full/predictions.jsonl"
    MODELS[k][2] = un / "rel_full/results.jsonl"

_refcache = {}


def ref_json(k):
    if k not in _refcache:
        _refcache[k] = json.loads((BASE / ("analysis_newmodels_20261114/%s/results.json" % k)).read_text())
    return _refcache[k]


def ref_for(model_key, kind, scope, contrast, stratum="dependent"):
    """item-level reference: 4B own results.json; 8B from context_8b inside the qwen3_vl_4b results.json"""
    if model_key in ("qwen3_vl_4b", "internvl35_4b"):
        R = ref_json(model_key)
        return R["rel"]["generation_correct"][scope][stratum][contrast] if kind == "rel" else R["expa"][scope]["generation_correct"][contrast]
    R = ref_json("qwen3_vl_4b")["context_8b"][MODELS[model_key][0]]
    return R["rel"][scope][stratum][contrast] if kind == "rel" else R["expa"][scope]["generation_correct"][contrast]


def cluster_percentile(sums, cnt, seed, mode):
    K = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(B, K))
    d = sums[idx].sum(1) / cnt[idx].sum(1)
    if mode == "expA":  # same as task1_task3_expA.cluster_boot
        d = np.round(d, 12)
        return 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)
    d = np.sort(100 * d)  # rel_boot convention
    return d[int(0.025 * (B - 1))], d[int(0.975 * (B - 1))]


def contains0(lo, hi):
    return bool(lo <= 0 <= hi)


def expa_rows(model_key, seed=SEED_A, scopes=None):
    label, pred, _ = MODELS[model_key]
    rows = A.load_predictions(Path(pred))
    ok = {}
    for r in rows:
        if r["status"] == "success":
            ok.setdefault(r["pair_id"], set()).add(r["condition"])
    complete = {p for p, c in ok.items() if c == set(A.CONDS)}
    rows = [r for r in rows if r["pair_id"] in complete]
    prim = A.pair_sets()["primary_full"] & complete
    A.SEED, A.B = seed, B
    out = []
    for sc, f in (scopes or F.VARIANTS).items():
        P = {p for p in prim if f(p.split("::")[-2], p.split("::")[-1])}
        items, m = A.item_matrix(rows, "generation_correct", P)
        it = A.bootstrap(m)
        apps = sorted({app_of(i) for i in items})
        ix = {a: j for j, a in enumerate(apps)}
        grp = np.array([ix[app_of(i)] for i in items])
        cnt = np.bincount(grp, minlength=len(apps)).astype(float)
        C = A.contrasts(m)
        for k in ("T_minus_O", "R_minus_T", "R_minus_O"):
            sums = np.bincount(grp, weights=C[k], minlength=len(apps))
            lo, hi = cluster_percentile(sums, cnt, seed, "expA")
            ilo, ihi = it[k]["lo"], it[k]["hi"]
            row = dict(model=label, exp="A", scope=sc, contrast=k, estimate=it[k]["pp"], item_ci_lo=ilo, item_ci_hi=ihi,
                       app_ci_lo=lo, app_ci_hi=hi, width_ratio=(hi - lo) / (ihi - ilo),
                       zero_call_changed=contains0(ilo, ihi) != contains0(lo, hi),
                       item_zero=contains0(ilo, ihi), app_zero=contains0(lo, hi),
                       pairs=len(P), items=len(items), apps=len(apps), seed=seed, B=B, stratum="all")
            if seed == SEED_A:  # compare with results.json
                ref = ref_for(model_key, "A", sc, k)
                row["item_matches_results_json"] = all(round(it[k][a], 1) == round(ref[b], 1) for a, b in (("pp", "pp"), ("lo", "lo"), ("hi", "hi")))
            out.append(row)
    return out


def rel_rows(model_key, notes):
    label, _, relp = MODELS[model_key]
    raw, q = F.rel_quartets(prep, "ctx", Path(relp))
    out = []
    for sc, f in F.VARIANTS.items():
        sel = [x for x in q if f(x["query_language"], x["gui_language"]) and x["text_dependency"] == "dependent"]
        ids = sorted({r["parallel_id"] for r in sel})
        ix = {p: i for i, p in enumerate(ids)}
        K = len(ids)
        n = np.zeros(K)
        sums = {c: np.zeros(K) for c in F.REL_CON}
        for r in sel:
            i = ix[r["parallel_id"]]
            n[i] += 1
            for c in F.REL_CON:
                sums[c][i] += r[c]
        itres = F.rel_boot(sel)
        apps = sorted({app_of(p) for p in ids})
        ax = {a: j for j, a in enumerate(apps)}
        grp = np.array([ax[app_of(p)] for p in ids])
        acnt = np.bincount(grp, weights=n, minlength=len(apps))
        for c in F.REL_CON:
            e = itres[c]
            ref = ref_for(model_key, "rel", sc, c)
            match = all(round(e[a], 1) == round(ref[a], 1) for a in ("est_pp", "lo", "hi"))
            if not match:
                notes.append("REL item-level mismatch %s %s %s: mine %.3f [%.3f,%.3f] vs results.json %.3f [%.3f,%.3f]" % (
                    label, sc, c, e["est_pp"], e["lo"], e["hi"], ref["est_pp"], ref["lo"], ref["hi"]))
            s = np.bincount(grp, weights=sums[c], minlength=len(apps))
            lo, hi = cluster_percentile(s, acnt, 42, "rel")
            out.append(dict(model=label, exp="REL", scope=sc, contrast=c, estimate=e["est_pp"], item_ci_lo=e["lo"], item_ci_hi=e["hi"],
                            app_ci_lo=lo, app_ci_hi=hi, width_ratio=(hi - lo) / (e["hi"] - e["lo"]),
                            zero_call_changed=contains0(e["lo"], e["hi"]) != contains0(lo, hi),
                            item_zero=contains0(e["lo"], e["hi"]), app_zero=contains0(lo, hi),
                            pairs=int(n.sum()), items=K, apps=len(apps), seed=42, B=B, stratum="dependent",
                            item_matches_results_json=match))
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def blocked(msg):
    with open(OUT / "blocked.md", "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("repro", "all"):
        r = expa_rows("qwen3_8b", seed=20261114, scopes={"all4": F.VARIANTS["all4"]})
        ref = list(csv.DictReader(open(BASE / "robustness_free_20261114/outputs/task3_cluster_bootstrap_expA_qwen3.csv")))
        lines, okall = [], True
        for x in r:
            for kind, lo, hi in (("item", "item_ci_lo", "item_ci_hi"), ("app", "app_ci_lo", "app_ci_hi")):
                y = next(z for z in ref if z["resampling"] == kind and z["contrast"] == x["contrast"])
                ok = abs(float(y["lo"]) - x[lo]) < 1e-3 and abs(float(y["hi"]) - x[hi]) < 1e-3 and abs(float(y["pp"]) - x["estimate"]) < 1e-3
                okall &= ok
                lines.append("%s %s: mine [%.3f,%.3f] ref [%s,%s] %s" % (kind, x["contrast"], x[lo], x[hi], y["lo"], y["hi"], "OK" if ok else "MISMATCH"))
        (OUT / "repro_check.txt").write_text("\n".join(lines) + "\nALL_OK=%s\n" % okall)
        print("\n".join(lines), okall)
        if not okall:
            blocked("- 재현 실패: 실험 A qwen3 8B 앱 군집 값이 기존 task3_cluster_bootstrap_expA_qwen3.csv와 다름. repro_check.txt 참고. 중단.")
            sys.exit(2)
    if what in ("expA", "all"):
        allr, alt = [], []
        for k in MODELS:
            try:
                allr += expa_rows(k)
                alt += expa_rows(k, seed=20261114)
            except Exception as ex:
                blocked("- Exp A %s: %r" % (k, ex))
            if allr:
                write(OUT / "app_cluster_expA.csv", allr)
                write(OUT / "app_cluster_expA_seed20261114.csv", alt)
                print("expA done", k)
    if what in ("rel", "all"):
        allr, notes = [], []
        for k in MODELS:
            try:
                allr += rel_rows(k, notes)
            except Exception as ex:
                blocked("- REL %s: %r" % (k, ex))
            if allr:
                write(OUT / "app_cluster_rel.csv", allr)
                print("rel done", k)
        (OUT / "rel_item_check.txt").write_text("\n".join(notes) + ("\n" if notes else "") + "mismatches=%d\n" % len(notes))
        print(notes)
