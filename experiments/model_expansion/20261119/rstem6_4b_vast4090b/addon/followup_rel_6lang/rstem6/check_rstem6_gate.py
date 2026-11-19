"""Gate for the R_stem / R_opts 6-language run (O, R_stem, R_opts, R).

  python check_rstem6_gate.py smoke|complete <out_dir> --model KEY --batch N [--expected-rows N] [--full-n N] [--mock]

Rules (problems = exit 1, warnings only reported). Written to <out_dir>/gate_<mode>.json.
  - plan: inferences == rows == success rows, summary complete, every pair has all four conditions
  - batches: batch 4 = every batch holds exactly one pair's O, R_stem, R_opts, R; batch 1 = every batch holds one input; recorded batch_size == --batch
  - complete: rows == --expected-rows (18,960)
  - parse failures: smoke <= 3, complete <= max(2, 0.35 % of the rows)
  - sanity rules shared with the earlier gates (check_sanity_shared.py = check_rel6_gate.py of the MiniCPM 6-language addon): identical labels, A-D probability mass,
    visual / prompt token range per family, system prompt rule (InternVL2.5, Qwen3)
  - reference agreement 1 "rstem4lang" (qwen3_vl_8b = run of 2026-11-09, internvl35_8b = run of 2026-11-10; the same pair x condition rows of the 4-language R_stem/R_opts run,
    reference/rstem4lang_<model>_labels.jsonl.gz): label agreement on the common rows must be >= 0.90 (problem); accuracy per condition on the common rows is
    compared with the earlier run (warning when it differs by more than 0.05, complete stage only)
  - reference agreement 2 "rel_OR" (all five models): the O and R rows have the same questions as the O/R rows of the earlier REL runs of the same model (4-language and fr/ja new_only,
    reference/rel_OR_<model>_labels.jsonl.gz; same question text checked when the file was built): label agreement >= 0.90 (problem, when >= 30 common rows)
  - the accuracy sanity of the shared rules (warning below 0.30 in the smoke) applies to every model
  - ETA fields from the median inference time
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_sanity_shared as shared  # noqa: E402

CONDS = ("O", "R_opts", "R_stem", "R")
AGREE_MIN = 0.90
PARSE_RATE = 0.0035
ACC_WARN = 0.05


def load(out: Path) -> list[dict]:
    rows = {}
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["input_id"]] = row   # last record per id wins (resume appends)
    return list(rows.values())


def reference_check(ok: list[dict], path: Path, conds: tuple[str, ...] | None) -> dict | None:
    if not path.exists():
        return None
    ref = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref[(r["pair_id"], r["condition"])] = r
    n = same = 0
    by = Counter()
    acc_new, acc_old = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for row in ok:
        old = ref.get((row["pair_id"], row["condition"]))
        if old is None or (conds is not None and row["condition"] not in conds):
            continue
        n += 1
        agree = row["parsed_generated_label"] == old["label"]
        same += agree
        by[(row["condition"], agree)] += 1
        a, b = acc_new[row["condition"]], acc_old[row["condition"]]
        a[0] += bool(row["generation_correct"]); a[1] += 1
        b[0] += bool(old["correct"]); b[1] += 1
    return {"file": path.name, "common_rows": n, "label_agreement": round(same / n, 4) if n else None,
            "by_condition": {f"{c}/{'same' if a else 'diff'}": k for (c, a), k in sorted(by.items())},
            "accuracy_now": {c: round(v[0] / v[1], 4) for c, v in sorted(acc_new.items())},
            "accuracy_earlier_run": {c: round(v[0] / v[1], 4) for c, v in sorted(acc_old.items())}}


def main() -> None:
    argv = sys.argv[1:]

    def opt(name, cast=str, default=None):
        if name in argv:
            i = argv.index(name)
            v = cast(argv[i + 1])
            del argv[i:i + 2]
            return v
        return default

    mock = "--mock" in argv
    if mock:
        argv.remove("--mock")
        shared.MOCK_RENDER = True
    model = opt("--model", str, None)
    batch = opt("--batch", int, 1)
    expected_rows = opt("--expected-rows", int)
    full_n = opt("--full-n", int)
    if len(argv) != 2 or argv[0] not in ("smoke", "complete") or model is None:
        raise SystemExit(__doc__)
    mode, out = argv[0], Path(argv[1])
    rows = load(out)
    statuses = Counter(r.get("status") for r in rows)
    ok = [r for r in rows if r.get("status") == "success"]
    parse_failures = sum(1 for r in ok if not r["parse_success"])
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    problems, warnings = [], []
    by_pair = defaultdict(set)
    for r in ok:
        by_pair[r["pair_id"]].add(r["condition"])
    bad_pairs = sum(1 for c in by_pair.values() if c != set(CONDS))
    if bad_pairs:
        problems.append(f"{bad_pairs} pairs do not have all four conditions")
    batches = defaultdict(list)
    for r in ok:
        batches[r["batch_id"]].append((r["pair_id"], r["condition"]))
    if batch == 4:
        misaligned = sum(1 for b in batches.values() if len({p for p, _ in b}) != 1 or sorted(c for _, c in b) != sorted(CONDS))
        if misaligned:
            problems.append(f"{misaligned} batches are not exactly one pair's four conditions")
    else:
        misaligned = sum(1 for b in batches.values() if len(b) != 1)
        if misaligned:
            problems.append(f"{misaligned} batches hold more than one input (expected batch size 1)")
    bs = Counter(r.get("batch_size") for r in ok)
    if set(bs) - {batch}:
        problems.append(f"batch sizes {dict(bs)} differ from the expected {batch}")
    if len(ok) != plan["inferences"] or len(rows) != plan["inferences"]:
        problems.append(f"success rows {len(ok)} / all rows {len(rows)} != planned {plan['inferences']}")
    if not summary.get("complete"):
        problems.append(f"incomplete: {summary.get('statuses')}")
    if mode == "complete" and expected_rows is not None and len(rows) != expected_rows:
        problems.append(f"rows {len(rows)} != {expected_rows}")
    limit = 3 if mode == "smoke" else max(2, math.ceil(PARSE_RATE * plan["inferences"]))
    if parse_failures > limit:
        problems.append(f"parse failures {parse_failures} > {limit}")
    sane_problems, sane_warnings, sane_info = shared.sanity_checks(ok, model, mode)
    problems += sane_problems
    warnings += sane_warnings
    refs = {}
    for name, fname in (("rstem4lang", f"rstem4lang_{model}_labels.jsonl.gz"), ("rel_OR", f"rel_OR_{model}_labels.jsonl.gz")):
        r = reference_check(ok, HERE / "reference" / fname, None)
        if r is None:
            warnings.append(f"no reference file {fname}: '{name}' comparison not evaluated")
            continue
        refs[name] = r
        if mock:
            warnings.append(f"--mock: reference '{name}' agreement {r['label_agreement']} on {r['common_rows']} common rows is not a gate rule (mock labels)")
        elif r["common_rows"] < 30:
            warnings.append(f"reference '{name}': only {r['common_rows']} common rows, agreement {r['label_agreement']} not evaluated as a rule")
        elif r["label_agreement"] < AGREE_MIN:
            problems.append(f"answers agree with the earlier {name} run on only {r['label_agreement']} of {r['common_rows']} common rows (< {AGREE_MIN})")
        if mode == "complete" and not mock:
            for c, v in r["accuracy_now"].items():
                old = r["accuracy_earlier_run"].get(c)
                if old is not None and abs(v - old) > ACC_WARN:
                    warnings.append(f"reference '{name}': accuracy of {c} on the common rows {v} vs earlier run {old}")
    acc = defaultdict(lambda: [0, 0])
    by_lang = {"query": {}, "gui": {}}
    for r in ok:
        c = acc[r["condition"]]
        c[0] += 1
        c[1] += bool(r["generation_correct"])
        for side, lang in (("query", r["query_language"]), ("gui", r["gui_language"])):
            cell = by_lang[side].setdefault(lang, [0, 0, 0])
            cell[0] += 1
            cell[1] += bool(r["generation_correct"])
            cell[2] += not r["parse_success"]
    result = {"mode": mode, "model": model, "batch": batch, "rows": len(rows), "statuses": dict(statuses), "batch_sizes": dict(bs),
              "parse_failures": parse_failures, "parse_failure_limit": limit, "misaligned_batches": misaligned,
              "accuracy": summary.get("accuracy"), "by_condition(n,correct)": dict(acc), "by_language(n,correct,unparsed)": by_lang,
              "references": refs, "sanity": sane_info, "warnings": warnings,
              "timing": shared.eta_info(ok, full_n or plan["inferences"]), "problems": problems,
              "status": "passed" if not problems else "failed"}
    (out / f"gate_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
