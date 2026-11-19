"""Gates for the Experiment A (O, T, R) fr/ja run of Qwen3-VL-8B. Derived from followup_rel_4lang/expA/check_T_gate.py (sha 70a06d38...) with these changes:
  X1 no earlier reference file (the pairs are new). Instead the O and R answers are compared with the REL new_only run of the same server and model (--rel-results, optional;
     reported, and a problem only below the 0.90 anchor threshold of the 4-language gate, at the complete stage).
  X2 batch rule from --batch: 3 = every batch is exactly one pair's three conditions; 1 = every batch holds one input.
  X3 expected row count from --expected-rows (complete) instead of 5,688; parse-failure limits scale with the row count (0.35 % like the 4-language run, at least 2).
  X4 label sanity rules, Qwen prompt-token range and system-prompt rule are imported from check_q3_gate.py (same folder, shared with the REL gate).
  X5 per-language and per-split report, ETA fields (median s per inference, ETA of the full run).

  python check_T6_gate.py <main|retry> smoke|complete <out_dir> --batch N [--expected-rows N] [--full-n N] [--rel-results <REL results.jsonl>]
Exit 0 = pass, 1 = fail. Result in gate_<mode>.json.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import check_q3_gate as shared  # noqa: E402

ANCHOR_MIN = 0.90
PARSE_RATE = 0.0035


def load(out: Path) -> list[dict]:
    rows = {}
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["input_id"]] = row
    return list(rows.values())


def anchor_vs_rel(rows: list[dict], rel_path: Path) -> dict:
    ref = {}
    with rel_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            iid = r["input_id"]
            if iid.endswith("::O") or iid.endswith("::R"):
                ref[iid] = r
    same = total = missing = text_diff = 0
    by = Counter()
    for row in rows:
        if row["condition"] not in ("O", "R") or row.get("status") != "success":
            continue
        old = ref.get(f"rel_orcf::{row['pair_id']}::{row['condition']}")
        if old is None or old.get("status") != "success":
            missing += 1
            continue
        if row["rendered_prompt"].find(old["question"]) < 0:
            text_diff += 1
            continue
        total += 1
        agree = row["parsed_generated_label"] == old["parsed_generated_label"]
        same += agree
        by[(row["condition"], agree)] += 1
    return {"anchor_rows": total, "label_agreement": round(same / total, 4) if total else None, "rows_without_rel_row": missing,
            "question_text_differs_from_rel_row": text_diff,
            "by_condition": {f"{c}/{'same' if a else 'diff'}": n for (c, a), n in sorted(by.items())}}


def t_equals_r_agreement(rows: list[dict], t_cond: str) -> dict:
    by_pair = {}
    for r in rows:
        if r.get("status") == "success":
            by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r
    n = same = 0
    for c in by_pair.values():
        if t_cond in c and "R" in c and c[t_cond]["rendered_prompt"] == c["R"]["rendered_prompt"]:
            n += 1
            same += c[t_cond]["parsed_generated_label"] == c["R"]["parsed_generated_label"]
    return {"pairs_T_equals_R": n, "same_answer": same}


def main() -> None:
    argv = sys.argv[1:]

    def opt(name, cast=str, default=None):
        if name in argv:
            i = argv.index(name)
            v = cast(argv[i + 1])
            del argv[i:i + 2]
            return v
        return default

    if "--mock" in argv:
        argv.remove("--mock")
        shared.MOCK_RENDER = True
    batch = opt("--batch", int, 3)
    expected_rows = opt("--expected-rows", int)
    full_n = opt("--full-n", int)
    rel_results = opt("--rel-results", Path)
    if len(argv) != 3 or argv[0] not in ("main", "retry") or argv[1] not in ("smoke", "complete"):
        raise SystemExit(__doc__)
    variant, mode, out = argv
    out = Path(out)
    t_cond = "T" if variant == "main" else "T_retry"
    rows = load(out)
    statuses = Counter(r.get("status") for r in rows)
    ok = [r for r in rows if r.get("status") == "success"]
    parse_failures = sum(1 for r in ok if not r["parse_success"])
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    problems, warnings = [], []
    batches = {}
    for r in ok:
        batches.setdefault(r["batch_id"], []).append((r["pair_id"], r["condition"]))
    if batch == 3:
        misaligned = sum(1 for b in batches.values() if len({p for p, _ in b}) != 1 or sorted(c for _, c in b) != sorted(["O", t_cond, "R"]))
        if misaligned:
            problems.append(f"{misaligned} batches are not exactly one pair's O, {t_cond}, R")
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
    sane_problems, sane_warnings, sane_info = shared.sanity_checks(ok, "qwen3_vl_8b", mode)
    problems += sane_problems
    warnings += sane_warnings
    anchor = None
    if rel_results is not None and rel_results.exists():
        anchor = anchor_vs_rel(rows, rel_results)
        if mode == "complete" and anchor["label_agreement"] is not None and anchor["label_agreement"] < ANCHOR_MIN:
            problems.append(f"O/R answers agree with the REL new_only run on only {anchor['label_agreement']} (< {ANCHOR_MIN})")
        if anchor["question_text_differs_from_rel_row"]:
            problems.append(f"{anchor['question_text_differs_from_rel_row']} O/R questions differ from the REL rows of the same pair")
    elif rel_results is not None:
        warnings.append(f"REL results {rel_results} not found: O/R anchor comparison not evaluated")
    by_lang = {"query": {}, "gui": {}}
    by_split = {}
    for r in ok:
        for side, lang in (("query", r["query_language"]), ("gui", r["gui_language"])):
            c = by_lang[side].setdefault(lang, [0, 0, 0])
            c[0] += 1
            c[1] += bool(r["generation_correct"])
            c[2] += not r["parse_success"]
        c = by_split.setdefault(f"{r['split']}/{r['condition']}", [0, 0])
        c[0] += 1
        c[1] += bool(r["generation_correct"])
    result = {"variant": variant, "mode": mode, "batch": batch, "rows": len(rows), "statuses": dict(statuses), "batch_sizes": dict(bs),
              "parse_failures": parse_failures, "parse_failure_limit": limit, "misaligned_batches": misaligned,
              "accuracy": summary.get("accuracy"), "by_language(n,correct,unparsed)": by_lang, "by_split_condition(n,correct)": by_split,
              "anchor_vs_rel_new_only_run": anchor, "T_equals_R_same_input_check": t_equals_r_agreement(rows, t_cond),
              "model_key": "qwen3_vl_8b", "sanity": sane_info, "warnings": warnings,
              "timing": shared.eta_info(ok, full_n or plan["inferences"]), "problems": problems,
              "status": "passed" if not problems else "failed"}
    (out / f"gate_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
