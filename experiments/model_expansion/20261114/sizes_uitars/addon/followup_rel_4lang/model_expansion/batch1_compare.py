"""Compare batch-1 Exp A answers of Qwen3-VL-8B with the batch-4 REL answers (O and R rows) of the 2026-11-07 run.

  python batch1_compare.py <predictions.jsonl of run_rel_T_generic.py> <reference_qwen3_20261107_OR.jsonl> <out.json>

Only O and R rows can be compared (the reference has no T rows). Matching key: (pair_id, condition).
The question text of the reference must appear in the rendered prompt of the new row; rows where it does not are counted
separately and left out of the agreement. Result is a record only, no pass/fail. Exit 0 whenever the files could be read.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def last_per_id(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line cut off by a killed process
            rows[r["input_id"]] = r
    return rows


def main() -> None:
    pred_path, ref_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ref = {}
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ref[(r["pair_id"], r["condition"])] = r
    rows = last_per_id(pred_path)
    ok = [r for r in rows.values() if r.get("status") == "success"]
    n = same = question_mismatch = 0
    by_cond = Counter()
    diffs, gp_diff = [], []
    for r in ok:
        cond = r.get("condition")
        if cond not in ("O", "R"):
            continue
        old = ref.get((r["pair_id"], cond))
        if old is None or old.get("status") != "success":
            continue
        if old["question"] not in r["rendered_prompt"]:
            question_mismatch += 1
            continue
        n += 1
        agree = r.get("parsed_generated_label") == old.get("parsed_generated_label")
        same += agree
        by_cond[(cond, "same" if agree else "diff")] += 1
        if not agree and len(diffs) < 25:
            diffs.append({"pair_id": r["pair_id"], "condition": cond, "batch4": old.get("parsed_generated_label"),
                          "batch1": r.get("parsed_generated_label")})
        if isinstance(r.get("gold_probability"), (int, float)) and isinstance(old.get("gold_probability"), (int, float)):
            gp_diff.append(abs(r["gold_probability"] - old["gold_probability"]))
    result = {
        "what": "Qwen3-VL-8B REL questions at batch size 1 (this run) vs batch size 4 (2026-11-07 run); O and R rows",
        "predictions_file": str(pred_path), "reference_file": str(ref_path),
        "inferences_in_run": len(rows), "successful": len(ok),
        "batch_sizes": dict(Counter(r.get("batch_size") for r in ok)),
        "compared_rows": n, "same_label": same,
        "label_agreement": round(same / n, 4) if n else None,
        "by_condition": {f"{c}/{k}": v for (c, k), v in sorted(by_cond.items())},
        "question_text_mismatch_rows_skipped": question_mismatch,
        "mean_abs_gold_probability_diff": round(statistics.fmean(gp_diff), 5) if gp_diff else None,
        "max_abs_gold_probability_diff": round(max(gp_diff), 5) if gp_diff else None,
        "first_disagreements": diffs,
        "note": "A record of how much batch 4 vs batch 1 changes answers (bfloat16, padding, kernels). Not a pass/fail gate.",
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"BATCH1 compared={n} same={same} agreement={result['label_agreement']}")


if __name__ == "__main__":
    main()
