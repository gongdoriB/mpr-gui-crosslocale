"""Gates for models without an earlier run contract (generic mode). No anchor-agreement gate.

  python check_generic_gate.py rel  smoke    <out_dir> [--model KEY]            # REL O/R/C/F (run_inference.py output)
  python check_generic_gate.py rel  complete <out_dir> [18432] [--model KEY]    # expected successes (default: plan.json actual_inferences)
  python check_generic_gate.py expa smoke    <out_dir> [--model KEY]            # Exp A O/T/R (run_rel_T_generic.py output)
  python check_generic_gate.py expa complete <out_dir> [5688] [--model KEY]

KEY (qwen3_vl_2b, qwen3_vl_4b, internvl35_1b/2b/4b, uitars_15_7b) selects the visual/prompt token range.

Pass when: every planned row has a success record (last record per input_id), every batch has size 1,
no more than 50% of rows fail to parse, and the answers pass the sanity rules below.
Sanity rules (added 2026-11-14, each fails the gate):
  * all parsed generated labels identical (checked from 20 parsed rows on)
  * median A-D probability mass (sum of exp(label_logprobs)) < 0.5; if label_logprobs is missing the rule is reported as not evaluated
  * visual_token_count (or, when the adapter does not record it, prompt_token_count) outside the range of the model family:
      Qwen3-VL <= 2048 visual (prompt 800..2700), UI-TARS <= 2674 visual (prompt 800..3300), InternVL3.5 1792..2304 visual.
    A field that is not recorded is reported, never a failure.
Reported only: parse failures, accuracy (a smoke accuracy below 0.30 adds a warning, not a failure),
median seconds per inference and the ETA of the full step (eta_full_seconds, used by the queue to stop a step that cannot finish in its timeout).
Exit 0 = pass, 1 = fail. Result in <out_dir>/gate_<kind>_<mode>.json.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

MAX_PARSE_FAIL_RATE = 0.5
FULL_INFERENCES = {"rel": 18432, "expa": 5688}

MIN_ROWS_LABEL_RULE = 20
MIN_ABCD_MASS = 0.5
SMOKE_ACC_WARN = 0.30
# Token ranges. Qwen3-VL: max_pixels 2,097,152 / (32 px)^2 = 2,048. UI-TARS (Qwen2.5-VL): 2,097,152 / (28 px)^2 = 2,674.
# InternVL3.5 with max_num 8 + thumbnail: 1,792 (landscape) or 2,304 (portrait); the 8B 6x6 run had exactly these two values.
# Prompt-token ranges are used only when the adapter does not record visual_token_count (Qwen adapter: prompt only).
# Observed in the Qwen3-VL-8B runs (2026-11-07/08): prompt tokens 1,711..2,515.
TOKEN_RULES = {
    "qwen3_vl": {"visual": (0, 2048), "prompt": (800, 2700)},
    "uitars": {"visual": (0, 2674), "prompt": (800, 3300)},
    "internvl35": {"visual": (1792, 2304), "prompt": None},
    "minicpm": {"visual": (576, 640), "prompt": None},
}


def family_of(key: str | None) -> str | None:
    key = key or ""
    for prefix, fam in (("qwen3_vl", "qwen3_vl"), ("uitars", "uitars"), ("internvl35", "internvl35"), ("minicpm", "minicpm")):
        if key.startswith(prefix):
            return fam
    return None


def sanity_checks(ok: list[dict], key: str | None, mode: str) -> tuple[list[str], list[str], dict]:
    """Returns (problems, warnings, info) for the successful rows."""
    problems, warnings, info = [], [], {}
    labels = [r.get("parsed_generated_label") for r in ok if r.get("parsed_generated_label")]
    info["distinct_generated_labels"] = dict(Counter(labels))
    if len(labels) >= MIN_ROWS_LABEL_RULE:
        if len(set(labels)) == 1:
            problems.append(f"all {len(labels)} parsed labels are '{labels[0]}'")
    else:
        info["identical_labels_rule"] = f"not evaluated (only {len(labels)} parsed rows, need {MIN_ROWS_LABEL_RULE})"
    masses = []
    for r in ok:
        lp = r.get("label_logprobs")
        if isinstance(lp, dict) and lp:
            masses.append(sum(math.exp(v) for v in lp.values() if isinstance(v, (int, float))))
    if masses:
        med = statistics.median(masses)
        info["median_abcd_mass"] = round(med, 4)
        if med < MIN_ABCD_MASS:
            problems.append(f"median A-D probability mass {med:.3f} < {MIN_ABCD_MASS}")
    else:
        info["median_abcd_mass"] = "not evaluated: label_logprobs missing in the predictions"
    fam = family_of(key)
    rule = TOKEN_RULES.get(fam) if fam else None
    visual = [r["visual_token_count"] for r in ok if isinstance(r.get("visual_token_count"), (int, float))]
    prompt = [r["prompt_token_count"] for r in ok if isinstance(r.get("prompt_token_count"), (int, float))]
    info["visual_token_count"] = {"min": min(visual), "max": max(visual), "n": len(visual)} if visual else "not recorded"
    info["prompt_token_count"] = {"min": min(prompt), "max": max(prompt), "n": len(prompt)} if prompt else "not recorded"
    if rule is None:
        info["token_range_rule"] = f"not evaluated (model key {key!r} has no family rule)"
    elif visual:
        lo, hi = rule["visual"]
        bad = [v for v in visual if not lo <= v <= hi]
        if bad:
            problems.append(f"{len(bad)}/{len(visual)} rows have visual_token_count outside {lo}..{hi} (min {min(bad)}, max {max(bad)})")
    elif prompt and rule["prompt"]:
        lo, hi = rule["prompt"]
        bad = [v for v in prompt if not lo <= v <= hi]
        info["token_range_rule"] = "visual_token_count not recorded by this adapter; prompt_token_count used"
        if bad:
            problems.append(f"{len(bad)}/{len(prompt)} rows have prompt_token_count outside {lo}..{hi} (min {min(bad)}, max {max(bad)})")
    else:
        info["token_range_rule"] = "not evaluated: neither visual_token_count nor a usable prompt_token_count recorded"
    correct = [bool(r.get("generation_correct", r.get("correct"))) for r in ok if r.get("generation_correct", r.get("correct")) is not None]
    if correct:
        acc = sum(correct) / len(correct)
        info["accuracy"] = round(acc, 4)
        if mode == "smoke" and acc < SMOKE_ACC_WARN:
            warnings.append(f"smoke accuracy {acc:.3f} < {SMOKE_ACC_WARN} (chance is 0.25); check the output before trusting the full run")
    return problems, warnings, info


def eta_info(ok: list[dict], full_n: int) -> dict:
    ms = [r["inference_runtime_ms"] for r in ok if isinstance(r.get("inference_runtime_ms"), (int, float))]
    if not ms:
        return {"median_s_per_inference": None, "eta_full_seconds": None, "full_inferences": full_n}
    med = statistics.median(ms) / 1000
    return {"median_s_per_inference": round(med, 3), "full_inferences": full_n,
            "eta_full_seconds": round(med * full_n, 2), "eta_full_hours": round(med * full_n / 3600, 2)}



def load(out: Path) -> dict[str, dict]:
    rows = {}
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["input_id"]] = row  # last record per id wins (resume appends)
    return rows


def main() -> None:
    argv = sys.argv[1:]
    key = None
    if "--model" in argv:
        i = argv.index("--model")
        key = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 3 or argv[0] not in ("rel", "expa") or argv[1] not in ("smoke", "complete"):
        raise SystemExit(__doc__)
    kind, mode, out = argv[0], argv[1], Path(argv[2])
    expected_arg = int(argv[3]) if len(argv) > 3 else None
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    planned = plan["actual_inferences"] if kind == "rel" else plan["inferences"]
    expected = expected_arg if expected_arg is not None else planned
    rows = load(out)
    statuses = Counter(r.get("status") for r in rows.values())
    ok = [r for r in rows.values() if r.get("status") == "success"]
    parse_failures = sum(1 for r in ok if not r["parse_success"])
    batch_sizes = Counter(r.get("batch_size") for r in ok)
    problems = []
    if len(ok) != expected or len(rows) != expected:
        problems.append(f"success rows {len(ok)} / all rows {len(rows)} != expected {expected}")
    if planned != expected:
        problems.append(f"plan says {planned}, expected {expected}")
    if set(batch_sizes) - {1}:
        problems.append(f"batch sizes other than 1: {dict(batch_sizes)}")
    rate = parse_failures / len(ok) if ok else 1.0
    if rate > MAX_PARSE_FAIL_RATE:
        problems.append(f"parse failure rate {rate:.3f} > {MAX_PARSE_FAIL_RATE}")
    sane_problems, warnings, sane_info = sanity_checks(ok, key, mode)
    problems += sane_problems
    by_cond = {}
    for r in ok:
        c = by_cond.setdefault(r.get("condition") or r["input_id"].rsplit("::", 1)[-1], [0, 0, 0])
        c[0] += 1
        c[1] += bool(r["generation_correct"])
        c[2] += not r["parse_success"]
    result = {"kind": kind, "mode": mode, "expected_rows": expected, "rows": len(rows), "statuses": dict(statuses),
              "batch_sizes": dict(batch_sizes), "parse_failures": parse_failures, "parse_failure_rate": round(rate, 4),
              "by_condition(n,correct,unparsed)": by_cond,
              "note": "REL rows are deduplicated inferences (conditions with identical prompts share one row)"
              if kind == "rel" else "",
              "model_key": key, "sanity": sane_info, "warnings": warnings, "timing": eta_info(ok, FULL_INFERENCES[kind]),
              "problems": problems, "status": "passed" if not problems else "failed"}
    (out / f"gate_{kind}_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
