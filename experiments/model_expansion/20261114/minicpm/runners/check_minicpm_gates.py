"""Gate for the MiniCPM-V REL O/R/C/F run (no earlier run of this model exists, so no anchor).

  python check_minicpm_gates.py rel smoke    <out_dir>
  python check_minicpm_gates.py rel complete <out_dir>

Pass = every planned inference succeeded, at most half of the answers are unparsed, and the sanity rules hold
(added 2026-11-14, each fails the gate):
  * all parsed generated labels identical (checked from 20 parsed rows on)
  * median A-D probability mass (sum of exp(label_logprobs)) < 0.5 (reported as not evaluated if label_logprobs is missing)
  * visual_token_count outside 576..640 (MiniCPM-V-4.5: 9 or 10 slices of 64 tokens; measured on the 1,464 REL images).
    GATE_VISUAL_RANGE=lo,hi overrides. A field that is not recorded is reported, never a failure.
Reported only: parse failures, speed, ETA, generation/scoring disagreement, accuracy (smoke accuracy below 0.30 adds a warning).
Exit 0 = pass, 1 = fail. Result in gate_<mode>.json.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

MAX_UNPARSED_FRACTION = 0.5
FULL_REL_INFERENCES = 18432
FULL_EXPA_INFERENCES = 5688
MIN_ROWS_LABEL_RULE = 20
MIN_ABCD_MASS = 0.5
SMOKE_ACC_WARN = 0.30
# Qwen3-VL: max_pixels 2,097,152 / (32 px)^2 = 2,048. UI-TARS (Qwen2.5-VL): 2,097,152 / (28 px)^2 = 2,674.
# InternVL3.5 (max_num 8 + thumbnail): 1,792 or 2,304; the 8B 6x6 run had exactly these two values.
# Prompt ranges are used only when the adapter does not record visual_token_count (Qwen adapter: prompt only; observed 1,712..2,515 on Qwen3-VL-8B 6x6).
TOKEN_RULES = {
    "qwen3_vl": {"visual": (0, 2048), "prompt": (800, 2700)},
    "uitars": {"visual": (0, 2674), "prompt": (800, 3300)},
    "internvl35": {"visual": (1792, 2304), "prompt": None},
    "minicpm": {"visual": (576, 640), "prompt": None},
}


def family_of(key):
    key = key or ""
    for prefix, fam in (("qwen3_vl", "qwen3_vl"), ("uitars", "uitars"), ("internvl35", "internvl35"), ("minicpm", "minicpm")):
        if key.startswith(prefix):
            return fam
    return None


def sanity_checks(ok, key, mode):
    """ok = list of successful rows. Returns (problems, warnings, info)."""
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
    override = os.environ.get("GATE_VISUAL_RANGE")
    visual = [r["visual_token_count"] for r in ok if isinstance(r.get("visual_token_count"), (int, float))]
    prompt = [r["prompt_token_count"] for r in ok if isinstance(r.get("prompt_token_count"), (int, float))]
    info["visual_token_count"] = {"min": min(visual), "max": max(visual), "n": len(visual)} if visual else "not recorded"
    info["prompt_token_count"] = {"min": min(prompt), "max": max(prompt), "n": len(prompt)} if prompt else "not recorded"
    if rule is None:
        info["token_range_rule"] = f"not evaluated (model key {key!r} has no family rule)"
    elif visual:
        lo, hi = tuple(int(x) for x in override.split(",")) if override else rule["visual"]
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



def main() -> None:
    kind, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if kind != "rel" or mode not in ("smoke", "complete"):
        raise SystemExit("Usage: check_minicpm_gates.py rel <smoke|complete> <out_dir>")
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cache = {}
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["input_id"]] = row
    rows = list(cache.values())
    success = [r for r in rows if r.get("status") == "success"]
    statuses = Counter(r.get("status") for r in rows)
    unparsed = sum(1 for r in success if not r["parse_success"])
    problems = []
    if len(success) != plan["actual_inferences"]:
        problems.append(f"successful inferences {len(success)} != planned {plan['actual_inferences']}")
    if summary["statuses"] != {"success": plan["condition_rows"]}:
        problems.append(f"condition rows not all successful: {summary['statuses']}")
    if mode == "complete" and plan["actual_inferences"] != FULL_REL_INFERENCES:
        problems.append(f"plan has {plan['actual_inferences']} inferences, expected {FULL_REL_INFERENCES}")
    fraction = unparsed / len(success) if success else 1.0
    if fraction > MAX_UNPARSED_FRACTION:
        problems.append(f"unparsed fraction {fraction:.3f} > {MAX_UNPARSED_FRACTION}")
    sane_problems, warnings, sane_info = sanity_checks(success, "minicpm_v45", mode)
    problems += sane_problems
    visual = [r["visual_token_count"] for r in success if r.get("visual_token_count") is not None]
    runtimes = [r["inference_runtime_ms"] for r in success if r.get("inference_runtime_ms") is not None]
    median_ms = statistics.median(runtimes) if runtimes else None
    result = {
        "kind": kind, "mode": mode, "planned_inferences": plan["actual_inferences"],
        "predictions": len(rows), "statuses": dict(statuses),
        "parse_failures": unparsed, "unparsed_fraction": round(fraction, 4),
        "visual_token_count": {"min": min(visual), "mean": round(statistics.mean(visual), 1), "max": max(visual)} if visual else None,
        "prompt_token_count_max": max((r["prompt_token_count"] for r in success if r.get("prompt_token_count") is not None), default=None),
        "label_scoring_methods": dict(Counter(r.get("label_scoring_method") for r in success)),
        "generation_scoring_disagreement_rate": round(
            sum(bool(r.get("generation_scoring_disagreement")) for r in success) / len(success), 4) if success else None,
        "median_inference_ms": median_ms,
        "eta_hours_full_rel_at_median": round(FULL_REL_INFERENCES * median_ms / 3.6e6, 2) if median_ms else None,
        "eta_hours_full_expA_at_median": round(FULL_EXPA_INFERENCES * median_ms / 3.6e6, 2) if median_ms else None,
        "eta_full_seconds": round(FULL_REL_INFERENCES * median_ms / 1000, 2) if median_ms else None,
        "sanity": sane_info, "warnings": warnings,
        "primary_descriptive_summary": summary.get("primary_descriptive_summary"),
        "problems": problems, "status": "passed" if not problems else "failed",
    }
    (out / f"gate_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
