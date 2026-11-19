"""Gate for the 6x6 cross-locale run (run_cross_locale.py output).

  python check_6x6_gate.py smoke    <output.jsonl> <plan.json> <gate.json> [--expected 216] [--model KEY]
  python check_6x6_gate.py complete <output.jsonl> <plan.json> <gate.json> [--expected 79020] [--model KEY]

KEY (qwen3_vl_2b, qwen3_vl_4b, internvl35_1b/2b/4b, uitars_15_7b, minicpm_v45) selects the visual/prompt token range.

plan.json is the file written by run_cross_locale.py --sample-manifest-out (key input_ids).
Pass when: the set of input ids with a success record (last record per id wins, resume appends)
equals the planned set and has the expected size; every success has batch_size 1; and no more than
50% of the successful rows fail to parse. Parse failures are reported (count and rate) either way.
Sanity rules (added 2026-11-14, each fails the gate):
  * all parsed generated labels identical (checked from 20 parsed rows on)
  * median A-D probability mass (sum of exp(label_logprobs)) < 0.5; if label_logprobs is missing the rule is reported as not evaluated
  * visual_token_count (or, when the adapter does not record it, prompt_token_count) outside the range of the model family:
      Qwen3-VL <= 2048 visual (prompt 800..2700), UI-TARS <= 2674 visual (prompt 800..3300), InternVL3.5 1792..2304 visual,
      MiniCPM-V-4.5 576..640 visual (measured on the 1,464 REL images; the 6x6 image set is larger, so GATE_VISUAL_RANGE=lo,hi overrides).
    A field that is not recorded is reported, never a failure.
Reported only: accuracy (a smoke accuracy below 0.30 adds a warning, not a failure), median seconds per inference, ETA of the full run.
Exit 0 = pass, 1 = fail. The last stdout line is a one-line key=value summary for the queue status.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

MAX_PARSE_FAIL_RATE = 0.5
FULL_SIZE = 79020

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
    args = sys.argv[1:]
    if len(args) < 4 or args[0] not in ("smoke", "complete"):
        raise SystemExit(__doc__)
    mode, out, plan_path, gate_path = args[0], Path(args[1]), Path(args[2]), Path(args[3])
    key = None
    if "--model" in args:
        i = args.index("--model")
        key = args[i + 1]
        del args[i:i + 2]
    expected = None
    if "--expected" in args:
        expected = int(args[args.index("--expected") + 1])
    if expected is None:
        expected = FULL_SIZE if mode == "complete" else 216
    planned = json.loads(plan_path.read_text(encoding="utf-8"))["input_ids"]
    planned_set = set(planned)
    last: dict[str, dict] = {}
    n_lines = 0
    torn = 0
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    n_lines += 1
                    try:
                        row = json.loads(line)
                        last[row["input_id"]] = row
                    except (json.JSONDecodeError, KeyError):
                        torn += 1  # a line cut off by a killed process; the id check below decides if it matters
    ok = {k: r for k, r in last.items() if r.get("status") == "success"}
    parse_fail = sum(1 for r in ok.values() if not r.get("parse_success"))
    rate = parse_fail / len(ok) if ok else 1.0
    batch_sizes = Counter(r.get("batch_size") for r in ok.values())
    missing = planned_set - set(ok)
    extra = set(ok) - planned_set
    problems = []
    if len(planned) != len(planned_set):
        problems.append(f"plan has duplicate ids ({len(planned)} vs {len(planned_set)} unique)")
    if len(planned_set) != expected:
        problems.append(f"plan size {len(planned_set)} != expected {expected}")
    if missing:
        problems.append(f"{len(missing)} planned inputs have no success record")
    if extra:
        problems.append(f"{len(extra)} success records are not in the plan")
    if len(ok) != expected:
        problems.append(f"unique successes {len(ok)} != expected {expected}")
    if set(batch_sizes) - {1}:
        problems.append(f"batch sizes other than 1: {dict(batch_sizes)}")
    if rate > MAX_PARSE_FAIL_RATE:
        problems.append(f"parse failure rate {rate:.3f} > {MAX_PARSE_FAIL_RATE}")
    sane_problems, warnings, sane_info = sanity_checks(list(ok.values()), key, mode)
    problems += sane_problems
    dims: dict[str, list[int]] = {}
    for r in ok.values():
        d = dims.setdefault(r.get("dimension") or "?", [0, 0])
        d[0] += 1
        d[1] += 0 if r.get("parse_success") else 1
    runtimes = [r["runtime_ms"] for r in ok.values() if isinstance(r.get("runtime_ms"), (int, float))]
    med_s = statistics.median(runtimes) / 1000 if runtimes else None
    mean_s = statistics.fmean(runtimes) / 1000 if runtimes else None
    vt = [r["visual_token_count"] for r in ok.values() if isinstance(r.get("visual_token_count"), (int, float))]
    result = {
        "mode": mode, "expected": expected, "planned": len(planned_set), "output_lines": n_lines, "unreadable_lines": torn,
        "unique_success": len(ok), "unique_failed_last": len(last) - len(ok),
        "missing": len(missing), "extra": len(extra), "batch_sizes": dict(batch_sizes),
        "parse_failures": parse_fail, "parse_failure_rate": round(rate, 4),
        "parse_rate": round(1 - rate, 4) if ok else None,
        "by_dimension(n,unparsed)": dims,
        "median_s_per_inference": round(med_s, 3) if med_s is not None else None,
        "mean_s_per_inference": round(mean_s, 3) if mean_s is not None else None,
        "eta_hours_full_at_median": round(med_s * FULL_SIZE / 3600, 1) if med_s is not None else None,
        "visual_tokens_max": max(vt) if vt else None,
        "eta_full_seconds": round(med_s * FULL_SIZE, 2) if med_s is not None else None,
        "model_key": key, "sanity": sane_info, "warnings": warnings,
        "problems": problems, "status": "passed" if not problems else "failed",
    }
    Path(gate_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"GATE {result['status']} unique_success={len(ok)}/{expected} parse_rate={result['parse_rate']} "
          f"unparsed={parse_fail} median_s={result['median_s_per_inference']} "
          f"eta_full_h={result['eta_hours_full_at_median']} acc={sane_info.get('accuracy')}")
    for w in warnings:
        print(f"WARNING: {w}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
