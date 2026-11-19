"""Gates for the Experiment A run (O, T, R).

  python check_T_gate.py <qwen3|internvl35|generic> smoke    <out_dir>
  python check_T_gate.py <qwen3|internvl35|generic> complete <out_dir>

generic  : model without an earlier O/R run (no anchor reference, no package comparison); used for MiniCPM-V-4.5.
           smoke and complete both require every row to succeed (batch size 1); parse failures
           are reported and fail the gate only when more than half of the rows are unparsed.
           Sanity rules (added 2026-11-14, each fails the gate): all parsed labels identical (from 20 parsed rows on);
           median A-D probability mass (sum of exp(label_logprobs)) < 0.5; visual_token_count outside 576..640
           (GATE_VISUAL_RANGE=lo,hi overrides). Fields that are not recorded are reported, never a failure.
           Reported only: accuracy (smoke accuracy below 0.30 adds a warning), median s/inference and the ETA of the full run
           (eta_full_seconds, used by the queue to stop a step that cannot finish within its timeout).

smoke    : stop unless the 3-item run is complete, parses, batches are as planned, and the
           re-run O/R answers agree with the earlier run of the same model
complete : stop unless all 5,688 main rows succeeded; report anchor agreement
Also reported (not a gate): answer agreement of T and R on pairs where the T question equals
the R question (same input in the same run; differences would be batch-level numerical noise).
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

REL = Path(__file__).resolve().parent.parent
RULES = {
    "qwen3": {"reference": "rstem_ropts_inputs/reference_qwen3_20261107_OR.jsonl", "smoke_min_anchor_agreement": 0.90},
    "internvl35": {"reference": "rstem_ropts_inputs/reference_internvl35_20261109_OR.jsonl", "smoke_min_anchor_agreement": 0.97},
}
GENERIC = {"max_unparsed_fraction": 0.5, "complete_rows": 5688}
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

COMMON = {"smoke_max_parse_failures": 2, "complete_rows": 5688, "complete_max_parse_failures": 20}


def load(out: Path) -> list[dict]:
    rows = {}
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["input_id"]] = row
    return list(rows.values())


def anchor_agreement(rows: list[dict], ref_path: Path) -> dict:
    ref = {}
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ref[(r["pair_id"], r["condition"])] = r
    same = total = 0
    by_condition = Counter()
    for row in rows:
        if row["condition"] not in ("O", "R") or row.get("status") != "success":
            continue
        old = ref[(row["pair_id"], row["condition"])]
        if row["rendered_prompt"].find(old["question"]) < 0:
            raise ValueError(f"Anchor question text differs from the earlier run: {row['input_id']}")
        total += 1
        agree = row["parsed_generated_label"] == old["parsed_generated_label"]
        same += agree
        by_condition[(row["condition"], agree)] += 1
    return {"anchor_rows": total, "label_agreement": round(same / total, 4) if total else None,
            "by_condition": {f"{c}/{'same' if a else 'diff'}": n for (c, a), n in sorted(by_condition.items())}}


def t_equals_r_agreement(rows: list[dict]) -> dict:
    by_pair = {}
    for r in rows:
        if r.get("status") == "success":
            by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r
    n = same = 0
    for c in by_pair.values():
        if "T" in c and "R" in c and c["T"]["rendered_prompt"] == c["R"]["rendered_prompt"]:
            n += 1
            same += c["T"]["parsed_generated_label"] == c["R"]["parsed_generated_label"]
    return {"pairs_T_equals_R": n, "same_answer": same}


def software_vs_earlier(out: Path, model: str) -> dict:
    """Report (not a gate) package versions of this run against the earlier run contract of the model."""
    contract = {"qwen3": "rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json",
                "internvl35": "rstem_ropts_inputs/run_contract_internvl35_rel_20261109.json"}[model]
    old = json.loads((REL / contract).read_text(encoding="utf-8"))["software_versions"]
    now = json.loads((out / "run_contract.json").read_text(encoding="utf-8"))["software_versions"]
    diff = {k: [old.get(k), now.get(k)] for k in sorted(set(old) | set(now)) if k != "platform" and old.get(k) != now.get(k)}
    return {"differences_except_platform": diff}


def main_generic(mode: str, out: Path) -> None:
    rows = load(out)
    statuses = Counter(r.get("status") for r in rows)
    success = [r for r in rows if r.get("status") == "success"]
    parse_failures = sum(1 for r in success if not r["parse_success"])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    problems = []
    if not summary.get("complete"):
        problems.append(f"incomplete: {summary.get('statuses')}")
    multi = sum(1 for n in Counter(r["batch_id"] for r in success).values() if n != 1)
    if multi:
        problems.append(f"{multi} batches hold more than one input (expected batch size 1)")
    if mode == "complete" and len(rows) != GENERIC["complete_rows"]:
        problems.append(f"rows {len(rows)} != {GENERIC['complete_rows']}")
    if mode not in ("smoke", "complete"):
        raise SystemExit("Use smoke or complete")
    fraction = parse_failures / len(rows) if rows else 1.0
    if fraction > GENERIC["max_unparsed_fraction"]:
        problems.append(f"unparsed fraction {fraction:.3f} > {GENERIC['max_unparsed_fraction']}")
    sane_problems, warnings, sane_info = sanity_checks(success, "minicpm_v45", mode)
    problems += sane_problems
    ms = [r["inference_runtime_ms"] for r in success if isinstance(r.get("inference_runtime_ms"), (int, float))]
    median_s = statistics.median(ms) / 1000 if ms else None
    t_r = t_equals_r_agreement(rows)
    result = {"model": "generic", "rules": GENERIC, "sanity": sane_info, "warnings": warnings,
              "median_s_per_inference": round(median_s, 3) if median_s is not None else None,
              "eta_full_seconds": round(median_s * GENERIC["complete_rows"], 2) if median_s is not None else None, "rows": len(rows), "statuses": dict(statuses),
              "parse_failures": parse_failures, "unparsed_fraction": round(fraction, 4),
              "accuracy": summary.get("accuracy"), "T_equals_R_same_input_check": t_r,
              "visual_token_count_example": next((r.get("visual_token_count") for r in success), None),
              "problems": problems, "status": "passed" if not problems else "failed"}
    (out / f"gate_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    if problems:
        sys.exit(1)


def main() -> None:
    model, mode, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if model == "generic":
        return main_generic(mode, out)
    rule = RULES[model]
    rows = load(out)
    statuses = Counter(r.get("status") for r in rows)
    parse_failures = sum(1 for r in rows if r.get("status") == "success" and not r["parse_success"])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    agreement = anchor_agreement(rows, REL / rule["reference"])
    problems = []
    batches = {}
    for r in rows:
        if r.get("status") == "success":
            batches.setdefault(r["batch_id"], []).append((r["pair_id"], r["condition"]))
    if model == "qwen3":
        misaligned = sum(1 for b in batches.values()
                         if len({p for p, _ in b}) != 1 or sorted(c for _, c in b) != ["O", "R", "T"])
        if misaligned:
            problems.append(f"{misaligned} batches are not exactly one pair's O, T, R")
    else:
        misaligned = sum(1 for b in batches.values() if len(b) != 1)
        if misaligned:
            problems.append(f"{misaligned} batches hold more than one input (expected batch size 1)")
    if not summary.get("complete"):
        problems.append(f"incomplete: {summary.get('statuses')}")
    if mode == "smoke":
        if parse_failures > COMMON["smoke_max_parse_failures"]:
            problems.append(f"parse failures {parse_failures}")
        if (agreement["label_agreement"] or 0) < rule["smoke_min_anchor_agreement"]:
            problems.append(f"anchor agreement {agreement['label_agreement']} below threshold")
    elif mode == "complete":
        if len(rows) != COMMON["complete_rows"]:
            problems.append(f"rows {len(rows)} != {COMMON['complete_rows']}")
        if parse_failures > COMMON["complete_max_parse_failures"]:
            problems.append(f"parse failures {parse_failures}")
    else:
        raise SystemExit("Use smoke or complete")
    result = {"model": model, "rules": {**rule, **COMMON}, "rows": len(rows), "statuses": dict(statuses),
              "parse_failures": parse_failures, "misaligned_batches": misaligned,
              "accuracy": summary.get("accuracy"), "anchor_vs_earlier_run": agreement,
              "T_equals_R_same_input_check": t_equals_r_agreement(rows),
              "software_versions_vs_earlier_run": software_vs_earlier(out, model),
              "note_anchor_below_0_99": (agreement["label_agreement"] or 0) < 0.99,
              "problems": problems, "status": "passed" if not problems else "failed"}
    (out / f"gate_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
