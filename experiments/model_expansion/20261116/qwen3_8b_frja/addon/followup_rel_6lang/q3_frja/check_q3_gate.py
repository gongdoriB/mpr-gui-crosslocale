"""Gate for the Qwen3-VL-8B fr/ja REL O/R/C/F run (new_only) and for the consistency subset run.
Copy of check_rel6_gate.py (InternVL2.5 REL6 addon, sha 'check_rel6_gate.py' there) with these changes only:
  Q1 --batch N   batch size every success row must have (default 1). The InternVL rule "batch size 1" is replaced by this.
  Q2 family qwen3_vl: the Qwen adapter records no visual_token_count, so prompt_token_count must be 800..2700 (earlier 4-language run: 1,711..2,515);
     every rendered prompt must hold the english_fixed system sentence and the Qwen chat markers, and no default InternVL system message.
  Q3 no hard-coded full count: --full-n must be given for the ETA (new_only = 26,057); kind 'cc' (consistency subset) is checked like 'rel'.
  Q4 timing also reports the median of inference_runtime_ms per batch size.
Unchanged: success rows == expected, parse failure <= 50 %, label sanity rules (identical labels, A-D mass), ETA fields, per-language report.

  python check_q3_gate.py rel|cc smoke|complete <out_dir> --model qwen3_vl_8b --batch N --full-n N [--full-rows N] [expected]
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
FULL_INFERENCES = {"rel": None, "cc": None, "expa": None}   # Q3: given by --full-n

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
    # InternVL2.5-8B, max_num 7 + thumbnail, 256 tokens per tile: the REL run of 2026-10-05 had exactly 768 (3 tiles, 13,984 rows) and
    # 1,792 (7 tiles, 4,448 rows); prompt tokens 858..2,145.
    "internvl25": {"visual": (768, 1792), "prompt": None},
}
TOKEN_RULES["qwen3_vl"] = {"visual": (0, 2048), "prompt": (800, 2700)}   # Q2 (same as above, kept explicit)
QWEN_MARKERS = ("<|im_start|>system", "<|vision_start|><|image_pad|><|vision_end|>")
MOCK_RENDER = False   # --mock (local tests): the mock model renders system prompt + question without the chat template, so the chat markers are not required
SYSTEM_SENTENCE = "You are a GUI reasoning assistant."
DEFAULT_CHINESE_MARK = "书生"


def family_of(key: str | None) -> str | None:
    key = key or ""
    for prefix, fam in (("qwen3_vl", "qwen3_vl"), ("uitars", "uitars"), ("internvl35", "internvl35"), ("internvl25", "internvl25"), ("minicpm", "minicpm")):
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
    if fam == "internvl25":
        rendered = [r.get("rendered_prompt") for r in ok if isinstance(r.get("rendered_prompt"), str)]
        info["rendered_prompt_checked"] = len(rendered)
        if not rendered:
            info["system_prompt_rule"] = "not evaluated: rendered_prompt not recorded"
        else:
            bad = [p for p in rendered if SYSTEM_SENTENCE not in p or DEFAULT_CHINESE_MARK in p]
            if bad:
                problems.append(f"{len(bad)}/{len(rendered)} rendered prompts lack the english_fixed system prompt or contain the default Chinese system message")
    if fam == "qwen3_vl":   # Q2
        rendered = [r.get("rendered_prompt") for r in ok if isinstance(r.get("rendered_prompt"), str)]
        info["rendered_prompt_checked"] = len(rendered)
        if not rendered:
            info["system_prompt_rule"] = "not evaluated: rendered_prompt not recorded"
        else:
            bad = [p for p in rendered if SYSTEM_SENTENCE not in p or DEFAULT_CHINESE_MARK in p or (not MOCK_RENDER and not all(m in p for m in QWEN_MARKERS))]
            if bad:
                problems.append(f"{len(bad)}/{len(rendered)} rendered prompts lack the english_fixed system prompt or the Qwen chat markers")
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
    by_bs = {}
    for r in ok:
        if isinstance(r.get("inference_runtime_ms"), (int, float)):
            by_bs.setdefault(str(r.get("batch_size")), []).append(r["inference_runtime_ms"] / 1000)
    return {"median_s_per_inference": round(med, 3), "median_s_by_batch_size": {k: round(statistics.median(v), 3) for k, v in sorted(by_bs.items())},
            "full_inferences": full_n,
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
    global MOCK_RENDER
    if "--mock" in argv:
        argv.remove("--mock")
        MOCK_RENDER = True
    batch = 1
    if "--batch" in argv:   # Q1
        i = argv.index("--batch")
        batch = int(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) < 3 or argv[0] not in ("rel", "cc") or argv[1] not in ("smoke", "complete"):
        raise SystemExit(__doc__)
    kind, mode, out = argv[0], argv[1], Path(argv[2])
    full_rows = None
    if "--full-n" in argv:
        i = argv.index("--full-n")
        FULL_INFERENCES[kind] = int(argv[i + 1])
        del argv[i:i + 2]
    if FULL_INFERENCES[kind] is None:
        raise SystemExit("--full-n is required")
    if "--full-rows" in argv:
        i = argv.index("--full-rows")
        full_rows = int(argv[i + 1])
        del argv[i:i + 2]
    expected_arg = int(argv[3]) if len(argv) > 3 else None
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    planned = plan["actual_inferences"]
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
    # Q1: every batch has `batch` rows, except the tail batch of the run (planned count % batch) and, after an interrupted run, the rest of one cut batch.
    # Allowed: sizes below `batch` for at most 2 x batch rows; sizes above `batch` never.
    over = {k: v for k, v in batch_sizes.items() if k is None or k > batch}
    under_rows = sum(v for k, v in batch_sizes.items() if k is not None and k < batch)
    if over or under_rows > 2 * batch or None in batch_sizes:
        problems.append(f"batch sizes {dict(batch_sizes)} do not fit batch size {batch} (at most {2 * batch} rows may sit in smaller tail batches)")
    rate = parse_failures / len(ok) if ok else 1.0
    if rate > MAX_PARSE_FAIL_RATE:
        problems.append(f"parse failure rate {rate:.3f} > {MAX_PARSE_FAIL_RATE}")
    sane_problems, warnings, sane_info = sanity_checks(ok, key, mode)
    problems += sane_problems
    if full_rows is not None and mode == "complete":
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if plan.get("condition_rows") != full_rows or summary.get("condition_rows") != full_rows:
            problems.append(f"condition rows: plan {plan.get('condition_rows')}, summary {summary.get('condition_rows')}, expected {full_rows}")
        if summary.get("statuses") != {"success": full_rows}:
            problems.append(f"summary statuses {summary.get('statuses')} != all {full_rows} success")
    by_lang = {"query": {}, "gui": {}}
    for r in ok:
        parts = r["input_id"].split("::")
        for side, lang in (("query", parts[-3]), ("gui", parts[-2])):
            c = by_lang[side].setdefault(lang, [0, 0, 0])
            c[0] += 1
            c[1] += bool(r["generation_correct"])
            c[2] += not r["parse_success"]
    by_cond = {}
    for r in ok:
        c = by_cond.setdefault(r.get("condition") or r["input_id"].rsplit("::", 1)[-1], [0, 0, 0])
        c[0] += 1
        c[1] += bool(r["generation_correct"])
        c[2] += not r["parse_success"]
    result = {"kind": kind, "mode": mode, "expected_rows": expected, "rows": len(rows), "statuses": dict(statuses),
              "batch_sizes": dict(batch_sizes), "expected_batch_size": batch, "rows_in_smaller_tail_batches": under_rows, "parse_failures": parse_failures, "parse_failure_rate": round(rate, 4),
              "by_condition(n,correct,unparsed)": by_cond,
              "note": "REL rows are deduplicated inferences (conditions with identical prompts share one row)"
              if kind in ("rel", "cc") else "",
              "by_language(n,correct,unparsed)": by_lang,
              "model_key": key, "sanity": sane_info, "warnings": warnings, "timing": eta_info(ok, FULL_INFERENCES[kind]),
              "problems": problems, "status": "passed" if not problems else "failed"}
    (out / f"gate_{kind}_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
