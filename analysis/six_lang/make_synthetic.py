#!/usr/bin/env python3
"""Generate SYNTHETIC (mock) prediction files from the 6-language INPUT conditions, fixed seed. Not real model output.
Every outcome is a deterministic hash-based random draw (no real result file is read). Rows are order independent, so the
old 4-language rows of the 'full 6-language' mock are identical to the 'old' mock rows.
Usage: make_synthetic.py <out_dir>
"""
import hashlib, json, sys
from pathlib import Path

BASE = Path("${PROJECT_ROOT}")
SALT = "synthetic-20261116"
REL_COND = BASE / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl"
EXPA_OLD = BASE / "expA_translation_20261112/work"
EXPA_NEW = BASE / "expA_translation_6lang_20261115/work"
NEW = ("fr", "ja")


def u(*parts):
    h = hashlib.sha256((SALT + "|" + "|".join(map(str, parts))).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_new(q, g):
    return q in NEW or g in NEW


def hw(new):
    return {"gpus": [{"name": "SYNTHETIC-RTX4090" if new else "SYNTHETIC-B200"}]}


def rel_row(r, tag):
    new = tag == "new"
    item = u("item", r["parallel_id"]) - 0.5
    p = 0.55 + 0.25 * item
    p += {"O": 0.0, "R": 0.10, "C": 0.03, "F": 0.14}[r["condition"]]
    if r["gui_language"] == "th":
        p -= 0.10
    if r["gui_language"] == "ja":
        p -= 0.03
    iid = f"rel_orcf::{r['pair_id']}::{r['condition']}"
    ok = u("rel", iid) < min(max(p, 0.05), 0.95)
    parse_ok = u("parse", iid) > 0.01
    gen = bool(ok and parse_ok)
    sc = gen if u("sc", iid) > 0.03 else (not gen)
    cross = r["query_language"] != r["gui_language"]
    d = {k: r[k] for k in ("pair_id", "parallel_id", "state_key", "query_language", "gui_language", "matched", "gold_answer", "condition", "context_language", "reference_language", "referent_alignment", "context_localization", "primary_analysis",
                           "source_issue", "eligible_without_source_adjudication", "text_dependency", "has_reference_spans", "reference_edit_changes_text",
                           "equivalent_to_condition")}
    d.update(input_id=iid, status="success", experiment_fingerprint="SYNTH-" + ("new" if new else "old"), hardware=hw(new),
             batch_size=4, code_commit="synthetic", parse_success=parse_ok, generation_correct=gen, correct=gen, scoring_correct=bool(sc),
             parsed_generated_label="A" if gen else "B", scored_predicted_label="A" if sc else "B", prompt_token_count=700, output_token_count=2,
             visual_token_count=576, reused_prediction=False, primary_analysis_eligible=bool(cross and r["eligible_without_source_adjudication"]),
             synthetic=True)
    return d


def expa_row(r, cond):
    item = u("eitem", r["parallel_id"]) - 0.5
    p = 0.60 + 0.25 * item + {"O": 0.0, "T_gemma": -0.02, "T_gemma_retry": -0.01, "R": 0.12}[cond]
    if r["gui_language"] == "th":
        p -= 0.05
    cid = {"T_gemma": "T", "T_gemma_retry": "T_gemma_retry"}.get(cond, cond)
    iid = f"rel_expA::{r['pair_id']}::{cid}"
    ok = u("expa", iid if cond != "T_gemma_retry" else iid.replace("T_gemma_retry", "T"), cond == "T_gemma_retry") < min(max(p, 0.05), 0.95)
    parse_ok = u("eparse", iid) > 0.01
    gen = bool(ok and parse_ok)
    sc = gen if u("esc", iid) > 0.03 else (not gen)
    new = is_new(r["query_language"], r["gui_language"])
    return dict(input_id=iid, experiment_fingerprint="SYNTH-" + ("new" if new else "old"), hardware=hw(new), batch_size=3, code_commit="synthetic",
                pair_id=r["pair_id"], parallel_id=r["parallel_id"], split=r["split"], condition=cid, query_language=r["query_language"],
                gui_language=r["gui_language"], gold_answer=r["gold_answer"], parse_success=parse_ok, generation_correct=gen, correct=gen,
                scoring_correct=bool(sc), parsed_generated_label="A" if gen else "B", scored_predicted_label="A" if sc else "B",
                prompt_token_count=700, visual_token_count=576, status="success", synthetic=True)


def w(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rel = jl(REL_COND)
    old = [rel_row(r, "old") for r in rel if not is_new(r["query_language"], r["gui_language"])]
    new = [rel_row(r, "new") for r in rel if is_new(r["query_language"], r["gui_language"])]
    w(out / "rel_old.jsonl", old); w(out / "rel_new.jsonl", new); w(out / "rel_full6.jsonl", old + new)
    # Exp A: pairs from the T conditions (O/T/R rows per pair; pilot rows included as in real files)
    to, tn = jl(EXPA_OLD / "T_gemma_conditions.jsonl"), jl(EXPA_NEW / "T_gemma_conditions.jsonl")
    eo = [expa_row(r, c) for r in to for c in ("O", "T_gemma", "R")]
    en = [expa_row(r, c) for r in tn for c in ("O", "T_gemma", "R")]
    w(out / "expa_old.jsonl", eo); w(out / "expa_new.jsonl", en); w(out / "expa_full6.jsonl", eo + en)
    ro, rn = jl(EXPA_OLD / "T_gemma_retry_conditions.jsonl"), jl(EXPA_NEW / "T_gemma_retry_conditions.jsonl")
    w(out / "expa_retry.jsonl", [expa_row(r, "T_gemma_retry") for r in ro + rn])
    print("rows:", dict(rel_old=len(old), rel_new=len(new), expa_old=len(eo), expa_new=len(en), retry=len(ro) + len(rn)))


if __name__ == "__main__":
    main()
