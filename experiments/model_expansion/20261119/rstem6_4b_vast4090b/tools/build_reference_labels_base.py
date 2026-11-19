"""Builds reference/*.jsonl.gz of the R_stem/R_opts 6-language addon from earlier local result files (run once on the Mac; the gz files are part of the addon).
  rstem4lang_<model>_labels.jsonl.gz : all four conditions of the earlier 4-language R_stem/R_opts runs (qwen3_vl_8b 2026-11-09, internvl35_8b 2026-11-10)
  rel_OR_<model>_labels.jsonl.gz     : O and R rows of the earlier REL runs of the same model (4-language and fr/ja new_only), same questions (checked against rstem6_conditions.jsonl)
Each record: pair_id, condition, label (parsed_generated_label), scored (scored_predicted_label), correct (generation_correct), src (source file tag).
Run: python3 -I build_reference_labels.py <project root> <rstem6 inputs file> <out dir>"""
import gzip, json, sys
from pathlib import Path
root, inputs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
new = {}
for l in open(inputs, encoding="utf-8"):
    r = json.loads(l); new[(r["pair_id"], r["condition"])] = r["question"]
def rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if d.get("status") != "success":
                continue
            yield d
def key_of(d):
    p = d["input_id"].split("::", 1)[1]
    pair, cond = p.rsplit("::", 1)
    return pair, cond
SRC = {
 "qwen3_vl_8b": {"rstem": "rstem_ropts_qwen3_20261109/results_20261109/raw_from_github_324654c/full/predictions.jsonl.gz",
                 "rel": ["model_results_20261116/qwen3_8b_frja_rel/predictions.jsonl"]},
 "minicpm_v45": {"rel": ["model_results_20261116/minicpm/unpacked/minicpm_rel/full/predictions.jsonl",
                         "model_results_20261117/minicpm_6lang_rel/unpacked/followup_rel_6lang/results/rel6new_minicpm_v45/full/predictions.jsonl"]},
 "internvl35_8b": {"rstem": "rstem_ropts_internvl35_20261110/results_20261110/extracted/internvl35_rstem_ropts/full/predictions.jsonl",
                   "rel": ["model_results_20261116/internvl35_expA6/unpacked/followup_rel_6lang/results/rel6new_internvl35_8b/full/predictions.jsonl"]},
 "uitars_15_7b": {"rel": ["model_results_20261114/uitars_15_7b/unpacked/sizes_uitars_15_7b/rel_full/predictions.jsonl",
                          "model_results_20261116/uitars_expA6/unpacked/followup_rel_6lang/results/rel6new_uitars_15_7b/full/predictions.jsonl"]},
 "internvl25_8b": {"rel": ["model_results_20261116/internvl25_rel6/unpacked/rel6_internvl25_8b/full/predictions.jsonl"]},
}
def dump(path, recs):
    recs.sort(key=lambda r: (r["pair_id"], r["condition"]))
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for r in recs:
            f.write((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
for model, spec in SRC.items():
    if "rstem" in spec:
        recs, bad = [], 0
        for d in rows(root / spec["rstem"]):
            pair, cond = d["pair_id"], d["condition"]
            q = new.get((pair, cond))
            if q is None or q not in d["rendered_prompt"]:
                bad += 1; continue
            recs.append({"pair_id": pair, "condition": cond, "label": d["parsed_generated_label"], "scored": d["scored_predicted_label"], "correct": bool(d["generation_correct"]), "src": "rstem4lang"})
        assert bad == 0 and len(recs) == 7584, (model, bad, len(recs))
        dump(out / f"rstem4lang_{model}_labels.jsonl.gz", recs)
        print(model, "rstem4lang", len(recs))
    recs, seen, bad = [], set(), 0
    for src in spec["rel"]:
        n0 = len(recs)
        for d in rows(root / src):
            pair, cond = key_of(d)
            if cond not in ("O", "R"):
                continue
            q = new.get((pair, cond))
            if q is None:
                continue                      # pair not part of the 158 main items
            if q not in d["rendered_prompt"]:
                bad += 1; continue
            if (pair, cond) in seen:
                continue
            seen.add((pair, cond))
            recs.append({"pair_id": pair, "condition": cond, "label": d["parsed_generated_label"], "scored": d["scored_predicted_label"], "correct": bool(d["generation_correct"]), "src": Path(src).parts[1] if len(Path(src).parts) > 1 else src})
        print(model, "rel_OR from", src.split("/")[-4:-1], len(recs) - n0)
    assert bad == 0, (model, "question text differs in", bad)
    dump(out / f"rel_OR_{model}_labels.jsonl.gz", recs)
    print(model, "rel_OR total", len(recs), "of", 2 * 4740, "O/R rows of rstem6")
