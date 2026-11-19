"""Exp A, 6-language extension: list new directed pairs (at least one of fr/ja, q != g)
for the same 170 items (158 main + 12 pilot) as the 4-language Exp A, and their name spans.
Inputs are the 6-language REL files (followup_rel_6lang), whose 4-language part is byte-identical
in content to the 4-language files used on the server."""
import hashlib, json
from collections import Counter
from pathlib import Path

BASE = Path("${PROJECT_ROOT}")
OLD_MIXED = BASE / "rstem_ropts_qwen3_20261109/addon/followup_rel_4lang/rstem_ropts_inputs/mixed_conditions.jsonl"
DATA6 = BASE / "followup_rel_6lang/data/rel_all_366"
OUT = Path(__file__).resolve().parent / "work"
FIELDS = ("question_stem", "A", "B", "C", "D")
NEW = ("fr", "ja")
LANGS = ("en", "zh", "th", "ru", "fr", "ja")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

split = {}
for l in OLD_MIXED.open(encoding="utf-8"):
    r = json.loads(l); split[r["parallel_id"]] = r["split"]
maps = {json.loads(l)["parallel_id"]: json.loads(l) for l in (DATA6 / "reference_mappings.jsonl").open(encoding="utf-8")}
O = {}
for l in (DATA6 / "conditions_6lang.jsonl").open(encoding="utf-8"):
    r = json.loads(l)
    if r["condition"] == "O": O[(r["parallel_id"], r["query_language"], r["gui_language"])] = r

pairs, inventory, skipped = [], [], []
for pid in sorted(split):
    excl = set(maps[pid].get("excluded_locales") or [])
    for q in LANGS:
        for g in LANGS:
            if q == g or not ({q, g} & set(NEW)): continue
            if {q, g} & excl:
                skipped.append([pid, q, g]); continue
            o = O[(pid, q, g)]
            pair_id = f"{pid}::{q}::{g}"
            pairs.append({"pair_id": pair_id, "parallel_id": pid, "split": split[pid], "query_language": q,
                          "gui_language": g, "image_path": o["image_path"], "gold_answer": o["gold_answer"],
                          "source_issue": o["source_issue"], "eligible_without_source_adjudication": o["eligible_without_source_adjudication"]})
            sp = maps[pid]["spans"]
            for f in FIELDS:
                src, tgt = sp[q][f], sp[g][f]
                tb = {s["ref_id"]: s["text"] for s in tgt}
                assert {s["ref_id"] for s in src} == set(tb), (pair_id, f)
                for s in src:
                    inventory.append({"pair_id": pair_id, "parallel_id": pid, "split": split[pid], "query_language": q,
                                      "gui_language": g, "field": f, "ref_id": s["ref_id"], "source_text": s["text"],
                                      "screen_text": tb[s["ref_id"]]})
OUT.mkdir(exist_ok=True)
for name, rows in [("pairs_new.jsonl", pairs), ("span_inventory.jsonl", inventory)]:
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
groups = {(r["parallel_id"], r["query_language"], r["gui_language"]) for r in inventory}
report = {"items": len(split), "pairs_new": len(pairs), "pairs_by_split": dict(Counter(p["split"] for p in pairs)),
          "skipped_excluded_locale": skipped, "span_uses": len(inventory), "translation_groups": len(groups),
          "pairs_without_spans": len({p["pair_id"] for p in pairs} - {r["pair_id"] for r in inventory}),
          "inputs": {"mixed_conditions_4lang_sha256": sha(OLD_MIXED), "reference_mappings_6lang_sha256": sha(DATA6 / "reference_mappings.jsonl"),
                     "conditions_6lang_sha256": sha(DATA6 / "conditions_6lang.jsonl")}}
(OUT / "inputs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "skipped_excluded_locale"}, ensure_ascii=False, indent=1), "skipped", len(skipped))
