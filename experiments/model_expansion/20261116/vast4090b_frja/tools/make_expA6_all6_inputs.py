"""Builds addon/followup_rel_6lang/b2_frja/expA6_all6_inputs/ (Exp A inputs for ALL main pairs of the 6 languages: 1,896 earlier 4-language pairs + 2,844 new fr/ja pairs).
  old part : T = T_conditions.jsonl of the earlier 4-language Exp A run (sha 0ebbed0f..., rows are 'T_gemma_first'); retry = expA_translation_20261112/work/T_gemma_retry_conditions.jsonl (sha 5865950e...);
             O/R = the O/R rows of conditions_6lang.jsonl (= the frozen 4-language REL rows; question, gold, options unchanged), image paths prefixed with followup_rel_4lang/ ... as in the old T rows.
  new part : the 6-language Gemma files (expA_translation_6lang_20261115/work), main split only.
Condition labels are normalised to T_gemma / T_gemma_retry (the runner's file format); question texts and ids are untouched. Pilot pairs are not included (earlier run: main only).
Usage: make_expA6_all6_inputs.py <out dir> ; local only."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
R = Path("${PROJECT_ROOT}")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OLD_T = R / "expA_translation_20261112/addon/followup_rel_4lang/expA/expA_inputs/T_conditions.jsonl"
OLD_RETRY = R / "expA_translation_20261112/work/T_gemma_retry_conditions.jsonl"
NEW = R / "expA_translation_6lang_20261115/work"
FROZEN = R / "model_expansion_20261116/qwen3_8b_frja/addon/followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl"
frozen = {(r["pair_id"], r["condition"]): r for r in rd(FROZEN) if r["condition"] in ("O", "R")}
oldT = [r for r in rd(OLD_T) if r["split"] == "main"]
oldR = {r["pair_id"]: r for r in rd(OLD_RETRY)}
assert len(oldT) == 1896 and all(r["pair_id"] in oldR for r in oldT)
newT = [r for r in rd(NEW / "T_gemma_conditions.jsonl") if r["split"] == "main"]
newR = {r["pair_id"]: r for r in rd(NEW / "T_gemma_retry_conditions.jsonl")}
newOR = [r for r in rd(NEW / "OR_gemma_conditions.jsonl") if r["split"] == "main"]
assert len(newT) == 2844 and len(newOR) == 2 * 2844
T, TR, OR = [], [], []
for t in oldT:
    pid = t["pair_id"]
    T.append({**t, "condition": "T_gemma"})
    TR.append({**oldR[pid], "condition": "T_gemma_retry"})
    for c in ("O", "R"):
        f = frozen[(pid, c)]
        assert f["gold_answer"] == t["gold_answer"] and (f["query_language"], f["gui_language"]) == (t["query_language"], t["gui_language"])
        OR.append({"pair_id": pid, "parallel_id": f["parallel_id"], "split": t["split"], "query_language": f["query_language"], "gui_language": f["gui_language"], "condition": c,
                   "image_path": t["image_path"], "image_sha256": t["image_sha256"], "gold_answer": f["gold_answer"], "question_stem": f["question_stem"], "options": f["options"], "question": f["question"]})
for t in newT:
    T.append(t); TR.append(newR[t["pair_id"]])
OR += newOR
w = lambda n, rows: (out / n).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
w("T_gemma_conditions.jsonl", T); w("T_gemma_retry_conditions.jsonl", TR); w("OR_gemma_conditions.jsonl", OR)
changed = [t["pair_id"] for t, r in zip(T, TR) if t["question"] != r["question"]]
files = {n: sha(out / n) for n in ("OR_gemma_conditions.jsonl", "T_gemma_conditions.jsonl", "T_gemma_retry_conditions.jsonl")}
lock = {"schema": "rel-expA-T-6lang-inputs-v1", "scope": "all6", "created": "2026-11-16", "files": files, "frozen_conditions_sha256": sha(FROZEN),
        "expected_pairs": {"main": 4740}, "expected_retry_changed_pairs": len(changed), "require_new_language": False,
        "sources": {"old_T_conditions.jsonl": sha(OLD_T), "old_retry_conditions.jsonl": sha(OLD_RETRY), "new_T_gemma_conditions.jsonl": sha(NEW / "T_gemma_conditions.jsonl"),
                    "new_T_gemma_retry_conditions.jsonl": sha(NEW / "T_gemma_retry_conditions.jsonl"), "new_OR_gemma_conditions.jsonl": sha(NEW / "OR_gemma_conditions.jsonl")},
        "retry_changed_pairs_old_main": sum(1 for p in changed if any(t["pair_id"] == p for t in oldT)), "retry_changed_pairs_new_main": 0,
        "note": "old part: T of the earlier 4-language Exp A run (first Gemma translation), O/R from the frozen REL rows; new part: Gemma 6-language files, main split."}
lock["retry_changed_pairs_new_main"] = len(changed) - lock["retry_changed_pairs_old_main"]
(out / "expA6_all6_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: lock[k] for k in ("expected_pairs", "expected_retry_changed_pairs", "retry_changed_pairs_old_main", "retry_changed_pairs_new_main")}), Counter(r["condition"] for r in OR))
