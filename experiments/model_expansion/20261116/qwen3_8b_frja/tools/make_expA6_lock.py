"""Writes addon/followup_rel_6lang/q3_frja/expA6_inputs/expA6_lock.json (sha256 of the three Exp A input files, the frozen REL conditions file, expected counts).
Local only. Usage: python3 make_expA6_lock.py <addon dir> <expA_translation_6lang_20261115 dir>"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

addon, src = Path(sys.argv[1]), Path(sys.argv[2])
inp = addon / "followup_rel_6lang/q3_frja/expA6_inputs"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l]
files = {n: sha(inp / n) for n in ("OR_gemma_conditions.jsonl", "T_gemma_conditions.jsonl", "T_gemma_retry_conditions.jsonl")}
T, R = rd(inp / "T_gemma_conditions.jsonl"), {r["pair_id"]: r for r in rd(inp / "T_gemma_retry_conditions.jsonl")}
changed = [t["pair_id"] for t in T if t["question"] != R[t["pair_id"]]["question"]]
lock = {
    "schema": "rel-expA-T-6lang-inputs-v1",
    "created": "2026-11-16",
    "files": files,
    "frozen_conditions_sha256": sha(addon / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl"),
    "expected_pairs": dict(Counter(t["split"] for t in T)),
    "expected_retry_changed_pairs": len(changed),
    "retry_changed_pairs_by_split": dict(Counter(next(t["split"] for t in T if t["pair_id"] == p) for p in changed)),
    "require_new_language": True,
    "source_folder": "expA_translation_6lang_20261115 (README_KO.md); SHA256SUMS of that folder = " + sha(src / "SHA256SUMS"),
    "source_files_in_that_folder": {"work/" + n: v for n, v in files.items()},
    "translator": {"model_id": "mlx-community/gemma-3-12b-it-4bit", "revision": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558", "decoding": "greedy"},
    "note": "O and R rows are the O/R rows of conditions_6lang.jsonl for the same pair (the runner re-checks this); T = T_gemma (main) or T_gemma_retry (retry variant, pairs whose question changed only).",
}
(inp / "expA6_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(lock, ensure_ascii=False, indent=1))
