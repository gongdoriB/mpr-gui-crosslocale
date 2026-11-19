"""Selects the stratified subset of the EARLIER MiniCPM-V-4.5 4-language REL inferences (canonical ids) that is re-run on the new server (vast4090c).
Local only, deterministic (seed 20261116). Same procedure as model_expansion_20261116/qwen3_8b_frja/tools/consistency_select.py (which it copies, only the input
reader differs: the earlier MiniCPM predictions are a plain .jsonl, not .gz): strata = (condition, query language, GUI language) cells that hold at least one canonical
inference in the earlier run (52 cells), floor(N / cells) per cell, remainder to cells drawn with random.Random(seed) (cells with spare capacity only), ids inside a cell drawn
with the same RNG from the sorted ids. Writes consistency_subset_ids.json and consistency_reference.jsonl (the earlier answers of exactly these ids).
Usage: python3 -I consistency_select_minicpm.py <earlier predictions.jsonl> <earlier run_contract.json> <outdir> [N=600]"""
import hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261116
pred_path, contract_path, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 600
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = {}
with open(pred_path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rows[r["input_id"]] = r          # last record per id
ok = {k: r for k, r in rows.items() if r.get("status") == "success"}
cells = defaultdict(list)
for k in sorted(ok):
    parts = k.split("::")
    assert parts[0] == "rel_orcf", k
    cells[(parts[-1], parts[-3], parts[-2])].append(k)
keys = sorted(cells)
base, extra = divmod(N, len(keys))
alloc = {c: min(base, len(cells[c])) for c in keys}
rng = random.Random(SEED)
spare = [c for c in keys if alloc[c] < len(cells[c])]
for c in rng.sample(spare, extra):
    alloc[c] += 1
chosen = []
for c in keys:
    chosen += rng.sample(cells[c], alloc[c])
chosen.sort()
assert len(chosen) == N and len(set(chosen)) == N
outdir.mkdir(parents=True, exist_ok=True)
contract = json.loads(contract_path.read_text())
meta = {
    "schema": "rel-consistency-subset-v1", "seed": SEED, "n": N,
    "source_predictions": "model_results_20261116/minicpm/unpacked/minicpm_rel/full/predictions.jsonl", "source_predictions_sha256": sha(pred_path),
    "source_run_contract_sha256": sha(contract_path), "source_batch_size": contract["batch_size"],
    "source_experiment_fingerprint": contract["experiment_fingerprint"],
    "cells": len(keys), "per_cell_base": base, "per_cell_allocation": {"|".join(c): alloc[c] for c in keys},
    "by_condition": dict(Counter(i.split("::")[-1] for i in chosen)),
    "by_language_pair": dict(Counter("->".join(i.split("::")[-3:-1]) for i in chosen)),
    "ids": chosen,
}
(outdir / "consistency_subset_ids.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
keep = ("input_id", "parsed_generated_label", "scored_predicted_label", "generation_correct", "parse_success", "gold_probability", "label_probabilities",
        "label_logprobs", "batch_size", "inference_runtime_ms", "prompt_token_count", "visual_token_count")
with (outdir / "consistency_reference.jsonl").open("w", encoding="utf-8") as g:
    for i in chosen:
        r = ok[i]
        g.write(json.dumps({**{k: r.get(k) for k in keep}, "gpu": (r.get("hardware") or {}).get("gpus", [{}])[0].get("name"),
                            "software_versions": r.get("software_versions"), "rendered_prompt": r["rendered_prompt"]}, ensure_ascii=False) + "\n")
print(json.dumps({k: meta[k] for k in ("n", "cells", "per_cell_base", "by_condition")}, indent=1), len(set(i.split("::")[-3] + i.split("::")[-2] for i in chosen)), "language pairs")
