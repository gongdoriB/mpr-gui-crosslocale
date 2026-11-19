"""Builds inputs/method6_conditions.jsonl + inputs/method6_lock.json from the editor outputs (run_edit6.py) and the O rows of the R_stem/R_opts input file.

  python build_method6_conditions.py <model> <edit_out_dir> [--pairs-expected 4740]   (self-edit: the editor output must be the same model's)

M_S = rewrite_V-D-S.jsonl, M_H = rewrite_V-D-H.jsonl. Pair metadata (split, languages, image path/sha256, gold answer) is copied from the O row of
source/rstem6_conditions.jsonl (sha256 checked against source/rstem6_lock.json); the editor never saw the gold answer. Each rewrite row must
carry the same pair_id, source/gui language and image sha256 as that O row, and options A-D. question = stem + " A: .. B: .. C: .. D: ..".
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = {"M_S": "V-D-S", "M_H": "V-D-H"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def render(stem, options):
    return stem + " " + " ".join(f"{k}: {options[k]}" for k in "ABCD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("edit_out", type=Path)
    ap.add_argument("--pairs-expected", type=int, default=4740)
    a = ap.parse_args()
    src_lock = json.loads((HERE / "source/rstem6_lock.json").read_text())
    if sha(HERE / "source/rstem6_conditions.jsonl") != src_lock["file_sha256"]:
        raise ValueError("source/rstem6_conditions.jsonl differs from its lock")
    O = {r["pair_id"]: r for r in read(HERE / "source/rstem6_conditions.jsonl") if r["condition"] == "O"}
    if not (a.edit_out / "COMPLETE").exists():
        raise ValueError("editor output is not COMPLETE")
    contract = json.loads((a.edit_out / "contract.json").read_text())
    if contract.get("model") != a.model or contract.get("mock"):
        raise ValueError(f"editor output belongs to {contract.get('model')} (mock={contract.get('mock')}), not a real run of {a.model} (self-edit)")
    rows, stats, files = [], Counter(), {}
    per = {}
    for cond, cell in CELLS.items():
        f = a.edit_out / f"rewrite_{cell}.jsonl"
        files[f.name] = sha(f)
        rw = read(f)
        ids = [r["pair_id"] for r in rw]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate pair in " + f.name)
        per[cond] = {r["pair_id"]: r for r in rw}
        for r in rw:
            stats[f"{cond}/{r['status']}"] += 1
    pids = sorted(per["M_S"])
    if set(per["M_H"]) != set(pids) or len(pids) != a.pairs_expected or not set(pids) <= set(O):
        raise ValueError(f"pair sets differ or count {len(pids)} != {a.pairs_expected}")
    for pid in pids:
        o = O[pid]
        for cond in CELLS:
            r = per[cond][pid]
            if (r["source_language"], r["gui_language"], r["image_sha256"]) != (o["query_language"], o["gui_language"], o["image_sha256"]):
                raise ValueError(f"{cond} metadata differs from O: {pid}")
            if list(r["options"]) != list("ABCD") or any(not isinstance(v, str) or not v.strip() for v in r["options"].values()):
                raise ValueError(f"{cond} options malformed: {pid}")
            q = render(r["question_stem"], r["options"])
            if q != r["modified_question"]:
                raise ValueError(f"{cond} modified_question differs from stem+options: {pid}")
            stats[f"{cond}/equals_O"] += q == o["question"]
            rows.append({k: o[k] for k in ("pair_id", "parallel_id", "split", "query_language", "gui_language", "image_path", "image_sha256", "gold_answer")}
                        | {"condition": cond, "question_stem": r["question_stem"], "options": r["options"], "question": q})
    out = HERE / "inputs" / a.model; out.mkdir(parents=True, exist_ok=True)
    p = out / "method6_conditions.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lock = {"schema": "method6-inputs-v1", "model": a.model, "editor": contract["editor"], "created": datetime.now(timezone.utc).isoformat(), "pairs": len(pids), "rows": len(rows),
            "conditions": dict(Counter(r["condition"] for r in rows)), "file_sha256": sha(p),
            "edit_contract_sha256": sha(a.edit_out / "contract.json"), "rewrite_files_sha256": files,
            "source_rstem6_sha256": src_lock["file_sha256"], "stats": dict(sorted(stats.items()))}
    (out / "method6_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(lock, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
