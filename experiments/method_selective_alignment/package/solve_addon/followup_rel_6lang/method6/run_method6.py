"""Method 6-language solve run (M_S = method V-D-S: stem GUI names aligned, options original; M_H = method V-D-H: stem names aligned + options translated)
for one model on ONE RTX 4090. Adapted from run_rstem6.py (R_stem/R_opts 6-language runner); model loading, prompting, batching, parsing, label scoring
and the resumable cache are imported unchanged from mpr_crosslocale.inference.run_rel_followup. O / R_stem / R_opts / R of the same pairs come from the
R_stem/R_opts run of the same model (same harness, config, contract), so only M_S and M_H are inferred here.

Input: inputs/<model>/method6_conditions.jsonl (4,740 pairs x 2 = 9,480 rows; built by build_method6_conditions.py from THIS model's own editor outputs = self-edit), sha256 in inputs/<model>/method6_lock.json.
Checks before any inference (all abort):
  - input sha256 == lock; frozen REL file (conditions_6lang.jsonl) sha256 == locked value
  - every pair has exactly M_S and M_H once; same image (path, sha256), gold, parallel id, languages, split in both
  - parallel id, languages, gold answer and image (after assets/) equal the frozen REL O row of the pair
  - question == stem + " " + "A: .. B: .. C: .. D: .." for every row
  - every image file exists and has the row's sha256 (not in --dry-run / --mock-model)
Queue order: pairs sorted by pair_id, M_S then M_H. input_id: rel_method6::<pair_id>::<condition>.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent            # followup_rel_6lang/method6
DATA6 = HERE.parent                               # followup_rel_6lang
REPO = DATA6.parent                               # repository root (image paths of the rows are relative to it)
sys.path.insert(0, str(REPO / "01_code/src"))

from mpr_crosslocale.data.rel_followup import write_json  # noqa: E402
from mpr_crosslocale.inference.run_rel_followup import (  # noqa: E402
    append_prediction,
    code_identity,
    predict_chunk,
    read_cache,
    stable_hash,
)
from mpr_crosslocale.inference.runtime import git_commit, load_yaml, software_versions  # noqa: E402
from mpr_crosslocale.inference.system_prompts import build_system_prompt  # noqa: E402

CONDITIONS = ("M_S", "M_H")
EXPECTED_FROZEN_SHA256 = "63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed"   # conditions_6lang.jsonl (REL 6-language data lock)
LOCK_SCHEMA = "method6-inputs-v1"
EXPECTED_PAIRS = 4740
DEFAULT_INPUT = None   # self-edit: per model, inputs/<model>/method6_conditions.jsonl
DEFAULT_LOCK = None
DEFAULT_FROZEN = DATA6 / "data/rel_all_366/conditions_6lang.jsonl"
# model key -> reference contract (run_contract.json of the earlier run of that model). batch size and strict_batch_size are taken from the contract.
MODELS = {
    "qwen3_vl_8b": "ref_qwen3_rel_20261107.json",      # 2026-11-07 REL (the 2026-11-09 R_stem/R_opts run used the same code and config): batch 4, strict
    "minicpm_v45": "ref_minicpm_rel_20261116.json",    # batch 1, strict false
    "internvl35_8b": "ref_internvl35_rel_20261109.json",  # batch 1, strict false
    "uitars_15_7b": "ref_uitars_rel_20261114.json",    # batch 1, strict true
    "internvl25_8b": "ref_internvl25_expA_20261114.json",  # batch 1, strict true (no earlier REL run of this model; its Exp A run)
    "qwen3_vl_4b": "ref_qwen3_4b_expA_20261117.json",      # 2026-11-19 addition: 4B Exp A 6-language run on vast4090b (same repo code as this runner; the REL run hashed the 6lang runner instead), batch 1, strict true
    "internvl35_4b": "ref_internvl35_4b_expA_20261117.json",  # 2026-11-19 addition: same, batch 1, strict true
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(stem: str, options: dict) -> str:
    return stem + " " + " ".join(f"{k}: {options[k]}" for k in "ABCD")


def input_id(pair_id: str, condition: str) -> str:
    return f"rel_method6::{pair_id}::{condition}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def after_assets(p: str) -> str:
    return p.split("assets/", 1)[1]


def load_rows(inputs: Path, lock_path: Path, frozen_path: Path, data_root: Path, max_items: int | None, check_images: bool,
              or_source: Path | None = None):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("method6_lock.json has an unexpected schema")
    if sha256_file(inputs) != lock["file_sha256"]:
        raise ValueError(f"{inputs.name} differs from the lock {lock_path.name}")
    if sha256_file(frozen_path) != EXPECTED_FROZEN_SHA256:
        raise ValueError("Frozen REL conditions file (conditions_6lang.jsonl) differs from the locked sha256")
    rows_in = read_jsonl(inputs)
    if len(rows_in) != lock["rows"] or dict(Counter(r["condition"] for r in rows_in)) != lock["conditions"]:
        raise ValueError("Row or condition counts differ from the lock")
    by = defaultdict(dict)
    for r in rows_in:
        if r["condition"] in by[r["pair_id"]]:
            raise ValueError(f"Duplicate row {r['pair_id']} {r['condition']}")
        by[r["pair_id"]][r["condition"]] = r
    if len(by) != lock["pairs"] or any(set(c) != set(CONDITIONS) for c in by.values()):
        raise ValueError("Pairs do not hold exactly M_S and M_H each, or the pair count differs from the lock")
    frozen = {}
    for r in read_jsonl(frozen_path):
        if r["condition"] in ("O", "R"):
            frozen[(r["pair_id"], r["condition"])] = r
    src = None
    if or_source is not None:
        if sha256_file(or_source) != lock["source_sha256"]:
            raise ValueError("--or-source differs from the lock's source_sha256")
        src = {(r["pair_id"], r["condition"]): r for r in read_jsonl(or_source) if r["condition"] in ("O", "R")}

    def same_as(mine, froz, what):
        # frozen REL rows (conditions_6lang.jsonl) have no split / image sha256 and keep the image path relative to assets/; OR_gemma rows have all of them
        keys = ["question", "question_stem", "options", "gold_answer", "parallel_id", "query_language", "gui_language"]
        keys += [k for k in ("split",) if k in froz]
        for k in keys:
            if mine[k] != froz[k]:
                raise ValueError(f"{mine['condition']} row differs from the {what} row in '{k}': {mine['pair_id']}")
        if after_assets(mine["image_path"]) != after_assets(froz["image_path"]):
            raise ValueError(f"{mine['condition']} image differs from the {what} row: {mine['pair_id']}")
        if "image_sha256" in froz and mine["image_sha256"] != froz["image_sha256"]:
            raise ValueError(f"{mine['condition']} image sha256 differs from the {what} row: {mine['pair_id']}")

    sha_cache: dict[str, str] = {}
    pairs, stats = {}, Counter()
    for pid in sorted(by):
        c = by[pid]
        o = c["M_S"]
        fo = frozen[(pid, "O")]
        for k in ("parallel_id", "query_language", "gui_language", "gold_answer"):
            if o[k] != fo[k]:
                raise ValueError(f"M_S row differs from the frozen REL O row in '{k}': {pid}")
        if after_assets(o["image_path"]) != after_assets(fo["image_path"]):
            raise ValueError(f"M_S image differs from the frozen REL O row: {pid}")
        for x in c.values():
            for k in ("parallel_id", "split", "query_language", "gui_language", "image_path", "image_sha256", "gold_answer"):
                if x[k] != o[k]:
                    raise ValueError(f"Field {k} differs between the conditions of {pid}")
            if set(x["options"]) != set("ABCD") or x["question"] != render(x["question_stem"], x["options"]):
                raise ValueError(f"{x['condition']} question is not stem+options: {pid}")
        stats["M_S_equals_O"] += c["M_S"]["question"] == fo["question"]
        stats["M_H_equals_O"] += c["M_H"]["question"] == fo["question"]
        stats["M_H_equals_M_S"] += c["M_H"]["question"] == c["M_S"]["question"]
        if check_images:
            image = o["image_path"]
            if image not in sha_cache:
                sha_cache[image] = sha256_file(data_root / image)
            if sha_cache[image] != o["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {image}")
        pairs[pid] = {cn: {"pair_id": pid, "parallel_id": o["parallel_id"], "split": o["split"],
                           "query_language": o["query_language"], "gui_language": o["gui_language"],
                           "image_path": o["image_path"], "gold_answer": o["gold_answer"],
                           "condition": cn, "question": c[cn]["question"]} for cn in CONDITIONS}
    pair_ids = sorted(pairs)
    if max_items is not None:
        keep = sorted({pairs[p]["M_S"]["parallel_id"] for p in pair_ids})[:max_items]
        pair_ids = [p for p in pair_ids if pairs[p]["M_S"]["parallel_id"] in keep]
    rows = [pairs[pid][cn] for pid in pair_ids for cn in CONDITIONS]
    return rows, {"input_sha256": lock["file_sha256"], "frozen_sha256": EXPECTED_FROZEN_SHA256, "equal_question_pairs": dict(stats),
                  "images_checked": len(sha_cache), "or_source_checked": src is not None}


def summarize(rows: list[dict], cache: dict) -> dict:
    statuses = Counter()
    acc = defaultdict(lambda: [0, 0])
    parse_fail = 0
    for row in rows:
        pred = cache.get(input_id(row["pair_id"], row["condition"]))
        status = pred["status"] if pred else "pending"
        statuses[status] += 1
        if status == "success":
            parse_fail += not pred["parse_success"]
            cell = acc[(row["split"], row["condition"])]
            cell[0] += bool(pred["generation_correct"])
            cell[1] += 1
    return {
        "rows": len(rows), "statuses": dict(statuses), "parse_failures": parse_fail,
        "accuracy": {f"{s}/{c}": {"correct": a, "n": n, "rate": round(a / n, 4) if n else None}
                     for (s, c), (a, n) in sorted(acc.items())},
        "complete": statuses.get("success", 0) == len(rows),
    }


def check_images_only(args) -> dict:
    """--check-images: validate every row against the lock, the frozen file and the image files (sha256 of all images), run nothing else."""
    rows, audit = load_rows(args.inputs.resolve(), args.lock.resolve(), args.frozen_conditions.resolve(), args.data_root.resolve(), None, True, None)
    return {"rows": len(rows), "images_checked": audit["images_checked"], "input_sha256": audit["input_sha256"], "frozen_sha256": audit["frozen_sha256"],
            "equal_question_pairs": audit["equal_question_pairs"], "status": "ok"}


def run(args) -> dict:
    reference = json.loads((args.reference_contract or HERE / "contracts" / MODELS[args.model]).read_text(encoding="utf-8"))
    batch, strict = reference["batch_size"], reference["strict_batch_size"]
    if args.batch_size is None:
        args.batch_size = batch
    if args.batch_size != batch:
        raise ValueError(f"Use batch size {batch} for {args.model} (as in its earlier run)")
    data_root = args.data_root.resolve()
    rows, audit = load_rows(args.inputs.resolve(), args.lock.resolve(), args.frozen_conditions.resolve(), data_root, args.max_items,
                            check_images=not (args.dry_run or args.mock_model), or_source=None)
    config = load_yaml(args.model_config)
    if config != reference["model_config"]:
        raise ValueError(f"Model config differs from the earlier {args.model} run contract")
    system_prompt, _, template = build_system_prompt("english_fixed", "en")
    if system_prompt != reference["system_prompt"] or template != reference["prompt_template_version"]:
        raise ValueError(f"System prompt differs from the earlier {args.model} run contract")
    mode = "mock" if args.mock_model else "gpu"
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("model") != args.model:
        raise ValueError(f"inputs were edited by {lock.get('model')}, not by {args.model} (self-edit)")
    identity = {
        "schema": "rel-method6-v1",
        "model": args.model,
        "conditions": list(CONDITIONS),
        "frozen_dataset_sha256": audit["frozen_sha256"],
        "input_sha256": audit["input_sha256"],
        "method6_lock_sha256": sha256_file(args.lock),
        "model_config": config, "execution_mode": mode,
        "system_prompt": system_prompt, "prompt_template_version": template,
        "score_labels": True, "batch_size": args.batch_size,
        "strict_batch_size": strict,
        "queue_order": "pair_id sorted; M_S, M_H adjacent",
        "max_items": args.max_items,
        "selected_input_ids_sha256": stable_hash([input_id(r["pair_id"], r["condition"]) for r in rows]),
        "code_sha256": {**code_identity(), "run_method6.py": sha256_file(Path(__file__))},
        "software_versions": software_versions(),
    }
    fingerprint = stable_hash(identity)
    plan = {"model": args.model, "pairs": len(rows) // 2, "items": len({r["parallel_id"] for r in rows}),
            "inferences": len(rows), "batch_size": args.batch_size, "strict_batch_size": strict,
            "by_condition": dict(Counter(r["condition"] for r in rows)),
            "by_language_pair": dict(sorted(Counter(f"{r['query_language']}>{r['gui_language']}" for r in rows if r["condition"] == "M_S").items())),
            "equal_question_pairs_all_input": audit["equal_question_pairs"], "images_checked": audit["images_checked"],
            "or_source_checked": audit["or_source_checked"], "lock_pairs": lock["pairs"],
            "experiment_fingerprint": fingerprint}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)
    if args.dry_run:
        return plan

    contract_path = args.output_dir / "run_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract["experiment_fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different input/model/settings")
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new output directory")
    else:
        if (args.output_dir / "predictions.jsonl").exists():
            raise ValueError("Prediction cache exists without its run contract")
        contract = {**identity, "experiment_fingerprint": fingerprint, "code_commit": git_commit(),
                    "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(contract_path, contract)

    cache_path = args.output_dir / "predictions.jsonl"
    cache = read_cache(cache_path, fingerprint)
    keys = {input_id(r["pair_id"], r["condition"]): r for r in rows}
    if not set(cache) <= set(keys):
        raise ValueError("Unexpected IDs in prediction cache")
    # Keep the two conditions of a pair together across resumes: a pair with any non-successful condition is re-run completely.
    unfinished = {r["pair_id"] for k, r in keys.items() if cache.get(k, {}).get("status") != "success"}
    pending = [(k, r) for k, r in keys.items() if r["pair_id"] in unfinished]

    adapter, hardware = None, {"execution_mode": mode}
    if pending and not args.mock_model:
        import torch
        from mpr_crosslocale.inference.run_cross_locale import _build_model_adapter
        from mpr_crosslocale.inference.runtime import hardware_metadata
        if not torch.cuda.is_available():
            raise RuntimeError("GPU inference requires CUDA; use --dry-run or --mock-model locally")
        hardware = hardware_metadata()
        adapter = _build_model_adapter(SimpleNamespace(attn_implementation=None, device_map="cuda:0"), config)

    chunk_args = SimpleNamespace(mock_model=args.mock_model, no_score_labels=False, seed=args.seed,
                                 data_root=data_root, strict_batch_size=strict)
    versions, commit, failures = software_versions(), git_commit(), 0
    from tqdm import tqdm
    try:
        progress = tqdm(total=len(pending), desc=f"REL method6 {args.model} ({mode}, batch={args.batch_size})")
        batches = (predict_chunk(adapter, pending[i:i + args.batch_size], chunk_args, config, system_prompt)
                   for i in range(0, len(pending), args.batch_size))
        for key, row, result in chain.from_iterable(batches):
            prediction = {"input_id": key, "experiment_fingerprint": fingerprint,
                          "attempt": cache.get(key, {}).get("attempt", 0) + 1,
                          "timestamp": datetime.now(timezone.utc).isoformat(),
                          "hardware": hardware, "software_versions": versions, "code_commit": commit,
                          "pair_id": row["pair_id"], "parallel_id": row["parallel_id"],
                          "split": row["split"], "condition": row["condition"],
                          "query_language": row["query_language"], "gui_language": row["gui_language"],
                          "gold_answer": row["gold_answer"], "question": row["question"], **result}
            failures = 0 if result["status"] == "success" else failures + 1
            append_prediction(cache_path, prediction)
            cache[key] = prediction
            progress.update(1)
            if failures >= 3:
                raise RuntimeError("Stopped after three consecutive failures; inspect predictions.jsonl")
    finally:
        if "progress" in locals():
            progress.close()
        summary = summarize(rows, cache)
        write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--data-root", type=Path, default=REPO, help="repository root; image_path of every row is relative to it")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--frozen-conditions", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--reference-contract", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-items", type=int, help="Keep only the first N items (smoke test); N items = N x 30 pairs x 2 conditions")
    parser.add_argument("--batch-size", type=int, default=None, help="default = batch size of the earlier run of this model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--check-images", action="store_true", help="only validate the inputs and the sha256 of every image file, then exit")
    args = parser.parse_args(argv)
    if args.dry_run and args.mock_model:
        parser.error("Choose --dry-run or --mock-model")
    if args.inputs is None:
        args.inputs = HERE / "inputs" / args.model / "method6_conditions.jsonl"
    if args.lock is None:
        args.lock = HERE / "inputs" / args.model / "method6_lock.json"
    if args.check_images:
        print(json.dumps(check_images_only(args), ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.dry_run and not result.get("complete"):
        raise SystemExit("Incomplete inference: rerun the same command with --resume")


if __name__ == "__main__":
    main()
