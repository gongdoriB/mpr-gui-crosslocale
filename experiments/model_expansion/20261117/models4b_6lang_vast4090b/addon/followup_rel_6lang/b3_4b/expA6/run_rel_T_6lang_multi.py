"""Experiment A (O, T, R), 6-language version, for Qwen3-VL-4B / InternVL3.5-4B (vast4090b queue_b3_4b; the MODELS table is the only change of the 8B-queue copy).

Copy of q3_frja/expA6/run_rel_T_6lang.py (Qwen3 build, sha f2b097b0...) with the changes listed in CHANGES_VS_QWEN3_6LANG on top of CHANGES_VS_4LANG. The earlier 4-language runners were
run_rel_T.py (internvl35: batch 1, strict false) and run_rel_T_generic.py (internvl25_8b, uitars_15_7b: batch 1, strict true); prompts, parsing and scoring come from the same repo functions.

Copy of followup_rel_4lang/expA/run_rel_T.py (sha d232a13d...) with the changes listed in CHANGES_VS_4LANG. Model loading, prompting, batching, parsing, label scoring and the
resumable prediction cache are imported unchanged from mpr_crosslocale.inference.run_rel_followup (the module of the earlier Qwen3-VL-8B runs, sha 75fee94f...), exactly as the
4-language runner did. Nothing in the repository is modified.

    O  source-language reference names           (anchor, same question as the REL O row)
    T  names machine-translated into the GUI language without the screenshot (variant main: T_gemma_conditions.jsonl)
       or the same with the retried untranslated names (variant retry: T_gemma_retry_conditions.jsonl)
    R  names as written in the GUI-language version of the question (anchor, same question as the REL R row)
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

HERE = Path(__file__).resolve().parent
Q3 = HERE.parent                      # followup_rel_6lang/b3_4b
DATA6 = Q3.parent                     # followup_rel_6lang
REPO = DATA6.parent                   # repository root (the project root that image paths are relative to)
sys.path.insert(0, str(REPO / "01_code/src"))

from mpr_crosslocale.data.rel_followup import write_json  # noqa: E402  (unchanged 4-language module of the repo)
from mpr_crosslocale.inference.run_rel_followup import (  # noqa: E402
    append_prediction,
    code_identity,
    predict_chunk,
    read_cache,
    stable_hash,
)
from mpr_crosslocale.inference.runtime import git_commit, load_yaml, software_versions  # noqa: E402
from mpr_crosslocale.inference.system_prompts import build_system_prompt  # noqa: E402

CHANGES_VS_QWEN3_6LANG = """
M1 --model {qwen3_vl_4b, internvl35_4b} replaces the fixed qwen3: the reference contract (model config, system prompt, template, batch size, strict flag) is the run_contract.json of that model's earlier
   4-language Exp A run (contracts/ref_*_expA_*.json). strict_batch_size as in that run (both 4B models: true, run_rel_T_generic.py); batch size 1.
M2 --scope new (default; fr/ja pairs only, expA6_inputs) | all6 (every main pair of the 6 languages: the 1,896 earlier 4-language main pairs with the T rows of the earlier run (T_conditions.jsonl sha 0ebbed0f...)
   plus the 2,844 new main pairs; expA6_all6_inputs, built by tools/make_expA6_all6_inputs.py). For all6 the 4-language part has the same input ids and question texts as the earlier run.
M3 --split default main (the 4-language run used main only); the all6 inputs hold main pairs only.
M4 the schema is still rel-expA-T-6lang-v1; the identity additionally holds model and scope.
"""

CHANGES_VS_4LANG = """
E1 inputs: O and R rows come from OR_gemma_conditions.jsonl, T from T_gemma_conditions.jsonl (variant main) or T_gemma_retry_conditions.jsonl (variant retry), all listed with sha256
   in expA6_lock.json. The 4-language file rstem_ropts_inputs/mixed_conditions.jsonl is not used.
E2 frozen anchors: every O and R row must equal the O/R row of the same pair in conditions_6lang.jsonl (sha locked): question, gold answer, image path (up to the folder prefix) and
   image sha256. The 4-language runner compared T against the frozen REL rows in the same way.
E3 image paths are relative to the repository root (followup_rel_6lang/assets/... for fr/ja screens, followup_rel_4lang/assets/... for the others); data_root is the repository root.
   The 4-language runner stripped the followup_rel_4lang/ prefix and used that folder as data_root. The model gets the same file bytes (sha256 checked against the row).
E4 model families: only qwen3. The batch size is a parameter (1 or 3; 3 = one pair per batch = O, T, R adjacent, as before). The 4-language runner demanded 3.
E5 --split default all (3,060 pairs = 2,844 main + 216 pilot); the 4-language run used main only. Rows keep their split field.
E6 variant retry: only pairs whose T_retry question differs from the T question (lock: count), conditions O, T_retry, R, so every batch keeps the shape of the main run.
E7 identity (experiment fingerprint) holds schema rel-expA-T-6lang-v1, variant, all input sha256 and this runner's sha; input_id is unchanged (rel_expA::<pair_id>::<condition>).
E8 new --inputs-dir / --frozen-conditions / --lock options (defaults = the shipped files) so that local tests can feed the 4-language rows through this runner.
Unchanged: render(), the question check (question == stem + options), the condition queue order, summarize(), the resume rule, the record written per inference.
"""

EXPECTED_FROZEN_SHA256 = "63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed"   # conditions_6lang.jsonl (REL 6-language data lock)
NEW_LANGUAGES = ("fr", "ja")
VARIANTS = {
    "main": {"conditions": ("O", "T", "R"), "t_file": "T_gemma_conditions.jsonl", "t_condition_in_file": "T_gemma"},
    "retry": {"conditions": ("O", "T_retry", "R"), "t_file": "T_gemma_retry_conditions.jsonl", "t_condition_in_file": "T_gemma_retry"},
}
DEFAULT_INPUTS = Q3 / "expA6_inputs"
DEFAULT_INPUTS_ALL6 = Q3 / "expA6_all6_inputs"
MODELS = {   # strict = strict_batch_size of the earlier 4-language Exp A run of the same model (run_contract.json 2026-11-14); batches = allowed batch sizes
    "qwen3_vl_4b": {"strict": True, "batches": (1,), "contract": "ref_qwen3_vl_4b_expA_20261114.json"},
    "internvl35_4b": {"strict": True, "batches": (1,), "contract": "ref_internvl35_4b_expA_20261114.json"},
}
DEFAULT_FROZEN = DATA6 / "data/rel_all_366/conditions_6lang.jsonl"
OR_FILE = "OR_gemma_conditions.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(stem: str, options: dict) -> str:
    return stem + " " + " ".join(f"{k}: {options[k]}" for k in "ABCD")


def input_id(pair_id: str, condition: str) -> str:
    return f"rel_expA::{pair_id}::{condition}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_rows(inputs_dir: Path, lock_path: Path, frozen_path: Path, data_root: Path, variant: str, split: str,
              max_items: int | None, check_images: bool):
    spec = VARIANTS[variant]
    conditions = spec["conditions"]
    t_cond = conditions[1]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    t_file = inputs_dir / spec["t_file"]
    for name in (OR_FILE, spec["t_file"]):
        if sha256_file(inputs_dir / name) != lock["files"][name]:
            raise ValueError(f"{name} differs from the lock {lock_path.name}")
    if sha256_file(frozen_path) != lock["frozen_conditions_sha256"]:
        raise ValueError("Frozen REL conditions file differs from the lock")
    or_rows = read_jsonl(inputs_dir / OR_FILE)
    t_rows = read_jsonl(t_file)
    if dict(Counter(r["split"] for r in t_rows)) != lock["expected_pairs"] or len({r["pair_id"] for r in t_rows}) != len(t_rows):
        raise ValueError("Unexpected T pair counts or duplicate pair ids")
    if {r["condition"] for r in t_rows} != {spec["t_condition_in_file"]}:
        raise ValueError(f"T file must hold condition {spec['t_condition_in_file']} only")
    by_or = {(r["pair_id"], r["condition"]): r for r in or_rows}
    if len(by_or) != len(or_rows) or {c for _, c in by_or} != {"O", "R"} or {p for p, _ in by_or} != {r["pair_id"] for r in t_rows}:
        raise ValueError("O/R rows do not cover exactly the T pairs, or hold duplicates")
    if lock.get("require_new_language", True):
        for r in t_rows:
            if r["query_language"] == r["gui_language"] or not ({r["query_language"], r["gui_language"]} & set(NEW_LANGUAGES)):
                raise ValueError(f"Pair is not a mismatched pair involving fr or ja: {r['pair_id']}")
    frozen = {}
    for r in read_jsonl(frozen_path):
        if r["condition"] in ("O", "R"):
            frozen[(r["pair_id"], r["condition"])] = r

    unchanged_pairs = set()
    if variant == "retry":
        main_t = {r["pair_id"]: r for r in read_jsonl(inputs_dir / VARIANTS["main"]["t_file"])}
        if sha256_file(inputs_dir / VARIANTS["main"]["t_file"]) != lock["files"][VARIANTS["main"]["t_file"]]:
            raise ValueError("T_gemma_conditions.jsonl differs from the lock")
        unchanged_pairs = {r["pair_id"] for r in t_rows if main_t[r["pair_id"]]["question"] == r["question"]}

    sha_cache: dict[str, str] = {}
    pairs = {}
    for t in t_rows:
        pid = t["pair_id"]
        if variant == "retry" and pid in unchanged_pairs:
            continue
        if split != "all" and t["split"] != split:
            continue
        o, r = by_or[(pid, "O")], by_or[(pid, "R")]
        fo, fr = frozen[(pid, "O")], frozen[(pid, "R")]
        image = t["image_path"]
        if not (image == o["image_path"] == r["image_path"]) or not (t["image_sha256"] == o["image_sha256"] == r["image_sha256"]):
            raise ValueError(f"Image differs between O/T/R: {pid}")
        for mine, froz in ((o, fo), (r, fr)):
            if (mine["question"], mine["gold_answer"], mine["image_path"].split("assets/", 1)[1], mine["parallel_id"], mine["query_language"], mine["gui_language"]) != \
                    (froz["question"], froz["gold_answer"], froz["image_path"].split("assets/", 1)[1], froz["parallel_id"], froz["query_language"], froz["gui_language"]):
                raise ValueError(f"{mine['condition']} row differs from the frozen REL row: {pid}")
        if not (t["gold_answer"] == o["gold_answer"] == r["gold_answer"]):
            raise ValueError(f"Gold differs: {pid}")
        if (t["parallel_id"], t["query_language"], t["gui_language"], t["split"]) != (o["parallel_id"], o["query_language"], o["gui_language"], o["split"]):
            raise ValueError(f"Pair metadata differs: {pid}")
        for x in (t, o, r):
            if set(x["options"]) != set("ABCD") or x["question"] != render(x["question_stem"], x["options"]):
                raise ValueError(f"{x['condition']} question is not stem+options: {pid}")
        if check_images:
            if image not in sha_cache:
                sha_cache[image] = sha256_file(data_root / image)
            if sha_cache[image] != t["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {image}")
        base = {"pair_id": pid, "parallel_id": t["parallel_id"], "split": t["split"],
                "query_language": t["query_language"], "gui_language": t["gui_language"],
                "image_path": image, "gold_answer": t["gold_answer"],
                "text_dependency": fo["text_dependency"],
                "source_issue": t.get("source_issue"),
                "eligible_without_source_adjudication": t.get("eligible_without_source_adjudication")}
        pairs[pid] = {"O": {**base, "condition": "O", "question": o["question"]},
                      t_cond: {**base, "condition": t_cond, "question": t["question"]},
                      "R": {**base, "condition": "R", "question": r["question"]}}

    pair_ids = sorted(pairs)
    if max_items is not None:
        keep = sorted({pairs[p]["O"]["parallel_id"] for p in pair_ids})[:max_items]
        pair_ids = [p for p in pair_ids if pairs[p]["O"]["parallel_id"] in keep]
    rows = [pairs[pid][c] for pid in pair_ids for c in conditions]
    if variant == "retry" and split == "all" and max_items is None and len(pair_ids) != lock["expected_retry_changed_pairs"]:
        raise ValueError(f"retry variant: {len(pair_ids)} changed pairs, lock says {lock['expected_retry_changed_pairs']}")
    return rows, {"conditions_sha256": lock["frozen_conditions_sha256"], "retry_unchanged_pairs": len(unchanged_pairs)}


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


def run(args) -> dict:
    spec = VARIANTS[args.variant]
    conditions = spec["conditions"]
    spec_m = MODELS[args.model]
    if args.batch_size not in spec_m["batches"]:
        raise ValueError(f"Use batch size {spec_m['batches']} for {args.model}")
    data_root = args.data_root.resolve()
    inputs_dir = (args.inputs_dir or (DEFAULT_INPUTS_ALL6 if args.scope == "all6" else DEFAULT_INPUTS)).resolve()
    lock_path = (args.lock or inputs_dir / ("expA6_all6_lock.json" if args.scope == "all6" else "expA6_lock.json")).resolve()
    frozen_path = args.frozen_conditions.resolve()
    rows, audit = load_rows(inputs_dir, lock_path, frozen_path, data_root, args.variant, args.split, args.max_items,
                            check_images=not (args.dry_run or args.mock_model))
    config = load_yaml(args.model_config)
    reference = json.loads((args.reference_contract or Q3 / "contracts" / spec_m["contract"]).read_text(encoding="utf-8"))
    if config != reference["model_config"]:
        raise ValueError(f"Model config differs from the earlier {args.model} Exp A run contract")
    system_prompt, _, template = build_system_prompt("english_fixed", "en")
    if system_prompt != reference["system_prompt"] or template != reference["prompt_template_version"]:
        raise ValueError(f"System prompt differs from the earlier {args.model} Exp A run contract")
    mode = "mock" if args.mock_model else "gpu"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    identity = {
        "schema": "rel-expA-T-6lang-v1",
        "model": args.model,
        "scope": args.scope,
        "variant": args.variant,
        "conditions": list(conditions),
        "dataset_sha256": audit["conditions_sha256"],
        "inputs_sha256": {k: lock["files"][k] for k in sorted(lock["files"])},
        "expA6_lock_sha256": sha256_file(lock_path),
        "model_config": config, "execution_mode": mode,
        "system_prompt": system_prompt, "prompt_template_version": template,
        "score_labels": True, "batch_size": args.batch_size,
        "strict_batch_size": spec_m["strict"],
        "queue_order": "pair_id sorted; O, T, R adjacent" if args.variant == "main" else "pair_id sorted; O, T_retry, R adjacent",
        "split": args.split, "max_items": args.max_items,
        "selected_input_ids_sha256": stable_hash([input_id(r["pair_id"], r["condition"]) for r in rows]),
        "code_sha256": {**code_identity(), "run_rel_T_6lang.py": sha256_file(Path(__file__))},
        "software_versions": software_versions(),
    }
    fingerprint = stable_hash(identity)
    t_cond = conditions[1]
    plan = {"model": args.model, "scope": args.scope, "variant": args.variant, "pairs": len(rows) // 3, "items": len({r["parallel_id"] for r in rows}),
            "inferences": len(rows), "batch_size": args.batch_size,
            "by_condition": dict(Counter(r["condition"] for r in rows)),
            "by_split": dict(Counter(r["split"] for i, r in enumerate(rows) if i % 3 == 0)),
            "T_equals_O_pairs": sum(1 for i in range(0, len(rows), 3) if rows[i + 1]["question"] == rows[i]["question"]),
            "T_equals_R_pairs": sum(1 for i in range(0, len(rows), 3) if rows[i + 1]["question"] == rows[i + 2]["question"]),
            "T_condition": t_cond, "retry_unchanged_pairs_skipped": audit["retry_unchanged_pairs"],
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
    # Keep the three conditions of a pair together across resumes.
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
                                 data_root=data_root, strict_batch_size=spec_m["strict"])
    versions, commit, failures = software_versions(), git_commit(), 0
    from tqdm import tqdm
    try:
        progress = tqdm(total=len(pending), desc=f"REL expA6 {args.variant} {args.model} ({mode}, batch={args.batch_size})")
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
                          "gold_answer": row["gold_answer"], **result}
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
    parser.add_argument("--scope", choices=("new", "all6"), default="new")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="main")
    parser.add_argument("--data-root", type=Path, default=REPO, help="repository root; image_path of every row is relative to it")
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--frozen-conditions", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--reference-contract", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("main", "pilot", "all"), default="main")
    parser.add_argument("--max-items", type=int, help="Keep only the first N items (smoke test)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.mock_model:
        parser.error("Choose --dry-run or --mock-model")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.dry_run and not result.get("complete"):
        raise SystemExit("Incomplete inference: rerun the same command with --resume")


if __name__ == "__main__":
    main()
