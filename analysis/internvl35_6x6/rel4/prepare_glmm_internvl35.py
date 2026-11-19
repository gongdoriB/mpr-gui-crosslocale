"""Validate the two completed REL runs and build compact GLMM inputs.

Only the source-issue-free, cross-language O/R/C/F quartets enter the primary
analysis table. Published gzip files are read in place and never replaced.
"""

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference")
BASE = ROOT / "followup_rel_4lang"
MODELS = {
    "internvl": ("internvl2_5_8b", "INTERNVL_RESULTS_MANIFEST.json"),
    "qwen": ("qwen2_5_vl_7b_batch32", "QWEN_RESULTS_MANIFEST.json"),
}
CONDITIONS = ("O", "R", "C", "F")
LANGUAGES = {"en", "zh", "th", "ru"}
ROW_FIELDS = (
    "model", "parallel_id", "pair_id", "query_language", "gui_language",
    "text_dependency", "condition", "reference_alignment",
    "context_localization", "generation_correct", "reused_prediction",
    "equivalent_to_condition", "reference_edit_changes_text",
    "has_reference_spans",
)
QUARTET_FIELDS = (
    "model", "parallel_id", "pair_id", "query_language", "gui_language",
    "text_dependency", "y_O", "y_R", "y_C", "y_F", "R_minus_O",
    "F_minus_C", "C_minus_O", "F_minus_R", "interaction",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_gzip(path, specification, parse_rows):
    require(path.is_file(), f"Missing result: {path}")
    require(path.stat().st_size == specification["bytes"], f"Packed byte count differs: {path}")
    require(sha256_file(path) == specification["sha256"], f"Packed SHA-256 differs: {path}")
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    rows = []
    count = 0
    with gzip.open(path, "rb") as source:
        for line in source:
            require(line.endswith(b"\n"), f"Unterminated JSONL row: {path}:{count + 1}")
            raw_hash.update(line)
            raw_bytes += len(line)
            count += 1
            if parse_rows:
                rows.append(json.loads(line))
    require(raw_bytes == specification["uncompressed_bytes"], f"Unpacked byte count differs: {path}")
    require(raw_hash.hexdigest() == specification["uncompressed_sha256"], f"Unpacked SHA-256 differs: {path}")
    require(count == specification["uncompressed_rows"], f"Row count differs: {path}")
    return rows, count


def source_metadata(model, directory, manifest_name):
    manifest_path = BASE / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "run_contract.json").read_text(encoding="utf-8"))
    require(summary["statuses"] == {"success": 23424}, f"Incomplete {model} run")
    require(summary["actual_successful_inferences"] == 18432, f"Unexpected inference count: {model}")
    require(audit["structural_checks_passed"] and audit["rows"] == 23424, f"Failed data audit: {model}")
    require(audit["conditions_sha256"] == contract["dataset_sha256"], f"Input fingerprint mismatch: {model}")
    require(manifest["completed_full_output"] == directory.relative_to(ROOT).as_posix(), f"Wrong manifest target: {model}")
    prefix = directory.relative_to(ROOT).as_posix()
    for filename in ("summary.json", "data_audit.json", "run_contract.json", "plan.json"):
        path = directory / filename
        spec = manifest["files"][f"{prefix}/{filename}"]
        # A Windows checkout may convert tracked JSON line endings to CRLF.
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        require(len(canonical) == spec["bytes"] and hashlib.sha256(canonical).hexdigest() == spec["sha256"], f"Metadata differs: {path}")
    return manifest, summary, audit


def project_row(model, row):
    require(row["status"] == "success", f"Non-success result: {model}, {row['pair_id']}")
    require(row["query_language"] in LANGUAGES and row["gui_language"] in LANGUAGES, "Unknown language")
    require(row["condition"] in CONDITIONS and row["text_dependency"] in ("dependent", "independent"), "Unknown condition/dependency")
    require(type(row["generation_correct"]) is bool, "Missing binary generation outcome")
    require(type(row["primary_analysis_eligible"]) is bool, "Missing eligibility flag")
    require(row["primary_analysis"] == (row["query_language"] != row["gui_language"]), "Wrong cross-language flag")
    require(row["primary_analysis_eligible"] == (row["primary_analysis"] and not row["source_issue"]), "Wrong source-issue exclusion")
    if not row["parse_success"]:
        require(not row["generation_correct"], "Parse failure counted as correct")
    projected = {
        "model": model,
        "parallel_id": row["parallel_id"],
        "pair_id": row["pair_id"],
        "query_language": row["query_language"],
        "gui_language": row["gui_language"],
        "text_dependency": row["text_dependency"],
        "condition": row["condition"],
        "reference_alignment": int(row["referent_alignment"]),
        "context_localization": int(row["context_localization"]),
        "generation_correct": int(row["generation_correct"]),
        "reused_prediction": int(row["reused_prediction"]),
        "equivalent_to_condition": row["equivalent_to_condition"],
        "reference_edit_changes_text": int(row["reference_edit_changes_text"]),
        "has_reference_spans": int(row["has_reference_spans"]),
    }
    require(
        (projected["reference_alignment"], projected["context_localization"])
        == {"O": (0, 0), "R": (1, 0), "C": (0, 1), "F": (1, 1)}[row["condition"]],
        f"Condition coding mismatch: {row['pair_id']} {row['condition']}",
    )
    return projected


def quartet_table(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["pair_id"])
        require(row["condition"] not in groups[key], f"Duplicate condition: {key}")
        groups[key][row["condition"]] = row
    output = []
    for key, group in groups.items():
        require(set(group) == set(CONDITIONS), f"Incomplete quartet: {key}")
        anchor = group["O"]
        for row in group.values():
            require(all(row[k] == anchor[k] for k in ("parallel_id", "query_language", "gui_language", "text_dependency")), f"Quartet metadata differs: {key}")
        y = {condition: group[condition]["generation_correct"] for condition in CONDITIONS}
        output.append({
            "model": key[0], "parallel_id": anchor["parallel_id"], "pair_id": key[1],
            "query_language": anchor["query_language"], "gui_language": anchor["gui_language"],
            "text_dependency": anchor["text_dependency"],
            **{f"y_{condition}": y[condition] for condition in CONDITIONS},
            "R_minus_O": y["R"] - y["O"], "F_minus_C": y["F"] - y["C"],
            "C_minus_O": y["C"] - y["O"], "F_minus_R": y["F"] - y["R"],
            "interaction": (y["F"] - y["C"]) - (y["R"] - y["O"]),
        })
    return sorted(output, key=lambda row: (row["model"], row["pair_id"]))


def validate_summary(quartets, summaries):
    for model, summary in summaries.items():
        for dependency, published in summary["primary_descriptive_summary"].items():
            subset = [row for row in quartets if row["model"] == model and row["text_dependency"] == dependency]
            require(len(subset) == published["complete_unflagged_mismatch_quartets"], f"Quartet count differs: {model}/{dependency}")
            for label, column in (("R-O", "R_minus_O"), ("F-C", "F_minus_C"), ("C-O", "C_minus_O"), ("F-R", "F_minus_R")):
                observed = 100 * sum(row[column] for row in subset) / len(subset)
                expected = published["paired_contrasts"][label]["difference_pp"]
                require(abs(observed - expected) < 1e-9, f"Published contrast differs: {model}/{dependency}/{label}")
            interaction = 100 * sum(row["interaction"] for row in subset) / len(subset)
            require(abs(interaction - published["interaction_pp"]) < 1e-9, f"Published interaction differs: {model}/{dependency}")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = Path("${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/rel4"); out.mkdir(parents=True, exist_ok=True)
    A = Path("${PROJECT_ROOT}/internvl35_prep/results_internvl35_20261109/full_rel")
    summary = json.loads((A / "summary.json").read_text())
    audit = json.loads((A / "data_audit.json").read_text())
    raw_rows = [json.loads(l) for l in open("${PROJECT_ROOT}/internvl35_prep/results_internvl35_20261109/full_rel/results.jsonl", encoding="utf-8")]
    print("rows", len(raw_rows), "statuses", summary.get("statuses"), "audit_ok", audit.get("structural_checks_passed"))
    require(len(raw_rows) == 23424, "rows")
    require(sum(not r["parse_success"] for r in raw_rows) == summary["parse_failures"], "parse failures differ")
    require(sum(bool(r["source_issue"]) for r in raw_rows) == summary["source_issue_rows"], "source issue differs")
    results = [project_row("internvl35", r) for r in raw_rows if r["primary_analysis_eligible"]]
    require(len(results) == 17168, "eligible rows %d" % len(results))
    quartets = quartet_table(results)
    require(len(quartets) == 4292, "quartets")
    validate_summary(quartets, {"internvl35": summary})
    results.sort(key=lambda r: (r["model"], r["pair_id"], CONDITIONS.index(r["condition"])))
    write_csv(out / "glmm_rows.csv", ROW_FIELDS, results)
    write_csv(out / "quartet_contrasts.csv", QUARTET_FIELDS, quartets)
    rep = {"eligible_rows": len(results), "eligible_quartets": len(quartets), "parse_failures": summary["parse_failures"], "source_issue_rows_excluded": summary["source_issue_rows"], "dependency_quartets": {d: sum(q["text_dependency"] == d for q in quartets) for d in ("dependent", "independent")}, "files": {"glmm_rows.csv": sha256_file(out / "glmm_rows.csv"), "quartet_contrasts.csv": sha256_file(out / "quartet_contrasts.csv")}}
    (out / "preparation_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
