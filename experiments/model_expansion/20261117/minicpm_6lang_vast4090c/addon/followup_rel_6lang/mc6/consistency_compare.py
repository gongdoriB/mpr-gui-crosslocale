"""MiniCPM-V-4.5 build (2026-11-17) of q3_frja/consistency_compare.py (Qwen3 build, same output keys). Differences: the question-text check splits the rendered prompt at the MiniCPM image
placeholder "(<image>./</image>)" instead of "<|vision_end|>", the texts in "what" / "context" / "note" name MiniCPM, and the earlier-run batch size is 1 (strict_batch_size false) like the new one.

Compares the consistency re-run (new server) with the earlier MiniCPM-V-4.5 4-language REL predictions of the same canonical ids.
Report only: no pass/fail, the queue never stops because of it.

  python consistency_compare.py --run <dir with predictions.jsonl and run_contract.json> --reference consistency_reference.jsonl --ids consistency_subset_ids.json --out consistency_agreement.json

Agreement = the generated (parsed) label is the same. Also reported: same correctness, same scored label, mean/max |gold probability difference|, accuracy on both sides,
parse failures, and the same table for each condition, each language pair and each condition x language-pair cell. The earlier run (2026-11-16) used batch size 1 on an RTX 4090
of another Vast server (run contract); the batch size of the new run is read from its predictions."""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def rd(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def table(pairs, key):
    groups = defaultdict(list)
    for old, new in pairs:
        groups[key(old["input_id"])].append((old, new))
    out = {}
    for k, v in sorted(groups.items()):
        out[k] = {"n": len(v), "same_label": sum(o["parsed_generated_label"] == n["parsed_generated_label"] for o, n in v),
                  "label_agreement": round(sum(o["parsed_generated_label"] == n["parsed_generated_label"] for o, n in v) / len(v), 4),
                  "same_correctness": sum(bool(o["generation_correct"]) == bool(n["generation_correct"]) for o, n in v),
                  "old_accuracy": round(sum(bool(o["generation_correct"]) for o, _ in v) / len(v), 4),
                  "new_accuracy": round(sum(bool(n["generation_correct"]) for _, n in v) / len(v), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ids = json.loads(a.ids.read_text(encoding="utf-8"))
    old = {r["input_id"]: r for r in rd(a.reference)}
    new = {}
    for r in rd(a.run / "predictions.jsonl"):
        new[r["input_id"]] = r
    contract = json.loads((a.run / "run_contract.json").read_text(encoding="utf-8"))
    ok = [i for i in ids["ids"] if new.get(i, {}).get("status") == "success"]
    pairs = [(old[i], new[i]) for i in ok]
    mismatched_prompt = [i for i in ok if old[i].get("rendered_prompt") and old[i]["rendered_prompt"].split("(<image>./</image>)")[-1] != new[i]["rendered_prompt"].split("(<image>./</image>)")[-1]]
    gp = [abs(o["gold_probability"] - n["gold_probability"]) for o, n in pairs if o.get("gold_probability") is not None and n.get("gold_probability") is not None]
    cell = lambda i: i.split("::")[-1] + " " + "->".join(i.split("::")[-3:-1])
    report = {
        "what": "MiniCPM-V-4.5 REL re-run of old 4-language canonical inferences on the new server vs the earlier 4090 run (report only)",
        "subset_ids_sha256_of_list": __import__("hashlib").sha256(json.dumps(ids["ids"]).encode()).hexdigest(),
        "subset_seed": ids["seed"], "n_subset": len(ids["ids"]), "n_success_new": len(ok), "n_missing_or_failed_new": len(ids["ids"]) - len(ok),
        "old_run": {"batch_size": ids["source_batch_size"], "experiment_fingerprint": ids["source_experiment_fingerprint"],
                    "gpu": Counter(o.get("gpu") for o in old.values()).most_common(1)[0][0],
                    "software_versions": next(iter(old.values()))["software_versions"],
                    "median_inference_runtime_ms": statistics.median(o["inference_runtime_ms"] for o in old.values())},
        "new_run": {"batch_sizes": dict(Counter(n.get("batch_size") for n in new.values())),
                    "experiment_fingerprint": contract["experiment_fingerprint"],
                    "gpu": (next(iter(new.values())).get("hardware") or {}).get("gpus", [{}])[0].get("name") if new else None,
                    "software_versions": contract["software_versions"],
                    "median_inference_runtime_ms": statistics.median(n["inference_runtime_ms"] for n in new.values() if n.get("status") == "success") if ok else None},
        "overall": {"same_label": sum(o["parsed_generated_label"] == n["parsed_generated_label"] for o, n in pairs),
                    "label_agreement": round(sum(o["parsed_generated_label"] == n["parsed_generated_label"] for o, n in pairs) / len(pairs), 4) if pairs else None,
                    "same_scored_label": sum(o["scored_predicted_label"] == n["scored_predicted_label"] for o, n in pairs),
                    "same_correctness": sum(bool(o["generation_correct"]) == bool(n["generation_correct"]) for o, n in pairs),
                    "old_accuracy": round(sum(bool(o["generation_correct"]) for o, _ in pairs) / len(pairs), 4) if pairs else None,
                    "new_accuracy": round(sum(bool(n["generation_correct"]) for _, n in pairs) / len(pairs), 4) if pairs else None,
                    "new_parse_failures": sum(not n["parse_success"] for _, n in pairs),
                    "mean_abs_gold_probability_diff": round(statistics.mean(gp), 5) if gp else None,
                    "max_abs_gold_probability_diff": round(max(gp), 5) if gp else None},
        "by_condition": table(pairs, lambda i: i.split("::")[-1]),
        "by_language_pair(query->gui)": table(pairs, lambda i: "->".join(i.split("::")[-3:-1])),
        "by_condition_and_language_pair": table(pairs, cell),
        "question_text_mismatches_vs_earlier_prompt": mismatched_prompt,
        "context": "Earlier run (model_results_20261116/minicpm, 2026-11-16, another Vast RTX 4090 server, python 3.12.15) used batch size 1 like this re-run, so batch size does not explain differences; GPU kernels, driver and the Python build can. No earlier measurement of MiniCPM label stability exists. For Qwen3-VL-8B on this server the same check gave 0.9917 (batch 4 vs batch 4).",
        "note": "Report only. Agreement below about 0.90 would point to a data, image or environment difference and should be looked at before the fr/ja results are interpreted (0.90 is an arbitrary reference, not a pass/fail rule).",
    }
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_condition": {k: v["label_agreement"] for k, v in report["by_condition"].items()},
                      "new_batch_sizes": report["new_run"]["batch_sizes"], "n_success_new": len(ok)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
