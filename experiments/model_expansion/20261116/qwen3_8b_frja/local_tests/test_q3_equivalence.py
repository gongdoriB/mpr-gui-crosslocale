"""CPU tests, mock model, offline. Run with the Mac test venv python, cwd = a workspace repo made by tools/make_q3_template.sh with the addon unpacked
(see run_q3_tests.sh, section 'py'). Prints one line per check: PASS/FAIL, and a final count.

 A  REL new_only plan counts (29,148 rows / 26,057 inferences) and fr/ja filter semantics
 B  Exp A 6-language runner: row/inference counts for O/T/T_retry/R (split all/main/pilot, smoke), T==O / T==R pair counts, input checks reject tampered inputs
 C  prompt equivalence: 4-language Exp A rows through the NEW runner (mock) == OLD runner run_rel_T.py (mock): all 5,688 main inferences identical (prompt, answer, scores, batches);
    and the rendered prompts of the REAL earlier Qwen3-VL-8B runs (Exp A 5,688 rows, REL 18,432 canonical rows) are reproduced exactly from the new runners' question text
 D  consistency subset: selection deterministic (re-run gives the shipped file), ids are old 4-language canonical ids, disjoint from the fr/ja REL plan, results folders separate,
    the 600 rows run through the unchanged 6-language runner (mock) in their own folder, comparison script output structure
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(os.environ["Q3WS"]).resolve()          # workspace repo (cwd for the runners)
ROOT = Path("${PROJECT_ROOT}")
ADDON = Path(os.environ["Q3ADDON"]).resolve()      # addon folder in the output folder (for tools)
TOOLS = ADDON.parent / "tools"
PY = sys.executable
Q3 = REPO / "followup_rel_6lang/q3_frja"
CFG = Q3 / "qwen3_vl_8b_4090_channels_last.yaml"
OLD_QWEN_REL = ROOT / "internvl35_prep/mpr-gui-crosslocale/followup_rel_4lang/artifacts/qwen3_single4090_20261107/full_rel"
OLD_QWEN_EXPA = ROOT / "expA_results_20261114/qwen3/expA_qwen3/full"
TMP = Path(os.environ.get("Q3TMP", tempfile.mkdtemp(prefix="q3tests_")))
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "01_code/src"))
res = Counter()


def ck(name: str, ok: bool, detail: str = "") -> None:
    res["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "01_code/src")}, **kw)


def rd(p) -> list[dict]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
from mpr_crosslocale.inference.prompts import build_prompt_text  # noqa: E402
from mpr_crosslocale.inference.system_prompts import build_system_prompt  # noqa: E402

SYSTEM, _, _ = build_system_prompt("english_fixed", "en")


def qwen_chat(question: str) -> str:   # the Qwen3-VL chat rendering seen in every earlier prediction
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{build_prompt_text(question, 'mpr_label_only_v1')}<|im_end|>\n<|im_start|>assistant\n")


# ---------------------------------------------------------------- A: REL new_only plan
print("### A REL new_only plan")
a = TMP / "A_dry"
r = sh([PY, "followup_rel_6lang/run_inference_6lang.py", "--model-config", str(CFG), "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang",
        "--batch-size", "1", "--languages-subset", "new_only", "--dry-run", "--output-dir", str(a)])
ck("A1 dry-run new_only exits 0", r.returncode == 0, r.stderr[-300:])
plan = json.loads((a / "dry_run_plan.json").read_text())
ck("A2 29,148 condition rows, 26,057 inferences, 366 items, batch 1", (plan["condition_rows"], plan["actual_inferences"], plan["items"], plan["batch_size"]) == (29148, 26057, 366, 1), str(plan))
ck("A3 languages en zh th ru fr ja, subset new_only", plan["languages"] == ["en", "zh", "th", "ru", "fr", "ja"] and plan["languages_subset"] == "new_only")
conds = rd(REPO / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl")
sel = [c for c in conds if {"fr", "ja"} & {c["query_language"], c["gui_language"]}]
ck("A4 independent count: rows with fr or ja in conditions_6lang.jsonl = 29,148", len(sel) == 29148, str(len(sel)))
ck("A5 no kept row lacks fr/ja; 52,572 - 29,148 = 23,424 old 4-language rows left out (= earlier run's rows)", len(conds) - len(sel) == 23424)
ck("A6 plan.json of the InternVL2.5 queue (same module sha) agrees: 26,057 / 29,148",
   json.loads((ROOT / "model_expansion_20261116/internvl25_rel6/local_tests/plan_dry_new_only.json").read_text())["actual_inferences"] == 26057)
a4 = TMP / "A_dry_b4"
r4 = sh([PY, "followup_rel_6lang/run_inference_6lang.py", "--model-config", str(CFG), "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang",
         "--batch-size", "4", "--strict-batch-size", "--languages-subset", "new_only", "--dry-run", "--output-dir", str(a4)])
p4 = json.loads((a4 / "dry_run_plan.json").read_text())
ck("A7 batch 4 + --strict-batch-size accepted (Qwen family); same inference count, planned batches ceil(26057/4)=6515", r4.returncode == 0 and p4["actual_inferences"] == 26057 and p4["planned_model_batches"] == 6515, str(p4.get("planned_model_batches")))
ck("A8 batch 1 and batch 4 give different experiment fingerprints (never mixed in one folder)", plan["experiment_fingerprint"] != p4["experiment_fingerprint"])

# ---------------------------------------------------------------- B: Exp A counts
print("### B Exp A 6-language runner counts")
EXA = [PY, "followup_rel_6lang/q3_frja/expA6/run_rel_T_6lang.py", "--model-config", str(CFG), "--mock-model"]


def exa(variant, split, batch, out, extra=()):
    r = sh(EXA + ["--variant", variant, "--split", split, "--batch-size", str(batch), "--output-dir", str(out), *extra])
    return r, (json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else None)


def exa_dry(variant, split, batch, out):
    r = sh([x for x in EXA if x != "--mock-model"] + ["--variant", variant, "--split", split, "--batch-size", str(batch), "--output-dir", str(out), "--dry-run"])
    return r, json.loads((out / "plan.json").read_text())


for variant, split, pairs in (("main", "all", 3060), ("main", "main", 2844), ("main", "pilot", 216), ("retry", "all", 159), ("retry", "main", 148), ("retry", "pilot", 11)):
    r, p = exa_dry(variant, split, 3, TMP / f"B_{variant}_{split}")
    tc = "T" if variant == "main" else "T_retry"
    ok = r.returncode == 0 and p["pairs"] == pairs and p["inferences"] == 3 * pairs and p["by_condition"] == {"O": pairs, tc: pairs, "R": pairs}
    ck(f"B {variant}/{split}: {pairs} pairs -> {3 * pairs} inferences, O/{tc}/R each {pairs}", ok, (r.stderr[-300:] if not ok else ""))
    if variant == "main" and split == "all":
        ck("B main/all: items 170, T==O pairs 5, T==R pairs 179 (independent count from the input files)", (p["items"], p["T_equals_O_pairs"], p["T_equals_R_pairs"]) == (170, 5, 179), str((p["items"], p["T_equals_O_pairs"], p["T_equals_R_pairs"])))
        ck("B main/all: by split main 2,844 / pilot 216", p["by_split"] == {"main": 2844, "pilot": 216}, str(p["by_split"]))
r, p = exa_dry("main", "all", 1, TMP / "B_b1")
ck("B batch 1 accepted", r.returncode == 0 and p["batch_size"] == 1)
r, p = exa_dry("main", "all", 2, TMP / "B_b2") if False else (sh([x for x in EXA if x != "--mock-model"] + ["--variant", "main", "--batch-size", "2", "--output-dir", str(TMP / "B_b2"), "--dry-run"]), None)
ck("B batch 2 refused", r.returncode != 0 and "batch size 1 or 3" in (r.stderr + r.stdout))
r, p = exa_dry("retry", "all", 3, TMP / "B_retry_fp")
_, pm = exa_dry("main", "all", 3, TMP / "B_main_fp")
ck("B retry and main have different fingerprints", p["experiment_fingerprint"] != pm["experiment_fingerprint"])
# smoke selection: 3 items
r, p = exa("main", "all", 3, TMP / "B_smoke", ["--max-items", "3"])
ck("B smoke (--max-items 3): 54 pairs -> 162 inferences, 3 items", r.returncode == 0 and p["inferences"] == 162 and p["items"] == 3, str(p and (p["inferences"], p["items"])))

# input checks reject tampered inputs
src_in = Q3 / "expA6_inputs"


def tampered(name, edit):
    d = TMP / f"tamper_{name}"
    if d.exists():
        subprocess.run(["rm", "-rf", str(d)])
    d.mkdir(parents=True)
    for f in src_in.iterdir():
        (d / f.name).write_bytes(f.read_bytes())
    edit(d)
    r = sh([x for x in EXA if x != "--mock-model"] + ["--inputs-dir", str(d), "--lock", str(d / "expA6_lock.json"), "--variant", "main", "--batch-size", "3", "--output-dir", str(TMP / f"tamper_out_{name}"), "--dry-run"])
    return r


def e_hash(d):   # change a byte without updating the lock
    p = d / "T_gemma_conditions.jsonl"
    p.write_bytes(p.read_bytes().replace(b"Recherche", b"Rechercha", 1))


def e_question(d):  # change a question AND update the lock sha: must still fail the stem+options rule
    p = d / "T_gemma_conditions.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[0]["question"] = rows[0]["question"] + " x"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    lk = json.loads((d / "expA6_lock.json").read_text()); lk["files"]["T_gemma_conditions.jsonl"] = sha(p); (d / "expA6_lock.json").write_text(json.dumps(lk))


def e_or(d):   # O row differs from the frozen REL O row, lock updated
    p = d / "OR_gemma_conditions.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[0]["question"] = rows[0]["question"].replace("Search", "Find")
    rows[0]["question_stem"] = rows[0]["question_stem"].replace("Search", "Find")
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    lk = json.loads((d / "expA6_lock.json").read_text()); lk["files"]["OR_gemma_conditions.jsonl"] = sha(p); (d / "expA6_lock.json").write_text(json.dumps(lk))


def e_gold(d):  # T gold differs, lock updated
    p = d / "T_gemma_conditions.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[5]["gold_answer"] = "A" if rows[5]["gold_answer"] != "A" else "B"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    lk = json.loads((d / "expA6_lock.json").read_text()); lk["files"]["T_gemma_conditions.jsonl"] = sha(p); (d / "expA6_lock.json").write_text(json.dumps(lk))


def e_dup(d):  # duplicate a pair, lock updated
    p = d / "T_gemma_conditions.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[0]]) + "\n")
    lk = json.loads((d / "expA6_lock.json").read_text()); lk["files"]["T_gemma_conditions.jsonl"] = sha(p); (d / "expA6_lock.json").write_text(json.dumps(lk))


for name, edit, hint in (("hash", e_hash, "differs from the lock"), ("question", e_question, "not stem+options"), ("or", e_or, "differs from the frozen REL row"),
                         ("gold", e_gold, "Gold differs"), ("dup", e_dup, "Unexpected T pair counts")):
    r = tampered(name, edit)
    ck(f"B tampered input '{name}' is refused ({hint})", r.returncode != 0 and hint in (r.stderr + r.stdout), (r.stderr + r.stdout)[-200:] if r.returncode == 0 else "")

# ---------------------------------------------------------------- C: prompt equivalence with the old runner (mock) and with the real earlier predictions
print("### C prompt equivalence")
c = TMP / "C"
c.mkdir(parents=True, exist_ok=True)
old_T = rd(REPO / "followup_rel_4lang/expA/expA_inputs/T_conditions.jsonl")
c4 = rd(REPO / "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl")
by4 = {(r["pair_id"], r["condition"]): r for r in c4}
or_rows = []
for t in old_T:
    for cond in ("O", "R"):
        f = by4[(t["pair_id"], cond)]
        or_rows.append({"pair_id": f["pair_id"], "parallel_id": f["parallel_id"], "split": t["split"], "query_language": f["query_language"], "gui_language": f["gui_language"],
                        "condition": cond, "image_path": "followup_rel_4lang/" + f["image_path"], "image_sha256": t["image_sha256"], "gold_answer": f["gold_answer"],
                        "question_stem": f["question_stem"], "options": f["options"], "question": f["question"]})
t_rows = [{**t, "condition": "T_gemma"} for t in old_T]
w = lambda name, rows: (c / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
w("OR_gemma_conditions.jsonl", or_rows); w("T_gemma_conditions.jsonl", t_rows); w("T_gemma_retry_conditions.jsonl", t_rows)
lock4 = {"files": {n: sha(c / n) for n in ("OR_gemma_conditions.jsonl", "T_gemma_conditions.jsonl", "T_gemma_retry_conditions.jsonl")},
         "frozen_conditions_sha256": sha(REPO / "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl"), "expected_pairs": {"main": 1896, "pilot": 48},
         "expected_retry_changed_pairs": 0, "require_new_language": False}
(c / "lock.json").write_text(json.dumps(lock4))
old_out = c / "old_out"
ro = sh([PY, "followup_rel_4lang/expA/run_rel_T.py", "--model", "qwen3", "--batch-size", "3", "--model-config", str(CFG), "--mock-model", "--split", "main", "--output-dir", str(old_out)])
ck("C1 OLD runner run_rel_T.py (sha d232a13d) mock run, 5,688 inferences", ro.returncode == 0 and len(rd(old_out / "predictions.jsonl")) == 5688, ro.stderr[-300:])
new_out = c / "new_out"
rn = sh(EXA + ["--inputs-dir", str(c), "--lock", str(c / "lock.json"), "--frozen-conditions", "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl", "--variant", "main", "--split", "main",
               "--batch-size", "3", "--output-dir", str(new_out)])
ck("C2 NEW runner on the 4-language rows (mock), 5,688 inferences", rn.returncode == 0 and len(rd(new_out / "predictions.jsonl")) == 5688, rn.stderr[-300:])
IGN = {"experiment_fingerprint", "timestamp", "hardware", "software_versions", "code_commit", "attempt", "batch_runtime_ms", "inference_runtime_ms"}
po = {r["input_id"]: r for r in rd(old_out / "predictions.jsonl")}
pn = {r["input_id"]: r for r in rd(new_out / "predictions.jsonl")}
ck("C3 same input ids", set(po) == set(pn))
diff = [k for k in po if {a: b for a, b in po[k].items() if a not in IGN} != {a: b for a, b in pn[k].items() if a not in IGN}]
ck("C4 every record identical except timing/fingerprint fields (prompt, mock answer, label scores, pair/split/condition, batch id and size)", not diff, f"{len(diff)} differ, e.g. {diff[:2]}")
ck("C5 rendered_prompt identical in 5,688/5,688", all(po[k]["rendered_prompt"] == pn[k]["rendered_prompt"] for k in po))
ck("C6 batches: one pair (O, T, R) per batch in both runs", all(r["batch_size"] == 3 for r in pn.values()) and len({r["batch_id"] for r in pn.values()}) == 1896)
ck("C7 fingerprints differ (new schema/runner), as required", json.loads((old_out / "run_contract.json").read_text())["experiment_fingerprint"] != json.loads((new_out / "run_contract.json").read_text())["experiment_fingerprint"])
# real earlier Exp A predictions: rendered prompt = Qwen chat rendering of the same question text
real = rd(OLD_QWEN_EXPA / "predictions.jsonl")
q_by = {r["input_id"]: r for r in rd(new_out / "predictions.jsonl")}
qtext = {}
for t in old_T:
    qtext[(t["pair_id"], "T")] = t["question"]
for pr in or_rows:
    qtext[(pr["pair_id"], pr["condition"])] = pr["question"]
bad = [r["input_id"] for r in real if r["rendered_prompt"] != qwen_chat(qtext[(r["pair_id"], r["condition"])])]
ck("C8 REAL earlier Qwen3-VL-8B Exp A rendered prompts (5,688) == chat template + system prompt + build_prompt_text(question of the new runner's rows)", len(real) == 5688 and not bad, f"{len(bad)} differ")
# real earlier REL predictions vs 6-language data rows
rel_real = rd(OLD_QWEN_REL / "predictions.jsonl.gz")
by6 = {(c_["pair_id"], c_["condition"]): c_ for c_ in conds}
bad = []
for r in rel_real:
    parts = r["input_id"].split("::")
    pid, cond = "::".join(parts[1:-1]), parts[-1]
    if r["rendered_prompt"] != qwen_chat(by6[(pid, cond)]["question"]):
        bad.append(r["input_id"])
ck("C9 REAL earlier REL rendered prompts (18,432 canonical) == chat template + question text of conditions_6lang.jsonl", len(rel_real) == 18432 and not bad, f"{len(bad)} differ, e.g. {bad[:2]}")
ck("C10 their input ids exist in conditions_6lang as rows (pair_id, condition)", all(("::".join(r["input_id"].split("::")[1:-1]), r["input_id"].split("::")[-1]) in by6 for r in rel_real))

# ---------------------------------------------------------------- D: consistency subset
print("### D consistency subset")
ids_file = Q3 / "consistency_subset_ids.json"
ids = json.loads(ids_file.read_text())
d2 = TMP / "D_select"
r = subprocess.run([PY, str(TOOLS / "consistency_select.py"), str(OLD_QWEN_REL / "predictions.jsonl.gz"), str(OLD_QWEN_REL / "run_contract.json"), str(d2), "600"], capture_output=True, text=True)
ck("D1 selection script runs", r.returncode == 0, r.stderr[-200:])
ck("D2 deterministic: re-run gives byte-identical ids file and reference file as shipped", sha(d2 / "consistency_subset_ids.json") == sha(ids_file) and sha(d2 / "consistency_reference.jsonl") == sha(Q3 / "consistency_reference.jsonl"))
d3 = TMP / "D_select2"
subprocess.run([PY, str(TOOLS / "consistency_select.py"), str(OLD_QWEN_REL / "predictions.jsonl.gz"), str(OLD_QWEN_REL / "run_contract.json"), str(d3), "600"], capture_output=True, text=True)
ck("D3 second re-run identical (3 runs agree)", sha(d3 / "consistency_subset_ids.json") == sha(d2 / "consistency_subset_ids.json"))
ck("D4 600 unique ids, seed 20261116, 52 cells, 16 language pairs, 4 conditions", len(ids["ids"]) == len(set(ids["ids"])) == 600 and ids["seed"] == 20261116 and ids["cells"] == 52 and len(ids["by_language_pair"]) == 16 and set(ids["by_condition"]) == set("ORCF"), str(ids["by_condition"]))
cell_counts = Counter(i.split("::")[-1] + "|" + i.split("::")[-3] + "|" + i.split("::")[-2] for i in ids["ids"])
ck("D5 every cell has 11 or 12 ids (floor 600/52 = 11, remainder spread)", set(cell_counts.values()) <= {11, 12} and len(cell_counts) == 52 and sum(cell_counts.values()) == 600)
old_ids = {r["input_id"] for r in rel_real}
ck("D6 all ids are canonical inferences of the earlier run", set(ids["ids"]) <= old_ids)
# plan of the fr/ja REL run (canonical ids) from the mock full run
new_pred = REPO / "followup_rel_6lang/results/rel6new_qwen3_vl_8b/full/predictions.jsonl"
if new_pred.exists():
    new_ids = {r["input_id"] for r in rd(new_pred)}
    ck("D7 DISJOINT from the fr/ja REL results (canonical ids of rel6new_qwen3_vl_8b/full)", not (set(ids["ids"]) & new_ids) and len(new_ids) == 26057, str(len(set(ids["ids"]) & new_ids)))
else:
    ck("D7 (skipped: run the REL full mock first)", False)
ck("D8 every id has a language that is NOT fr/ja (old 4-language rows only)", all(not ({"fr", "ja"} & set(i.split("::")[-3:-1])) for i in ids["ids"]))
sc = {n: (Q3 / n).read_text() for n in ("run_rel6_qwen3.sh", "run_consistency_qwen3.sh", "run_expA6_qwen3.sh")}
outs = {n: [l for l in s.splitlines() if l.startswith("OUT=")][0].split("#")[0].strip() for n, s in sc.items()}
ck("D9 three different result folders, none inside followup_rel_4lang/results", len(set(outs.values())) == 3 and all("followup_rel_4lang" not in o for o in outs.values()), str(outs))
# consistency run (mock) through the unchanged runner in its own folder
cc = TMP / "D_run"
r = sh([PY, "followup_rel_6lang/q3_frja/consistency_run.py", "--ids", str(ids_file), "--", "--model-config", str(CFG), "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang",
        "--batch-size", "1", "--languages-subset", "all", "--mock-model", "--output-dir", str(cc)])
pc = rd(cc / "predictions.jsonl") if (cc / "predictions.jsonl").exists() else []
ck("D10 consistency run (mock) via the unchanged 6-language runner: 600 success rows, exactly the subset ids", r.returncode == 0 and len(pc) == 600 and {x["input_id"] for x in pc} == set(ids["ids"]), r.stderr[-300:])
plan_cc = json.loads((cc / "plan.json").read_text())
ck("D11 plan: 600 condition rows = 600 inferences, no reused rows", plan_cc["condition_rows"] == 600 and plan_cc["actual_inferences"] == 600 and plan_cc["reused_condition_rows"] == 0)
ck("D12 consistency fingerprint differs from the fr/ja REL run (separate experiment)", plan_cc["experiment_fingerprint"] != json.loads((REPO / "followup_rel_6lang/results/rel6new_qwen3_vl_8b/full/plan.json").read_text())["experiment_fingerprint"] if new_pred.exists() else True)
# unknown id refused
bad_ids = TMP / "bad_ids.json"
x = dict(ids); x["ids"] = ids["ids"][:-1] + ["rel_orcf::rel::9/none_1.jpg::en::zh::O"]
bad_ids.write_text(json.dumps(x))
r = sh([PY, "followup_rel_6lang/q3_frja/consistency_run.py", "--ids", str(bad_ids), "--", "--model-config", str(CFG), "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang",
        "--batch-size", "1", "--dry-run", "--output-dir", str(TMP / "D_bad")])
ck("D13 an id that does not exist stops the run before inference", r.returncode != 0 and "missing" in (r.stderr + r.stdout), (r.stderr + r.stdout)[-200:])
# compare script structure (mock answers vs the real earlier answers: agreement ~ chance, only the format is checked)
r = sh([PY, "followup_rel_6lang/q3_frja/consistency_compare.py", "--run", str(cc), "--reference", str(Q3 / "consistency_reference.jsonl"), "--ids", str(ids_file), "--out", str(TMP / "agree.json")])
ag = json.loads((TMP / "agree.json").read_text()) if (TMP / "agree.json").exists() else {}
ck("D14 comparison script: overall + by_condition + by_language_pair + cell table written", r.returncode == 0 and set(ag.get("by_condition", {})) == set("ORCF") and len(ag.get("by_language_pair(query->gui)", {})) == 16 and len(ag.get("by_condition_and_language_pair", {})) == 52, r.stderr[-300:])
# the mock model renders no chat template, so the question-text comparison is checked on a copy of the run whose prompts are rendered the way the real adapter does (qwen_chat)
real_fmt = TMP / "D_run_realfmt"
real_fmt.mkdir(exist_ok=True)
qq = {x["input_id"]: by6[("::".join(x["input_id"].split("::")[1:-1]), x["input_id"].split("::")[-1])]["question"] for x in pc}
with (real_fmt / "predictions.jsonl").open("w") as f:
    for x in pc:
        f.write(json.dumps({**x, "rendered_prompt": qwen_chat(qq[x["input_id"]])}, ensure_ascii=False) + "\n")
(real_fmt / "run_contract.json").write_bytes((cc / "run_contract.json").read_bytes())
r = sh([PY, "followup_rel_6lang/q3_frja/consistency_compare.py", "--run", str(real_fmt), "--reference", str(Q3 / "consistency_reference.jsonl"), "--ids", str(ids_file), "--out", str(TMP / "agree_fmt.json")])
agf = json.loads((TMP / "agree_fmt.json").read_text())
ck("D15 comparison: n 600 / success 600 / the question text of the 600 re-run rows equals the earlier REAL prompts (chat-format copy of the mock run)", agf.get("n_success_new") == 600 and agf.get("question_text_mismatches_vs_earlier_prompt") == [], str(agf.get("question_text_mismatches_vs_earlier_prompt"))[:200])
ck("D15b mock prompts (no chat template) are reported as mismatches, i.e. the check is not vacuous", len(ag.get("question_text_mismatches_vs_earlier_prompt", [])) == 600)
# synthetic reference check: feeding the new run as its own reference gives agreement 1.0
fake_ref = TMP / "fake_ref.jsonl"
with fake_ref.open("w") as f:
    for p in pc:
        f.write(json.dumps({"input_id": p["input_id"], "parsed_generated_label": p["parsed_generated_label"], "scored_predicted_label": p["scored_predicted_label"], "generation_correct": p["generation_correct"],
                            "gold_probability": p["gold_probability"], "batch_size": 4, "inference_runtime_ms": 1.0, "gpu": "x", "software_versions": {}, "rendered_prompt": p["rendered_prompt"]}) + "\n")
r = sh([PY, "followup_rel_6lang/q3_frja/consistency_compare.py", "--run", str(cc), "--reference", str(fake_ref), "--ids", str(ids_file), "--out", str(TMP / "agree2.json")])
ag2 = json.loads((TMP / "agree2.json").read_text())
ck("D16 comparison logic: identical answers -> label agreement 1.0, gold probability diff 0", ag2["overall"]["label_agreement"] == 1.0 and ag2["overall"]["max_abs_gold_probability_diff"] == 0.0)

print(f"\nRESULT: {res['pass']} passed, {res['fail']} failed")
sys.exit(1 if res["fail"] else 0)
