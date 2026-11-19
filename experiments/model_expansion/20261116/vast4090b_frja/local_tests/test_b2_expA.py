"""CPU tests, mock model, offline, for the vast4090b fr/ja build (Exp A multi-model runner/gate, REL plans, input builder).
Run with the Mac test venv python: B2WS=<workspace repo with the addon unpacked> B2KEYS="internvl25_8b internvl35_8b" python test_b2_expA.py
(B2WS for the InternVL repo = ivl25_repo copy; for UI-TARS = Qwen repo copy with the sizes_uitars generic runner; see mk_ws.sh). Prints PASS/FAIL per check and a count.

 A  plan counts per model/step (new scope: 2,844 pairs -> 8,532 / retry 148 pairs -> 444 / smoke 162; all6 scope: 4,740 pairs -> 14,220 / retry 262 -> 786 / smoke 270), batch rules
 B  prompt equivalence: the earlier 4-language rows through the NEW runner (mock) == the model's OLD 4-language runner (mock), all 5,688 main inferences identical
 C  the REAL rendered prompts of the earlier 4-language Exp A run (5,688) equal the model's chat template + system prompt + question text of the rows the new runner plans
 D  all6 (InternVL2.5): the 4-language subset of the planned rows == the earlier 4-language Exp A rows (input ids, order, question text, rendered prompts), mock records identical to the old runner's
 E  REL new_only (internvl35_8b, uitars_15_7b): dry-run plan 26,057 / 29,148 with the earlier run's strict flag; real earlier REL prompts (18,432) reproduced from conditions_6lang.jsonl
 F  input checks refuse tampered inputs; result folders are disjoint
"""
from __future__ import annotations

import gzip, hashlib, json, os, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

REPO = Path(os.environ["B2WS"]).resolve()
KEYS = os.environ["B2KEYS"].split()
ROOT = Path("${PROJECT_ROOT}")
PY = sys.executable
B2 = REPO / "followup_rel_6lang/b2_frja"
TMP = Path(os.environ.get("B2TMP", tempfile.mkdtemp(prefix="b2tests_")))
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "01_code/src"))
res = Counter()
REAL_EXPA = {"internvl25_8b": ROOT / "model_results_20261114/internvl25_expA/unpacked/expA_internvl25_8b/expa_full/predictions.jsonl",
             "internvl35_8b": ROOT / "expA_results_20261114/internvl35/expA_internvl35/full/predictions.jsonl",
             "uitars_15_7b": ROOT / "model_results_20261114/uitars_15_7b/unpacked/sizes_uitars_15_7b/expa_full/predictions.jsonl"}
REAL_REL = {"internvl35_8b": ROOT / "internvl35_prep/results_internvl35_20261109/full_rel/predictions.jsonl",
            "uitars_15_7b": ROOT / "model_results_20261114/uitars_15_7b/unpacked/sizes_uitars_15_7b/rel_full/predictions.jsonl"}
CFG = {"internvl25_8b": B2 / "models/internvl2_5_8b.yaml", "internvl35_8b": B2 / "models/internvl3_5_8b_4090.yaml", "uitars_15_7b": B2 / "models/uitars_1_5_7b_4090.yaml"}


def ck(name: str, ok: bool, detail: str = "") -> None:
    res["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "01_code/src")}, **kw)


def rd(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
from mpr_crosslocale.inference.prompts import build_prompt_text  # noqa: E402
from mpr_crosslocale.inference.system_prompts import build_system_prompt  # noqa: E402

SYSTEM, _, _ = build_system_prompt("english_fixed", "en")


def chat(key: str, question: str) -> str:
    img = "<|vision_start|><|image_pad|><|vision_end|>" if key == "uitars_15_7b" else "<image>\n"
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{img}{build_prompt_text(question, 'mpr_label_only_v1')}<|im_end|>\n<|im_start|>assistant\n"


def old_cmd(key: str, out: Path) -> list:
    cfg = CFG[key]
    if key == "internvl25_8b":
        return [PY, "followup_rel_4lang/expA_ivl25/run_rel_T_generic.py", "--model", key, "--reference-contract", "followup_rel_4lang/expA_ivl25/reference_contract_internvl25_rel_20261005.json",
                "--model-config", cfg, "--mock-model", "--split", "main", "--output-dir", out]
    if key == "internvl35_8b":
        return [PY, "followup_rel_4lang/expA/run_rel_T.py", "--model", "internvl35", "--batch-size", "1", "--model-config", cfg, "--mock-model", "--split", "main", "--output-dir", out]
    return [PY, "followup_rel_4lang/model_expansion/run_rel_T_generic.py", "--model", key, "--reference-contract", "followup_rel_4lang/rstem_ropts_inputs/run_contract_qwen3_rel_20261107.json",
            "--model-config", cfg, "--mock-model", "--split", "main", "--output-dir", out]


def new_cmd(key, *extra, batch=1):
    return [PY, "followup_rel_6lang/b2_frja/expA6/run_rel_T_6lang_multi.py", "--model", key, "--model-config", CFG[key], "--batch-size", batch, *extra]


IGN = {"experiment_fingerprint", "timestamp", "hardware", "software_versions", "code_commit", "attempt", "batch_runtime_ms", "inference_runtime_ms"}
strip = lambda r, extra=(): {a: b for a, b in r.items() if a not in IGN and a not in extra}
conds6 = rd(REPO / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl")
by6 = {(c["pair_id"], c["condition"]): c for c in conds6}
old_T = rd(REPO / "followup_rel_4lang/expA/expA_inputs/T_conditions.jsonl")
c4 = rd(REPO / "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl")
by4 = {(r["pair_id"], r["condition"]): r for r in c4}
ck("0 earlier T_conditions.jsonl in the workspace is the one of the earlier run (sha 0ebbed0f)", sha(REPO / "followup_rel_4lang/expA/expA_inputs/T_conditions.jsonl").startswith("0ebbed0f"))

# 4-language input set for the old-vs-new comparison
c = TMP / "C4"; c.mkdir(parents=True, exist_ok=True)
or_rows = []
for t in old_T:
    for cond in ("O", "R"):
        f = by4[(t["pair_id"], cond)]
        or_rows.append({"pair_id": f["pair_id"], "parallel_id": f["parallel_id"], "split": t["split"], "query_language": f["query_language"], "gui_language": f["gui_language"], "condition": cond,
                        "image_path": "followup_rel_4lang/" + f["image_path"], "image_sha256": t["image_sha256"], "gold_answer": f["gold_answer"], "question_stem": f["question_stem"], "options": f["options"], "question": f["question"]})
t_rows = [{**t, "condition": "T_gemma"} for t in old_T]
w = lambda d, name, rows: (d / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
w(c, "OR_gemma_conditions.jsonl", or_rows); w(c, "T_gemma_conditions.jsonl", t_rows); w(c, "T_gemma_retry_conditions.jsonl", t_rows)
(c / "lock.json").write_text(json.dumps({"files": {n: sha(c / n) for n in ("OR_gemma_conditions.jsonl", "T_gemma_conditions.jsonl", "T_gemma_retry_conditions.jsonl")},
                                         "frozen_conditions_sha256": sha(REPO / "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl"), "expected_pairs": {"main": 1896, "pilot": 48},
                                         "expected_retry_changed_pairs": 0, "require_new_language": False}))

for key in KEYS:
    print(f"### {key}")
    # ---------------- A plan counts
    for scope, variant, pairs in (("new", "main", 2844), ("new", "retry", 148)) + ((("all6", "main", 4740), ("all6", "retry", 262)) if key == "internvl25_8b" else ()):
        out = TMP / f"A_{key}_{scope}_{variant}"
        r = sh(new_cmd(key, "--scope", scope, "--variant", variant, "--output-dir", out, "--dry-run"))
        p = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else {}
        tc = "T" if variant == "main" else "T_retry"
        ck(f"A {scope}/{variant}: {pairs} pairs -> {3 * pairs} inferences", r.returncode == 0 and p.get("pairs") == pairs and p.get("inferences") == 3 * pairs and p.get("by_condition") == {"O": pairs, tc: pairs, "R": pairs} and p.get("batch_size") == 1,
           r.stderr[-300:] if r.returncode else str(p)[:200])
    r = sh(new_cmd(key, "--scope", "new", "--variant", "main", "--output-dir", TMP / f"A_{key}_b2", "--dry-run", batch=2))
    ck("A batch 2 refused", r.returncode != 0 and "Use batch size" in r.stderr)
    if key != "uitars_15_7b":
        r = sh(new_cmd(key, "--scope", "new", "--variant", "main", "--output-dir", TMP / f"A_{key}_b3", "--dry-run", batch=3))
        ck("A batch 3 refused (earlier run: batch 1 only)", r.returncode != 0)
    else:
        r = sh(new_cmd(key, "--scope", "new", "--variant", "main", "--output-dir", TMP / f"A_{key}_b3", "--dry-run", batch=3))
        ck("A uitars batch 3 allowed as an override (one pair per batch)", r.returncode == 0)
    r = sh(new_cmd(key, "--scope", "new", "--variant", "main", "--max-items", "3", "--mock-model", "--output-dir", TMP / f"A_{key}_smoke"))
    p = json.loads((TMP / f"A_{key}_smoke/plan.json").read_text())
    ck("A smoke --max-items 3 (new): 162 inferences", r.returncode == 0 and p["inferences"] == 162 and p["items"] == 3, r.stderr[-200:])
    if key == "internvl25_8b":
        r = sh(new_cmd(key, "--scope", "all6", "--variant", "main", "--max-items", "3", "--mock-model", "--output-dir", TMP / f"A_{key}_smoke6"))
        p = json.loads((TMP / f"A_{key}_smoke6/plan.json").read_text())
        ck("A smoke --max-items 3 (all6): 270 inferences", r.returncode == 0 and p["inferences"] == 270 and p["items"] == 3, str(p["inferences"]))
    rc = json.loads((B2 / "contracts" / {"internvl25_8b": "ref_internvl25_expA_20261114.json", "internvl35_8b": "ref_internvl35_expA_20261114.json", "uitars_15_7b": "ref_uitars_expA_20261114.json"}[key]).read_text())
    pm = json.loads((TMP / f"A_{key}_new_main/plan.json").read_text())
    ck("A model config of the run == earlier Exp A run contract; strict flag follows the earlier run", rc["model_config"] == __import__("yaml").safe_load(CFG[key].read_text()) and rc["batch_size"] == 1)
    r = sh(new_cmd(key, "--scope", "new", "--variant", "main", "--max-items", "1", "--mock-model", "--output-dir", TMP / f"A_{key}_strict"))
    ck("A run contract strict_batch_size = earlier run", json.loads((TMP / f"A_{key}_strict/run_contract.json").read_text())["strict_batch_size"] == rc["strict_batch_size"], str(rc["strict_batch_size"]))

    # ---------------- B old runner vs new runner on the 4-language rows
    old_out, new_out = TMP / f"B_{key}_old", TMP / f"B_{key}_new"
    ro = sh(old_cmd(key, old_out))
    ck("B1 OLD runner (mock) 5,688 inferences", ro.returncode == 0 and len(rd(old_out / "predictions.jsonl")) == 5688, ro.stderr[-300:])
    rn = sh(new_cmd(key, "--inputs-dir", c, "--lock", c / "lock.json", "--frozen-conditions", "followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl", "--variant", "main", "--split", "main", "--mock-model", "--output-dir", new_out))
    ck("B2 NEW runner on the 4-language rows (mock) 5,688 inferences", rn.returncode == 0 and len(rd(new_out / "predictions.jsonl")) == 5688, rn.stderr[-300:])
    po = {r["input_id"]: r for r in rd(old_out / "predictions.jsonl")}
    pn = {r["input_id"]: r for r in rd(new_out / "predictions.jsonl")}
    ck("B3 same input ids and same order", list(po) == list(pn))
    bad = [k for k in po if strip(po[k]) != strip(pn[k])]
    ck("B4 every record identical except timing/fingerprint fields (rendered prompt, mock answer, label scores, batch id/size)", not bad, f"{len(bad)} differ")
    ck("B5 rendered_prompt identical 5,688/5,688", all(po[k]["rendered_prompt"] == pn[k]["rendered_prompt"] for k in po))
    ck("B6 fingerprints differ (new schema/runner)", json.loads((old_out / "run_contract.json").read_text())["experiment_fingerprint"] != json.loads((new_out / "run_contract.json").read_text())["experiment_fingerprint"])

    # ---------------- C real earlier prompts
    real = rd(REAL_EXPA[key])
    q = {}
    for t in old_T:
        q[(t["pair_id"], "T")] = t["question"]
    for pr in or_rows:
        q[(pr["pair_id"], pr["condition"])] = pr["question"]
    bad = [r["input_id"] for r in real if r["rendered_prompt"] != chat(key, q[(r["pair_id"], r["condition"])])]
    ck("C REAL earlier Exp A rendered prompts (5,688) == chat template + system prompt + question text of the rows fed to the new runner", len(real) == 5688 and not bad, f"{len(bad)} differ")
    ck("C2 earlier real run used batch 1 and the strict flag the new runner uses", {r["batch_size"] for r in real} == {1})

    # ---------------- D all6 (InternVL2.5)
    if key == "internvl25_8b":
        sys.path.insert(0, str(B2 / "expA6"))
        import run_rel_T_6lang_multi as m  # noqa: E402
        d6 = B2 / "expA6_all6_inputs"
        rows, _ = m.load_rows(d6, d6 / "expA6_all6_lock.json", REPO / "followup_rel_6lang/data/rel_all_366/conditions_6lang.jsonl", REPO, "main", "main", None, False)
        ids = [m.input_id(r["pair_id"], r["condition"]) for r in rows]
        ck("D1 all6 plan: 14,220 rows, 4,740 pairs", len(rows) == 14220 and len({r["pair_id"] for r in rows}) == 4740)
        real_ids = [r["input_id"] for r in real]
        sub = [i for i in ids if i in set(real_ids)]
        ck("D2 the 4-language subset of the planned rows has exactly the earlier run's input ids (5,688) in the same order", sub == real_ids, f"{len(sub)} vs {len(real_ids)}")
        qtext = {m.input_id(r["pair_id"], r["condition"]): r["question"] for r in rows}
        bad = [r["input_id"] for r in real if r["rendered_prompt"] != chat(key, qtext[r["input_id"]])]
        ck("D3 prompts of the planned 4-language subset (rendered with the earlier chat template) are byte-identical to the earlier run's rendered_prompt (5,688)", not bad, f"{len(bad)} differ")
        new_ids = [i for i in ids if i not in set(real_ids)]
        ck("D4 the other 8,532 planned rows are the new fr/ja pairs (every pair involves fr or ja)", len(new_ids) == 8532 and all({"fr", "ja"} & set(i.split("::")[-3:-1]) for i in new_ids))
        out6 = TMP / "D_all6_mock"
        r6 = sh(new_cmd(key, "--scope", "all6", "--variant", "main", "--mock-model", "--output-dir", out6))
        p6 = {r["input_id"]: r for r in rd(out6 / "predictions.jsonl")} if (out6 / "predictions.jsonl").exists() else {}
        ck("D5 all6 mock run: 14,220 success rows", r6.returncode == 0 and len(p6) == 14220, r6.stderr[-200:])
        bad = [k for k in po if strip(po[k], ("batch_id",)) != strip(p6[k], ("batch_id",))]
        ck("D6 mock records of the 4-language subset in the all6 run == the OLD runner's records (all fields but timing/fingerprint/batch_id)", not bad, f"{len(bad)} differ")
        ck("D7 all6 gate-relevant: every batch holds one input, batch size 1", all(r["batch_size"] == 1 for r in p6.values()))
        g = sh([PY, "followup_rel_6lang/b2_frja/expA6/check_T6_gate_multi.py", "main", "complete", out6, "--model", key, "--batch", "1", "--expected-rows", "14220", "--full-n", "14220", "--mock"])
        ck("D8 completion gate passes on the all6 mock run", g.returncode == 0, (g.stdout + g.stderr)[-300:])
        rr = sh(new_cmd(key, "--scope", "all6", "--variant", "retry", "--mock-model", "--output-dir", TMP / "D_all6_retry"))
        pr_ = rd(TMP / "D_all6_retry/predictions.jsonl")
        ck("D9 all6 retry mock run: 786 rows (262 pairs: 114 earlier + 148 new)", rr.returncode == 0 and len(pr_) == 786 and len({r["pair_id"] for r in pr_}) == 262, rr.stderr[-200:])
        lk = json.loads((d6 / "expA6_all6_lock.json").read_text())
        ck("D10 lock: 4,740 main pairs, retry changed 114 + 148, sources shas recorded", lk["expected_pairs"] == {"main": 4740} and lk["retry_changed_pairs_old_main"] == 114 and lk["retry_changed_pairs_new_main"] == 148)
        # tampering
        tdir = TMP / "tamper_all6"; subprocess.run(["rm", "-rf", str(tdir)]); tdir.mkdir()
        for f in d6.iterdir():
            (tdir / f.name).write_bytes(f.read_bytes())
        p_ = tdir / "T_gemma_conditions.jsonl"; p_.write_bytes(p_.read_bytes().replace(b"Search", b"Searck", 1))
        rt = sh(new_cmd(key, "--scope", "all6", "--inputs-dir", tdir, "--lock", tdir / "expA6_all6_lock.json", "--variant", "main", "--output-dir", TMP / "tamper_all6_out", "--dry-run"))
        ck("D11 tampered all6 input refused (sha differs from the lock)", rt.returncode != 0 and "differs from the lock" in rt.stderr)
    # ---------------- gate on a new-scope mock run
    outn = TMP / f"G_{key}"
    sh(new_cmd(key, "--scope", "new", "--variant", "main", "--max-items", "3", "--mock-model", "--output-dir", outn))
    g = sh([PY, "followup_rel_6lang/b2_frja/expA6/check_T6_gate_multi.py", "main", "smoke", outn, "--model", key, "--batch", "1", "--full-n", "8532", "--mock"])
    ck("G smoke gate passes on the new-scope mock run and reports ETA fields", g.returncode == 0 and "eta_full_seconds" in g.stdout, (g.stdout + g.stderr)[-300:])
    # ---------------- E REL
    if key in REAL_REL:
        strict = ["--strict-batch-size"] if key == "uitars_15_7b" else []
        out = TMP / f"E_{key}"
        r = sh([PY, "followup_rel_6lang/run_inference_6lang.py", "--model-config", CFG[key], "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang", "--batch-size", "1", *strict, "--languages-subset", "new_only", "--dry-run", "--output-dir", out])
        p = json.loads((out / "dry_run_plan.json").read_text()) if (out / "dry_run_plan.json").exists() else {}
        ck("E1 REL new_only dry-run: 26,057 inferences / 29,148 rows / 366 items / batch 1", r.returncode == 0 and (p.get("actual_inferences"), p.get("condition_rows"), p.get("items"), p.get("batch_size")) == (26057, 29148, 366, 1), r.stderr[-300:])
        rc_ = json.loads((B2 / "contracts" / ("ref_internvl35_rel_20261109.json" if key == "internvl35_8b" else "ref_uitars_rel_20261114.json")).read_text())
        ck("E2 earlier REL contract: batch 1 and strict flag as used here", rc_["batch_size"] == 1 and rc_["strict_batch_size"] == bool(strict), str((rc_["batch_size"], rc_["strict_batch_size"])))
        om = TMP / f"E_{key}_mock"
        r = sh([PY, "followup_rel_6lang/run_inference_6lang.py", "--model-config", CFG[key], "--data-root", "followup_rel_6lang", "--legacy-data-root", "followup_rel_4lang", "--batch-size", "1", *strict, "--languages-subset", "new_only", "--max-items", "2", "--mock-model", "--output-dir", om])
        ck("E3 REL smoke (mock, --max-items 2): 160 rows / 112 inferences", r.returncode == 0 and json.loads((om / "plan.json").read_text())["actual_inferences"] == 112, r.stderr[-200:])
        g = sh([PY, "followup_rel_6lang/b2_frja/check_rel6_gate.py", "rel", "smoke", om, "--model", key, "--full-n", "26057"])
        ck("E4 REL smoke gate (mock) passes", g.returncode == 0, (g.stdout + g.stderr)[-300:])
        rel_real = rd(REAL_REL[key])
        bad = []
        for r_ in rel_real:
            parts = r_["input_id"].split("::"); pid, cond = "::".join(parts[1:-1]), parts[-1]
            if r_["rendered_prompt"] != chat(key, by6[(pid, cond)]["question"]):
                bad.append(r_["input_id"])
        ck("E5 REAL earlier REL rendered prompts (18,432 canonical) == chat template + question text of conditions_6lang.jsonl", len(rel_real) == 18432 and not bad, f"{len(bad)} differ")

# ---------------- F disjoint folders
scripts = {n: (B2 / n).read_text() for n in ("run_rel6_b2.sh", "run_expA6_b2.sh")}
outs = {n: [l for l in s.splitlines() if l.startswith("OUT=")][0].split("#")[0].strip() for n, s in scripts.items()}
ck("F1 REL and Exp A write to different folders, none inside followup_rel_4lang", len(set(outs.values())) == 2 and all("followup_rel_4lang" not in o for o in outs.values()), str(outs))
ck("F2 the per-model folders differ by key (rel6new_<KEY>, expA6_<KEY>), neither equals the earlier rel6_internvl25_8b", "rel6new_${KEY}" in outs["run_rel6_b2.sh"] and "expA6_${KEY}" in outs["run_expA6_b2.sh"])
print(f"\nRESULT: {res['pass']} passed, {res['fail']} failed")
sys.exit(1 if res["fail"] else 0)
