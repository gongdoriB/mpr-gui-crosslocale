"""Read-only check of the 6x6 raw data and manifest of ONE repository (cwd = repository root or --repo). Used by queue_b3_4b.sh before queue_6x6.sh may reuse the data.
Checks: 01_code/data/raw/mpr_gui_bench/SOURCE.json (Hugging Face revision c1edb808..., 3,685 files, GitHub commit e4f1cfcd..., 48 jsonl files), the manifest sha256
(4dd919fc... = the manifest of the 8B 6x6 runs), 13,470 manifest rows and that every image path of the manifest exists as a non-empty file.
Exit 0 = verified; 1 = something differs (message on stdout); 10 = no data present at all (the 6x6 setup would download it); any other code (e.g. 2 = python cannot open this file) is an error."""
import argparse, hashlib, json, sys
from pathlib import Path

HF_SHA = "c1edb808d424a2fa7bc4a2e601d39b431b04acd5"
GH_COMMIT = "e4f1cfcd11ee0d0dfa8ee6a0a97c2c782ce21aba"
MANIFEST_SHA = "4dd919fc9ddfeee6b3b807bba79191614bf7a475bafebc401329497c2a61aa59"
SIBLINGS, JSONL_FILES, ROWS = 3685, 48, 13470

p = argparse.ArgumentParser()
p.add_argument("--repo", default=".")
p.add_argument("--allow-no-manifest", action="store_true", help="data present, manifest not built yet (queue_6x6.sh builds it): only SOURCE.json and files are checked")
a = p.parse_args()
repo = Path(a.repo).resolve()
raw = repo / "01_code/data/raw/mpr_gui_bench"
man = repo / "01_code/data/manifests/mpr_gui_manifest.jsonl"
if not (raw / "SOURCE.json").is_file():
    print(f"no 6x6 raw data in {repo} (SOURCE.json missing)"); sys.exit(10)
problems = []
src = json.loads((raw / "SOURCE.json").read_text(encoding="utf-8"))
hf, gh = src.get("huggingface", {}), src.get("github", {})
if hf.get("sha") != HF_SHA: problems.append(f"Hugging Face revision {hf.get('sha')} != {HF_SHA}")
if hf.get("siblings") != SIBLINGS: problems.append(f"siblings {hf.get('siblings')} != {SIBLINGS}")
if hf.get("image_scope") != "all": problems.append(f"image_scope {hf.get('image_scope')} != all")
if gh.get("commit") != GH_COMMIT: problems.append(f"GitHub commit {gh.get('commit')} != {GH_COMMIT}")
if gh.get("jsonl_files") != JSONL_FILES: problems.append(f"jsonl_files {gh.get('jsonl_files')} != {JSONL_FILES}")
n_qas = len(list((raw / "qas").glob("*.jsonl")))
if n_qas != JSONL_FILES: problems.append(f"{n_qas} qas jsonl files on disk, expected {JSONL_FILES}")
if man.is_file():
    h = hashlib.sha256(man.read_bytes()).hexdigest()
    if h != MANIFEST_SHA: problems.append(f"manifest sha256 {h} != {MANIFEST_SHA}")
    rows = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) != ROWS: problems.append(f"manifest has {len(rows)} rows, expected {ROWS}")
    paths = {q for r in rows for q in r.get("image_paths", [])}
    missing = [q for q in sorted(paths) if not (repo / "01_code" / q).is_file() or (repo / "01_code" / q).stat().st_size == 0]
    if missing: problems.append(f"{len(missing)} of {len(paths)} manifest images missing or empty, e.g. {missing[:3]}")
    print(f"manifest rows {len(rows)}, distinct images {len(paths)}, missing {len(missing)}")
elif not a.allow_no_manifest:
    problems.append("manifest missing")
print(("OK: " if not problems else "PROBLEMS: ") + (f"{repo.name} raw data verified (HF {HF_SHA[:8]}, GitHub {GH_COMMIT[:8]}, siblings {SIBLINGS}, qas {n_qas})" if not problems else "; ".join(problems)))
sys.exit(1 if problems else 0)
