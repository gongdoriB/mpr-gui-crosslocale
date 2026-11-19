"""Builds the two 6-language modules from the unchanged 4-language originals by exact, asserted text replacements.
Local helper (not shipped). Every replacement must match exactly once; the originals are only read."""
import sys
from pathlib import Path

SRC = Path(sys.argv[1])   # .../01_code/src/mpr_crosslocale
OUT = Path(sys.argv[2])   # addon/01_code/src/mpr_crosslocale


def patch(text, pairs):
    for old, new in pairs:
        assert text.count(old) == 1, ("not exactly once", old[:80], text.count(old))
        text = text.replace(old, new)
    return text




TAIL = r'''def verify_bundle(root: Path, legacy_root: Path | None = None, *, images: bool = False) -> dict:
    """Check the data files and (optionally) every screenshot against bundle_manifest.json (sha256 and size).

    Each image entry says where it must be found: location 'own' = under root (fr/ja), 'legacy' = under legacy_root (en/zh/th/ru)."""
    manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    if manifest["schema"] != BUNDLE_SCHEMA:
        raise ValueError("Unknown bundle schema")
    bases = {"own": root, "legacy": legacy_root}
    for entry in manifest["files"] + (manifest["images"] if images else []):
        base = bases[entry.get("location", "own")]
        if base is None:
            raise ValueError(f"Legacy data root needed for {entry['path']}")
        path = safe_path(base, entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or digest(path) != entry["sha256"]
        ):
            raise ValueError(f"Bundle checksum mismatch: {entry['path']} ({entry.get('location', 'own')})")
    return manifest


def build_manifest(root: Path, legacy_root: Path, legacy_manifest: Path, out: Path) -> dict:
    """Local, one-time: write bundle_manifest.json for the 6-language data.

    en/zh/th/ru image sha256 values are taken from the 4-language bundle manifest used on the server AND re-computed from the local
    files; any disagreement stops the build. fr/ja sha256 values are computed from the local files (HF revision c1edb808)."""
    rows, report = validate_dataset(root, legacy_root, images=True, decode_images=True)
    image_paths = report.pop("_image_paths")
    legacy = {e["path"]: e for e in json.loads(legacy_manifest.read_text(encoding="utf-8"))["images"]}
    generation = json.loads((root / DATA_DIR / "generation_report.json").read_text(encoding="utf-8"))
    source = generation["source"]
    files = []
    for name in TRANSFER_FILES:
        relative = f"{DATA_DIR}/{name}"
        path = safe_path(root, relative)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    entries = []
    for name in sorted(image_paths):
        path = Path(image_paths[name])
        sha = digest(path)
        asset = name.removeprefix("assets/images/")
        language = name.split("/")[3]
        if language in LEGACY_LANGUAGES:
            ref = legacy.get(name)
            if ref is None or ref["sha256"] != sha or ref["bytes"] != path.stat().st_size:
                raise ValueError(f"Legacy image differs from the 4-language server manifest: {name}")
            location = "legacy"
        else:
            location = "own"
        url = f"https://huggingface.co/datasets/{source['hf_repo']}/resolve/{source['hf_revision']}/{quote(asset, safe='/')}"
        entries.append({"path": name, "location": location, "bytes": path.stat().st_size, "sha256": sha, "url": url})
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "files": files,
        "images": entries,
        "source": source,
        "audit": report,
        "images_by_location": dict(Counter(e["location"] for e in entries)),
    }
    write_json(out, manifest)
    return manifest


def fetch_images(root: Path, workers: int) -> dict:
    """Download the fr/ja screenshots (location 'own') from the pinned HF revision; every file is sha256-checked.
    The en/zh/th/ru screenshots (location 'legacy') are never touched here."""
    manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    source = manifest["source"]
    prefix = f"https://huggingface.co/datasets/{source['hf_repo']}/resolve/{source['hf_revision']}/"

    def fetch(entry):
        path = safe_path(root, entry["path"])
        expected_url = prefix + quote(entry["path"].removeprefix("assets/images/"), safe="/")
        if entry["url"] != expected_url or not expected_url.startswith("https://huggingface.co/"):
            raise ValueError("Unexpected image URL")
        if path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]:
            return "cached"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        for attempt in range(3):
            try:
                with urlopen(expected_url, timeout=60) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                if digest(temporary) != entry["sha256"]:
                    raise ValueError(f"Downloaded image checksum mismatch: {entry['path']}")
                temporary.replace(path)
                return "downloaded"
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    own = [e for e in manifest["images"] if e.get("location") == "own"]
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(fetch, own), 1):
            counts[result] += 1
            if index % 50 == 0:
                print(f"Images verified: {index}/{len(own)}", flush=True)
    return dict(counts)


def check_legacy_subset(rows: list[dict], legacy_conditions: Path) -> dict:
    """The en/zh/th/ru rows of the 6-language file must equal the 4-language file row by row (same order), except for the
    metadata fields in LEGACY_SUBSET_IGNORED_FIELDS that only the 6-language file carries."""
    legacy_rows = read_jsonl(legacy_conditions)
    subset = [r for r in rows if r["query_language"] in LEGACY_LANGUAGES and r["gui_language"] in LEGACY_LANGUAGES]
    if len(subset) != len(legacy_rows):
        raise ValueError(f"Legacy subset has {len(subset)} rows, the 4-language file {len(legacy_rows)}")
    ignored = set(LEGACY_SUBSET_IGNORED_FIELDS)
    extra_fields = Counter()
    for new, old in zip(subset, legacy_rows, strict=True):
        if {k: v for k, v in new.items() if k not in ignored} != {k: v for k, v in old.items() if k not in ignored}:
            raise ValueError(f"Row differs from the 4-language file: {new['pair_id']}::{new['condition']}")
        extra_fields.update(set(new) - set(old))
    return {"rows": len(subset), "ignored_fields_only_in_6lang": dict(extra_fields), "legacy_sha256": digest(legacy_conditions)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "build-manifest", "verify", "fetch-images", "check-legacy-subset"))
    parser.add_argument("--data-root", type=Path, default=Path("followup_rel_6lang"))
    parser.add_argument("--legacy-data-root", type=Path, default=None, help="default: <data-root>/../followup_rel_4lang")
    parser.add_argument("--legacy-manifest", type=Path, help="4-language bundle_manifest.json (build-manifest only)")
    parser.add_argument("--legacy-conditions", type=Path, help="conditions_4lang.jsonl (check-legacy-subset)")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    legacy = (args.legacy_data_root or root.parent / "followup_rel_4lang").resolve()
    if args.action == "build-manifest":
        if not args.legacy_manifest:
            parser.error("--legacy-manifest is required")
        result = build_manifest(root, legacy, args.legacy_manifest, root / BUNDLE_MANIFEST)["audit"]
    elif args.action == "fetch-images":
        if not 1 <= args.workers <= 16:
            parser.error("--workers must be between 1 and 16")
        result = fetch_images(root, args.workers)
    elif args.action == "verify":
        result = verify_bundle(root, legacy, images=True)["audit"]
    elif args.action == "check-legacy-subset":
        if not args.legacy_conditions:
            parser.error("--legacy-conditions is required")
        rows, _ = validate_dataset(root, legacy)
        result = check_legacy_subset(rows, args.legacy_conditions)
    else:
        _, result = validate_dataset(root, legacy, images=True, decode_images=True)
        result.pop("_image_paths")
    if args.report_out:
        write_json(args.report_out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''

# ------------------------------------------------------------------ data module
t = (SRC / "data/rel_followup.py").read_text(encoding="utf-8")
t = patch(t, [
 ('"""Validate and transfer the frozen, generated REL O/R/C/F dataset."""',
  '''"""Validate the frozen, generated 6-language REL O/R/C/F dataset (en, zh, th, ru, fr, ja).

Copy of data/rel_followup.py (4 languages) with the changes listed in CHANGES_VS_4LANG below; the original module is untouched.
Images live in two roots: fr/ja screenshots in the 6-language data root, en/zh/th/ru screenshots in the existing 4-language
data root (LEGACY_ROOT), so the 4-language images are not copied twice."""'''),
 ('LANGUAGES = ("en", "zh", "th", "ru")\n',
  '''LANGUAGES = ("en", "zh", "th", "ru", "fr", "ja")
LEGACY_LANGUAGES = ("en", "zh", "th", "ru")
NEW_LANGUAGES = ("fr", "ja")
'''),
 ('DATA_DIR = "data/rel_all_366"\nBUNDLE_MANIFEST = "bundle_manifest.json"\nTRANSFER_FILES = (\n    "conditions_4lang.jsonl",\n    "quartets_4lang.jsonl",\n    "reference_mappings.jsonl",\n    "manifest_366_4lang.json",\n    "assets_required.json",\n    "config.json",\n    "generation_report.json",\n    "source_issues.json",\n    "README_KO.md",\n)\n',
  '''DATA_DIR = "data/rel_all_366"
CONDITIONS_FILE = "conditions_6lang.jsonl"
BUNDLE_MANIFEST = "bundle_manifest.json"
BUNDLE_SCHEMA = "rel-followup-bundle-6lang-v1"
TRANSFER_FILES = (
    "conditions_6lang.jsonl",
    "reference_mappings.jsonl",
    "manifest_366_6lang.json",
    "config.json",
    "generation_report.json",
    "source_issues.json",
)
# Expected content of the frozen 6-language dataset (followup_rel_6lang/data/rel_all_366/SHA256SUMS_6lang, 2026-11-15).
EXPECTED_CONDITIONS_SHA256 = "63101abfa53782b78d3f2d3c5b2d23264288e650822734fc75027d1159f731ed"
EXPECTED_ROWS = 52572
EXPECTED_ITEMS = 366
EXPECTED_ITEMS_BY_DEPENDENCY = {"dependent": 302, "independent": 64}
EXPECTED_ROWS_BY_CONDITION = 13143
ROWS_PER_ITEM = 144          # 6 x 6 language pairs x 4 conditions
ROWS_PER_JA_EXCLUDED_ITEM = 100   # 5 x 5 pairs without ja x 4 conditions
# These three dependent items have no Japanese pairs (generation_report excluded_item_locales; README_KO.md of followup_rel_6lang;
# original indices 117, 172, 244 = Display & Brightness, Set reminder, Headphone Audio Levels).
JA_EXCLUDED_ITEMS = frozenset({"rel::2/setting_8.png", "rel::3/note_8.jpg", "rel::6/health_9.jpg"})
LEGACY_BUNDLE_MANIFEST = "bundle_manifest.json"   # 4-language data root; its image sha256 values are reused for en/zh/th/ru
# Fields that exist in the 6-language rows but not in the 4-language file used on the server (metadata only; prompts unaffected).
LEGACY_SUBSET_IGNORED_FIELDS = ("object_extraction",)

CHANGES_VS_4LANG = """
C1 reads conditions_6lang.jsonl and requires its sha256 to equal EXPECTED_CONDITIONS_SHA256 (new check, a lock).
C2 LANGUAGES has six entries (adds fr, ja).
C3 image files are looked up in the data root first and then in the legacy (4-language) data root; each must exist and be non-empty.
C4 'len(rows) == items * 64' is replaced by exact counts: 366 items, 52,572 rows, 13,143 rows per condition, 302/64 items by
   dependency, 144 rows per item and 100 rows for each of the three ja-excluded items.
C5 'every item has every language pair' now allows exactly one exception: the ja-excluded items have no pair in which ja is the
   query or the GUI language, and must have all 25 other pairs (new check).
C6 the reverse-pair text symmetry check is applied wherever the pair exists (a missing reverse pair is an error).
C7 (new) the screenshot folder language in image_path must equal the row's GUI language.
C8 pack/unpack/fetch/verify are replaced: verify_bundle checks every file and image by sha256 at the location given in the manifest
   (own = fr/ja, legacy = en/zh/th/ru); fetch-images downloads only fr/ja; build-manifest and check-legacy-subset are new.
Everything else (field reconstruction from the reference spans, condition language rules, equivalence metadata, quartet
invariants, source-issue eligibility, matched quartet identical) is unchanged.
"""
'''),
 # safe_path stays; images located via two roots
 ('def validate_dataset(root: Path, *, images: bool = False, decode_images: bool = False) -> tuple:\n    """Check all quartets and reconstruct every field from immutable reference spans."""\n    rows = read_jsonl(root / DATA_DIR / "conditions_4lang.jsonl")\n',
  '''def locate_image(root: Path, legacy_root: Path | None, name: str) -> Path | None:
    """Path of a screenshot: own root first (fr/ja), then the legacy 4-language root (en/zh/th/ru); None if absent."""
    for base in (root, legacy_root):
        if base is not None:
            candidate = safe_path(base, name)
            if candidate.is_file():
                return candidate
    return None


def validate_dataset(
    root: Path, legacy_root: Path | None = None, *, images: bool = False, decode_images: bool = False
) -> tuple:
    """Check all quartets and reconstruct every field from immutable reference spans (6 languages)."""
    conditions_path = root / DATA_DIR / CONDITIONS_FILE
    conditions_sha = digest(conditions_path)
    if conditions_sha != EXPECTED_CONDITIONS_SHA256:
        raise ValueError(f"conditions_6lang.jsonl sha256 {conditions_sha} differs from the locked {EXPECTED_CONDITIONS_SHA256}")
    rows = read_jsonl(conditions_path)
'''),
 ('    if len(maps) != len(mappings) or not maps:\n        raise ValueError("Reference mappings must contain unique, nonempty item IDs")\n',
  '    if len(maps) != len(mappings) or len(maps) != EXPECTED_ITEMS:\n        raise ValueError("Reference mappings must contain 366 unique item IDs")\n    if not JA_EXCLUDED_ITEMS <= set(maps):\n        raise ValueError("ja-excluded items are not in the reference mappings")\n'),
 ('        image = safe_path(root, row["image_path"])\n        if not row["image_path"].startswith("assets/images/"):\n            raise ValueError("Image is outside assets/images")\n        image_names.add(row["image_path"])\n        if images and (not image.is_file() or not image.stat().st_size):\n            raise ValueError(f"Missing/empty image: {image}")\n',
  '        safe_path(root, row["image_path"])\n        if not row["image_path"].startswith("assets/images/"):\n            raise ValueError("Image is outside assets/images")\n        if row["image_path"].split("/")[3] != g:\n            raise ValueError(f"Image language folder differs from the GUI language: {row[\'image_path\']}")\n        image_names.add(row["image_path"])\n'),
])
t = patch(t, [
 ('    unique_calls = 0\n    for pid in maps:\n        for q in LANGUAGES:\n            for g in LANGUAGES:\n                quartet = groups[pid, q, g]\n                if set(quartet) != set(CONDITIONS):\n                    raise ValueError(f"Incomplete quartet: {pid}/{q}/{g}")\n',
  '''    unique_calls = 0
    for pid in maps:
        for q in LANGUAGES:
            for g in LANGUAGES:
                if pid in JA_EXCLUDED_ITEMS and "ja" in (q, g):
                    if (pid, q, g) in groups:
                        raise ValueError(f"ja pair present for a ja-excluded item: {pid}/{q}/{g}")
                    continue
                if (pid, q, g) not in groups:
                    raise ValueError(f"Missing language pair: {pid}/{q}/{g}")
                quartet = groups[pid, q, g]
                if set(quartet) != set(CONDITIONS):
                    raise ValueError(f"Incomplete quartet: {pid}/{q}/{g}")
'''),
 ('                reverse = groups[pid, g, q]\n', '                if (pid, g, q) not in groups:\n                    raise ValueError(f"Reverse pair missing: {pid}/{g}/{q}")\n                reverse = groups[pid, g, q]\n'),
 ('    if len(rows) != len(maps) * 64:\n        raise ValueError("Unexpected total row count")\n    if decode_images:\n        from PIL import Image\n\n        for name in sorted(image_names):\n            with Image.open(safe_path(root, name)) as image:\n                image.verify()\n            with Image.open(safe_path(root, name)) as image:\n                image.load()\n',
  '''    per_item = Counter(r["parallel_id"] for r in rows)
    for pid in maps:
        expected_rows = ROWS_PER_JA_EXCLUDED_ITEM if pid in JA_EXCLUDED_ITEMS else ROWS_PER_ITEM
        if per_item[pid] != expected_rows:
            raise ValueError(f"Item {pid} has {per_item[pid]} rows, expected {expected_rows}")
    if len(rows) != EXPECTED_ROWS or sum(per_item.values()) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected total row count {len(rows)} (expected {EXPECTED_ROWS})")
    by_condition = Counter(r["condition"] for r in rows)
    if any(by_condition[c] != EXPECTED_ROWS_BY_CONDITION for c in CONDITIONS):
        raise ValueError(f"Unexpected rows per condition: {dict(by_condition)}")
    by_dependency = Counter({r["parallel_id"]: r["text_dependency"] for r in rows}.values())
    if dict(by_dependency) != EXPECTED_ITEMS_BY_DEPENDENCY:
        raise ValueError(f"Unexpected items by dependency: {dict(by_dependency)}")
    image_paths = {}
    for name in sorted(image_names):
        found = locate_image(root, legacy_root, name)
        if found is not None and found.stat().st_size:
            image_paths[name] = found
        elif images:
            raise ValueError(f"Missing/empty image (searched {root} and {legacy_root}): {name}")
    if decode_images:
        from PIL import Image

        for name in sorted(image_paths):
            with Image.open(image_paths[name]) as image:
                image.verify()
            with Image.open(image_paths[name]) as image:
                image.load()
'''),
 ('        "schema": "rel-followup-audit-v1",\n', '        "schema": "rel-followup-audit-6lang-v1",\n        "languages": list(LANGUAGES),\n        "ja_excluded_items": sorted(JA_EXCLUDED_ITEMS),\n        "rows_per_item": dict(Counter(per_item.values())),\n        "rows_by_condition": dict(by_condition),\n'),
 ('        "conditions_sha256": digest(root / DATA_DIR / "conditions_4lang.jsonl"),\n', '        "conditions_sha256": conditions_sha,\n        "images_located": len(image_paths),\n'),
 ('    return rows, report\n', '    report["_image_paths"] = {k: str(v) for k, v in image_paths.items()}\n    return rows, report\n'),
])
# report must stay JSON-serialisable and compact: keep the image map out of the written report

cut = t.index("def verify_bundle(")
t = t[:cut] + TAIL
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "data/rel_followup_6lang.py").write_text(t, encoding="utf-8")

# ------------------------------------------------------------------ runner
r = (SRC / "inference/run_rel_followup.py").read_text(encoding="utf-8")
r = patch(r, [
 ('"""Run frozen REL O/R/C/F inputs through the existing model adapters."""',
  '''"""Run frozen 6-language REL O/R/C/F inputs (en, zh, th, ru, fr, ja) through the existing model adapters.

Copy of inference/run_rel_followup.py with the changes listed in CHANGES_VS_4LANG (module data/rel_followup_6lang.py is used
instead of data/rel_followup.py). input_id, prompt building, label scoring, answer parsing and the system prompt are unchanged."""'''),
 ('from mpr_crosslocale.data.rel_followup import (\n    BUNDLE_MANIFEST,\n    CONDITIONS,\n    digest,\n    validate_dataset,\n    verify_bundle,\n    write_json,\n)\n',
  'from mpr_crosslocale.data.rel_followup_6lang import (\n    BUNDLE_MANIFEST,\n    CONDITIONS,\n    LANGUAGES,\n    NEW_LANGUAGES,\n    digest,\n    validate_dataset,\n    verify_bundle,\n    write_json,\n)\n'),
 ('def stable_hash(value) -> str:', '''CHANGES_VS_4LANG = """
R1 imports the 6-language data module (validation, bundle check).
R2 new --languages-subset {all,new_only}: new_only keeps rows whose query or GUI language is fr or ja (default all).
R3 new --legacy-data-root: second image root for en/zh/th/ru screenshots (default <data-root>/../followup_rel_4lang).
   Image paths handed to the model adapter come from the located files; the model input is the same file bytes as in the 4-language runs.
R4 the experiment fingerprint additionally contains languages, languages_subset, runner name and the legacy-root image sha list
   (through bundle_manifest_sha256); code_identity lists data/rel_followup_6lang.py instead of data/rel_followup.py.
R5 plan.json / run_contract.json record languages_subset and the language list.
Unchanged: input_id, select_rows, inference_plan (dedup), read_cache, predict (prompt, system prompt), parse, scoring, export_results.
"""


def stable_hash(value) -> str:'''),
 ('        package / "data/rel_followup.py",\n', '        package / "data/rel_followup_6lang.py",\n'),
 ('def predict(adapter, row: dict, args, config: dict, system_prompt: str) -> dict:',
  '''def filter_languages(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "new_only":
        return [r for r in rows if r["query_language"] in NEW_LANGUAGES or r["gui_language"] in NEW_LANGUAGES]
    raise ValueError(f"Unknown --languages-subset: {subset}")


def image_file(args, row: dict) -> str:
    located = getattr(args, "image_paths", {}).get(row["image_path"])
    return str(Path(located).resolve()) if located else str((args.data_root / row["image_path"]).resolve())


def predict(adapter, row: dict, args, config: dict, system_prompt: str) -> dict:'''),
])
assert r.count('"image_paths": [str((args.data_root / row["image_path"]).resolve())],') == 2
r = r.replace('"image_paths": [str((args.data_root / row["image_path"]).resolve())],', '"image_paths": [image_file(args, row)],')
r = patch(r, [
 ('    args.data_root = args.data_root.resolve()\n    rows, audit = validate_dataset(args.data_root, images=True)\n    if (args.data_root / BUNDLE_MANIFEST).exists():\n        verify_bundle(args.data_root, images=True)\n',
  '''    args.data_root = args.data_root.resolve()
    args.legacy_data_root = (
        args.legacy_data_root.resolve() if getattr(args, "legacy_data_root", None) else args.data_root.parent / "followup_rel_4lang"
    )
    args.languages_subset = getattr(args, "languages_subset", "all")
    rows, audit = validate_dataset(args.data_root, args.legacy_data_root, images=True)
    args.image_paths = audit.pop("_image_paths")
    if (args.data_root / BUNDLE_MANIFEST).exists():
        verify_bundle(args.data_root, args.legacy_data_root, images=True)
'''),
 ('    rows = select_rows(rows, args.population, args.max_items, args.seed)\n    num_shards',
  '    rows = filter_languages(rows, args.languages_subset)\n    rows = select_rows(rows, args.population, args.max_items, args.seed)\n    num_shards'),
 ('    identity = {\n        "dataset_sha256": audit["conditions_sha256"],\n',
  '    identity = {\n        "experiment": "rel_orcf_6lang_v1",\n        "languages": list(LANGUAGES),\n        "languages_subset": args.languages_subset,\n        "dataset_sha256": audit["conditions_sha256"],\n'),
 ('    plan = {\n        "items": len({r["parallel_id"] for r in rows}),\n', '    plan = {\n        "languages": list(LANGUAGES),\n        "languages_subset": args.languages_subset,\n        "items": len({r["parallel_id"] for r in rows}),\n'),
 ('    parser.add_argument("--data-root", type=Path, default=Path("followup_rel_4lang"))\n',
  '    parser.add_argument("--data-root", type=Path, default=Path("followup_rel_6lang"))\n    parser.add_argument("--legacy-data-root", type=Path, help="4-language data root holding the en/zh/th/ru screenshots (default: <data-root>/../followup_rel_4lang)")\n    parser.add_argument("--languages-subset", choices=("all", "new_only"), default="all", help="new_only: rows whose query or GUI language is fr or ja")\n'),
])
(OUT / "inference").mkdir(parents=True, exist_ok=True)
(OUT / "inference/run_rel_followup_6lang.py").write_text(r, encoding="utf-8")
print("ok")
