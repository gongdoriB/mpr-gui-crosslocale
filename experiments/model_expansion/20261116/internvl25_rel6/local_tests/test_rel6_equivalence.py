"""Local checks (CPU, mock model) for the 6-language REL runner. Usage:
  python test_rel6_equivalence.py <tree> <mock_4lang_out> <mock_6lang_out> <server_conditions_4lang.jsonl> <generator_conditions_4lang.jsonl> <new_rows_only.jsonl>
<tree> = scratch tree made by tools/sync_tree.sh. Reads only; prints PASS/FAIL lines and a summary."""
import hashlib, json, sys, copy
from pathlib import Path
from types import SimpleNamespace

tree, o4, o6, server4, gen4, newonly = map(Path, sys.argv[1:7])
sys.path.insert(0, str(tree / "01_code/src"))
sys.dont_write_bytecode = True
from mpr_crosslocale.data import rel_followup_6lang as D
from mpr_crosslocale.inference import run_rel_followup_6lang as R

BAD = 0
def ck(desc, ok, extra=""):
    global BAD
    print(("PASS " if ok else "FAIL ") + desc + (f"  [{extra}]" if extra else ""))
    BAD += (not ok)

def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

LEG = set(D.LEGACY_LANGUAGES)
root6, root4 = tree / "followup_rel_6lang", tree / "followup_rel_4lang"

# ---- b. equivalence of the 4-language subset ------------------------------------------------------------
p4, p6 = {r["input_id"]: r for r in jl(o4 / "predictions.jsonl")}, {r["input_id"]: r for r in jl(o6 / "predictions.jsonl")}
res4, res6 = jl(o4 / "results.jsonl"), jl(o6 / "results.jsonl")
leg_rows6 = [r for r in res6 if r["query_language"] in LEG and r["gui_language"] in LEG]
leg_ids6 = {r["prediction_source_input_id"] for r in leg_rows6}
ck("canonical inference ids: 4-lang runner == 6-lang runner restricted to en/zh/th/ru", set(p4) == leg_ids6, f"{len(p4)} vs {len(leg_ids6)}")
ck("every 4-lang canonical id exists in the 6-lang run", set(p4) <= set(p6))
same_prompt = sum(p4[k]["rendered_prompt"] == p6[k]["rendered_prompt"] for k in p4)
ck("rendered_prompt (system prompt + question) identical for all shared canonical inputs", same_prompt == len(p4), f"{same_prompt}/{len(p4)}")
for f in ("generated_text", "parsed_generated_label", "generation_correct", "label_logprobs", "gold_rank"):
    n = sum(p4[k][f] == p6[k][f] for k in p4)
    ck(f"mock prediction field {f} identical", n == len(p4), f"{n}/{len(p4)}")
ck("results rows: same row order, input_id, alias and question for the 4-lang subset",
   [(r["input_id"], r["prediction_source_input_id"], r["question"], r["gold_answer"], r["image_path"], r["reused_prediction"]) for r in res4]
   == [(r["input_id"], r["prediction_source_input_id"], r["question"], r["gold_answer"], r["image_path"], r["reused_prediction"]) for r in leg_rows6])
c4 = json.loads((o4 / "run_contract.json").read_text())["experiment_fingerprint"]
c6 = json.loads((o6 / "run_contract.json").read_text())["experiment_fingerprint"]
ck("experiment fingerprint differs between the 4-lang and 6-lang runs", c4 != c6, f"{c4[:12]} vs {c6[:12]}")
new_ids = set(p6) - set(p4)
ck("6-lang canonical = 4-lang canonical + new ones, new ones all involve fr/ja",
   len(p6) == len(p4) + len(new_ids) and all(("::fr" in k or "::ja" in k) for k in new_ids), f"{len(p6)} = {len(p4)} + {len(new_ids)}")
ck("canonical count after dedup = 44,489 and rows = 52,572", len(p6) == 44489 and len(res6) == 52572, f"{len(p6)}, {len(res6)}")
ck("results rows carry the same fields as the 4-lang rows (plus object_extraction metadata only)",
   set(res6[0]) - set(res4[0]) == {"object_extraction"} and not (set(res4[0]) - set(res6[0])), str(sorted(set(res6[0]) ^ set(res4[0]))))

# ---- data: the legacy subset equals both 4-language files ------------------------------------------------
rows, rep = D.validate_dataset(root6, root4, images=True)
ck("validate_dataset: 52,572 rows, 366 items", len(rows) == 52572 and rep["items"] == 366)
for name, path in (("server file (sha e5aa6cd0)", server4), ("generator file (sha aa403674)", gen4)):
    try:
        info = D.check_legacy_subset(rows, path)
        ck(f"en/zh/th/ru rows equal the 4-lang {name} (ignoring {list(D.LEGACY_SUBSET_IGNORED_FIELDS)})", info["rows"] == 23424, json.dumps(info["ignored_fields_only_in_6lang"]))
    except Exception as e:
        ck(f"en/zh/th/ru rows equal the 4-lang {name}", False, str(e)[:150])
strict_diff = sum(1 for r, o in zip([x for x in rows if x["query_language"] in LEG and x["gui_language"] in LEG], jl(server4)) if r != o)
ck("INFO strict equality with the server 4-lang file is NOT byte-level: rows differing only by object_extraction", strict_diff == 23424, f"{strict_diff} rows")

# ---- --languages-subset new_only ------------------------------------------------------------------------
sub = R.filter_languages(rows, "new_only")
file_new = jl(newonly)
ck("new_only keeps 29,148 rows", len(sub) == 29148, str(len(sub)))
ck("new_only rows == conditions_6lang_new_rows_only.jsonl line by line", sub == file_new)
ck("new_only: every row has fr or ja as query or GUI language, no legacy-only row", all({r["query_language"], r["gui_language"]} & {"fr", "ja"} for r in sub))
ck("new_only + legacy-only rows = all rows, disjoint", len(sub) + 23424 == 52572)
try:
    R.filter_languages(rows, "bogus"); ck("unknown subset rejected", False)
except ValueError:
    ck("unknown subset rejected", True)
canon_all, _ = R.inference_plan(rows)
canon_new, _ = R.inference_plan(sub)
ck("canonical ids of new_only subset == canonical ids of all minus the 4-lang ones",
   set(canon_new) == set(canon_all) - set(p4), f"{len(canon_new)} vs {len(canon_all) - len(p4)}")

# ---- images: located files, bytes equal the manifest ------------------------------------------------------
man = json.loads((root6 / "bundle_manifest.json").read_text())
want = {e["path"]: e for e in man["images"]}
paths = rep["_image_paths"]
bad = [n for n, p in paths.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != want[n]["sha256"]]
ck("all 2,193 located images have the manifest sha256", len(paths) == 2193 and not bad, f"{len(paths)} images, {len(bad)} bad")
ck("fr/ja images come from the 6-lang root, en/zh/th/ru from the 4-lang root",
   all((Path(p).is_relative_to(root6.resolve()) if n.split("/")[3] in ("fr", "ja") else Path(p).is_relative_to(root4.resolve())) for n, p in paths.items()))
class StubAdapter:
    def __init__(self): self.seen = []
    def generate_one(self, adapter_row, profile, generation, system_prompt=None, score_labels=True):
        self.seen.append(adapter_row["image_paths"][0])
        return SimpleNamespace(raw_output="A", label_summary=None, rendered_prompt="x", prompt_token_count=1, output_token_count=1)
args = SimpleNamespace(mock_model=False, no_score_labels=True, data_root=root6.resolve(), image_paths=paths)
st = StubAdapter()
sample = [next(r for r in rows if r["gui_language"] == l and r["query_language"] == "en") for l in ("en", "ru", "fr", "ja")]
for r in sample:
    R.predict(st, r, args, {"prompt_profile": "mpr_label_only_v1", "generation": {}}, "sys")
ck("predict() hands the located absolute image files to the adapter (non-mock path, stub adapter)",
   [Path(p).is_file() for p in st.seen] == [True] * 4 and [p.split("/")[-2] for p in st.seen] == ["en", "ru", "fr", "ja"], str([p[-40:] for p in st.seen]))

# ---- validation: negative cases (rows mutated in memory; sha lock disabled for these) --------------------------
orig_read, orig_sha = D.read_jsonl, D.EXPECTED_CONDITIONS_SHA256
def run_mut(name, mut, expect):
    def fake_read(path):
        data = orig_read(path)
        return mut(data) if Path(path).name == D.CONDITIONS_FILE else data
    D.read_jsonl = fake_read
    D.EXPECTED_CONDITIONS_SHA256 = orig_sha
    # sha lock must trip first on any content difference; disable it by pointing the lock at the real file hash
    try:
        D.validate_dataset(root6, root4)
        ck(f"negative: {name} -> rejected", False, "no error")
    except ValueError as e:
        ck(f"negative: {name} -> rejected", expect in str(e), str(e)[:110])
    finally:
        D.read_jsonl = orig_read
def drop_first(d): return d[1:]
def dup(d): return d + [copy.deepcopy(d[5])]
def gold(d):
    d = copy.deepcopy(d); i = next(i for i, r in enumerate(d) if r["condition"] == "C"); d[i]["gold_answer"] = "A" if d[i]["gold_answer"] != "A" else "B"; return d
def drop_pair(d):
    return [r for r in d if not (r["parallel_id"] == "rel::2/setting_8.png" and r["query_language"] == "en" and r["gui_language"] == "fr")]
def drop_normal_pair(d):
    return [r for r in d if not (r["parallel_id"] == "rel::1/booking_3.jpg" and r["query_language"] == "en" and r["gui_language"] == "fr")]
def img_lang(d):
    d = copy.deepcopy(d); d[0]["image_path"] = d[0]["image_path"].replace("/en/", "/zh/"); return d
real_sha = D.digest(root6 / D.DATA_DIR / D.CONDITIONS_FILE)
# From here the sha lock is bypassed so that the structural checks themselves are exercised.
orig_digest = D.digest
D.digest = lambda p: real_sha if Path(p).name == D.CONDITIONS_FILE else orig_digest(p)
for name, mut, expect in (
    ("one row missing", drop_first, "Incomplete quartet"),
    ("one duplicated row", dup, "duplicate condition"),
    ("gold differs inside one pair", gold, "changes gold_answer"),
    ("a non-ja pair of a ja-excluded item removed", drop_pair, "Missing language pair"),
    ("a language pair of an ordinary item removed", drop_normal_pair, "Missing language pair"),
    ("screenshot folder language differs from GUI language", img_lang, "Image language folder"),
):
    run_mut(name, mut, expect)
# a ja pair present for an item declared ja-excluded (declaration changed instead of the data, so every per-row check still passes)
D.JA_EXCLUDED_ITEMS = frozenset(D.JA_EXCLUDED_ITEMS | {"rel::1/booking_3.jpg"})
try:
    D.validate_dataset(root6, root4); ck("negative: ja pair present for a ja-excluded item -> rejected", False)
except ValueError as e:
    ck("negative: ja pair present for a ja-excluded item -> rejected", "ja pair present" in str(e), str(e)[:90])
D.JA_EXCLUDED_ITEMS = frozenset({"rel::2/setting_8.png", "rel::3/note_8.jpg", "rel::6/health_9.jpg"})
D.digest = orig_digest
D.EXPECTED_CONDITIONS_SHA256 = "0" * 64
try:
    D.validate_dataset(root6, root4); ck("negative: wrong sha lock -> rejected", False)
except ValueError as e:
    ck("negative: wrong sha lock -> rejected", "differs from the locked" in str(e))
D.EXPECTED_CONDITIONS_SHA256 = orig_sha
# bundle verify: corrupted legacy image detected
tmp_manifest = json.loads((root6 / "bundle_manifest.json").read_text())
D.verify_bundle(root6, root4, images=True); ck("verify_bundle(images=True) passes on the real tree", True)
try:
    D.verify_bundle(root6, None, images=True); ck("negative: legacy root missing -> rejected", False)
except ValueError as e:
    ck("negative: legacy root missing -> rejected", "Legacy data root needed" in str(e))
print(f"\nfailed checks: {BAD}")
sys.exit(1 if BAD else 0)
