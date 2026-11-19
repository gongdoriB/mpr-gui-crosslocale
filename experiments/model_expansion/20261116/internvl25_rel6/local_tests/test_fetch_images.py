"""fetch_images() of the 6-language data module without any network: urlopen is replaced by a reader of local files.
usage: python -I test_fetch_images.py <tree> <scratch dir>"""
import io, json, shutil, sys
from pathlib import Path
tree, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(tree / "01_code/src")); sys.dont_write_bytecode = True
from mpr_crosslocale.data import rel_followup_6lang as D
BAD = 0
def ck(d, ok):
    global BAD; print(("PASS " if ok else "FAIL ") + d); BAD += (not ok)
root = scratch / "fetch_root"; shutil.rmtree(root, ignore_errors=True); (root / "data").mkdir(parents=True)
src_root = tree / "followup_rel_6lang"
man = json.loads((src_root / "bundle_manifest.json").read_text())
shutil.copy(src_root / "bundle_manifest.json", root / "bundle_manifest.json")
own = [e for e in man["images"] if e["location"] == "own"]
by_url = {e["url"]: e for e in own}
calls = []
mode = {"corrupt": None, "fail_first": 0}
class R(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False
def fake_urlopen(url, timeout=60):
    calls.append(url)
    assert url.startswith("https://huggingface.co/datasets/chenruihan/MPR-GUI-Bench/resolve/c1edb808d424a2fa7bc4a2e601d39b431b04acd5/"), url
    if mode["fail_first"] > 0:
        mode["fail_first"] -= 1; raise OSError("transient")
    data = (src_root / by_url[url]["path"]).read_bytes()
    if mode["corrupt"] and url.endswith(mode["corrupt"]): data = data[:-5] + b"xxxxx"
    return R(data)
D.urlopen = fake_urlopen; D.time.sleep = lambda s: None
r = D.fetch_images(root, 4)
ck(f"downloads all 729 fr/ja images ({r})", r == {"downloaded": 729} and len(calls) == 729)
ck("every downloaded file has the manifest sha256", all(D.digest(root / e["path"]) == e["sha256"] for e in own))
ck("no legacy (en/zh/th/ru) image was requested or written", not any(f"/{l}/" in u for u in calls for l in ("en", "zh", "th", "ru")) and not any((root / e["path"]).exists() for e in man["images"] if e["location"] == "legacy"))
calls.clear(); r2 = D.fetch_images(root, 4)
ck("second call: all cached, no request", r2 == {"cached": 729} and not calls)
victim = root / own[0]["path"]; victim.write_bytes(b"broken"); calls.clear()
r3 = D.fetch_images(root, 4)
ck("a corrupted file is downloaded again", r3 == {"cached": 728, "downloaded": 1} and len(calls) == 1 and D.digest(victim) == own[0]["sha256"])
victim.unlink(); mode["corrupt"] = own[0]["url"].rsplit("/", 1)[-1]
try:
    D.fetch_images(root, 4); ck("a download with wrong content is rejected", False)
except ValueError as e:
    ck("a download with wrong content is rejected (checksum mismatch, no file kept)", "checksum mismatch" in str(e) and not victim.exists())
mode["corrupt"] = None; mode["fail_first"] = 2
r4 = D.fetch_images(root, 1)
ck("transient network errors are retried", r4 == {"downloaded": 1, "cached": 728})
bad = json.loads((root / "bundle_manifest.json").read_text()); next(e for e in bad["images"] if e["location"] == "own")["url"] = "https://evil.example/x.png"
(root / "bundle_manifest.json").write_text(json.dumps(bad))
try:
    D.fetch_images(root, 1); ck("an unexpected URL in the manifest is refused", False)
except ValueError as e:
    ck("an unexpected URL in the manifest is refused", "Unexpected image URL" in str(e))
print("failed checks:", BAD); sys.exit(1 if BAD else 0)
