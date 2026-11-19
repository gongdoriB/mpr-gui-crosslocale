"""Fake HF caches for the OFFLINE b3 queue tests. Usage: make_fake_hf_b3.py <target dir for qwen3_vl_4b> <target dir for internvl35_4b>
Real small files of Qwen3-VL-4B-Instruct (pinned revision, from the local HF cache: config + tokenizer) and sparse safetensors of the true byte size (qwen3_vl_4b: 4,967,229,296 + 3,908,490,048 = 8,875,719,344, as listed in the local cache).
The InternVL3.5-4B cache uses the Qwen3-VL-4B tokenizer as a STAND-IN (A-D = 32..35 in both) and two sparse files of 4.7 GB (true bytes unknown: 확인 필요)."""
import json, shutil, sys
from pathlib import Path
src = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots").iterdir())
def mk(target, mid, rev, cfg, sizes):
    t = Path(target) / "hub" / ("models--" + mid.replace("/", "--")) / "snapshots" / rev
    shutil.rmtree(Path(target), ignore_errors=True); t.mkdir(parents=True)
    for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"): shutil.copy(src / n, t / n)
    (t / "config.json").write_text(json.dumps(cfg))
    for i, s in enumerate(sizes, 1):
        with open(t / f"model-0000{i}-of-0000{len(sizes)}.safetensors", "wb") as f: f.truncate(s)
mk(sys.argv[1], "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", json.load(open(src / "config.json")), [4_967_229_296, 3_908_490_048])
mk(sys.argv[2], "OpenGVLab/InternVL3_5-4B", "481f6e32467eab4e922ccd7fd6cf420441a62331", {"architectures": ["InternVLChatModel"], "llm_config": {"architectures": ["Qwen3ForCausalLM"]}}, [4_700_000_000, 4_700_000_000])
print("ok")
