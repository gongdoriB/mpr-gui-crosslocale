"""Builds a fake Hugging Face cache for Qwen/Qwen3-VL-8B-Instruct @ the pinned revision for the OFFLINE queue tests: real small files taken from the locally cached
Qwen3-VL-4B-Instruct snapshot (tokenizer files: STAND-IN for the 8B tokenizer, config.json rewritten to the 8B architecture fields), and four SPARSE safetensors files
whose sizes add up to 17,534,000,000 bytes (the real total is 확인 필요; the queue checks a 16.5-18.5 GB range). Usage: make_fake_hf.py <target dir>"""
import json, shutil, sys
from pathlib import Path
rev = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
src = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots").iterdir())
t = Path(sys.argv[1]) / "hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" / rev
shutil.rmtree(Path(sys.argv[1]), ignore_errors=True)
t.mkdir(parents=True)
for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"):
    shutil.copy(src / n, t / n)
cfg = json.loads((src / "config.json").read_text())
cfg["architectures"] = ["Qwen3VLForConditionalGeneration"]; cfg["model_type"] = "qwen3_vl"
(t / "config.json").write_text(json.dumps(cfg))
sizes = [4_400_000_000, 4_400_000_000, 4_400_000_000, 4_334_000_000]
for i, s in enumerate(sizes, 1):
    with open(t / f"model-0000{i}-of-00004.safetensors", "wb") as f:
        f.truncate(s)
print("fake hf cache:", t, sum(sizes))
