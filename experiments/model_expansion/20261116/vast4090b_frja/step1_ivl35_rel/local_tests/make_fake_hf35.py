"""Fake HF cache for OpenGVLab/InternVL3_5-8B @ pinned revision for OFFLINE queue tests: Qwen3-VL-4B tokenizer files as STAND-IN (A-D = 32..35 like the real tokenizer),
config.json with InternVLChatModel, 4 sparse safetensors adding up to 17,070,000,000 bytes. Usage: make_fake_hf35.py <target dir>"""
import json, shutil, sys
from pathlib import Path
rev = "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"
src = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots").iterdir())
t = Path(sys.argv[1]) / "hub/models--OpenGVLab--InternVL3_5-8B/snapshots" / rev
shutil.rmtree(Path(sys.argv[1]), ignore_errors=True); t.mkdir(parents=True)
for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
    shutil.copy(src / n, t / n)
(t / "config.json").write_text(json.dumps({"architectures": ["InternVLChatModel"], "llm_config": {"architectures": ["Qwen3ForCausalLM"]}}))
for i, s in enumerate([4_400_000_000, 4_400_000_000, 4_400_000_000, 3_870_000_000], 1):
    with open(t / f"model-0000{i}-of-00004.safetensors", "wb") as f: f.truncate(s)
print("fake hf cache", t)
