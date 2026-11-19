"""Fake HF caches for the OFFLINE b2 queue tests. Usage: make_fake_hf_b2.py <target dir for ivl35> <target dir for uitars>
(the InternVL2.5 fake cache already exists: real small files of the pinned revision + sparse safetensors of the true size). Tokenizers are Qwen3-VL-4B STAND-INS (A-D = 32..35 as in the real tokenizers)."""
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
mk(sys.argv[1], "OpenGVLab/InternVL3_5-8B", "9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79", {"architectures": ["InternVLChatModel"], "llm_config": {"architectures": ["Qwen3ForCausalLM"]}}, [4_400_000_000, 4_400_000_000, 4_400_000_000, 3_870_000_000])
mk(sys.argv[2], "ByteDance-Seed/UI-TARS-1.5-7B", "683d002dd99d8f95104d31e70391a39348857f4e", {"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl"}, [8_300_000_000] * 3 + [8_280_000_000])
print("ok")
