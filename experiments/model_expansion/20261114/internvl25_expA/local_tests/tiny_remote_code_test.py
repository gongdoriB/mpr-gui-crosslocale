"""Local check, no real weights: do the PINNED remote code files of OpenGVLab/InternVL2_5-8B @ e9e4c0d load and generate under the
installed torch/transformers (server pins: torch 2.9.1, transformers 4.57.6), together with the mpr-ivl InternVL25Adapter
(ensure_generation_mixin, eager attention, first-token A-D scoring)? A random-weight model with tiny dimensions is built from the real config.
usage: python -I tiny_remote_code_test.py <SNAP dir with the small remote-code files> <REPO dir> <WORK dir> <image file>"""
import json, shutil, sys, types, importlib.util
from pathlib import Path

snap, repo, work, image = map(Path, sys.argv[1:5])
sys.path.insert(0, str(repo / "01_code/src"))
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
tiny = work / "tiny"
shutil.rmtree(tiny, ignore_errors=True); tiny.mkdir(parents=True)
for f in snap.iterdir():
    if f.suffix in (".py", ".json", ".model") and f.name != "config.json":
        shutil.copy(f, tiny / f.name)
cfg = json.loads((snap / "config.json").read_text())
cfg["llm_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attn_implementation="eager")
cfg["vision_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, qk_normalization=False,
                            use_flash_attn=False)
cfg["use_backbone_lora"] = 0; cfg["use_llm_lora"] = 0
(tiny / "config.json").write_text(json.dumps(cfg))
from transformers import AutoConfig, AutoModel
c = AutoConfig.from_pretrained(tiny, trust_remote_code=True)
torch.manual_seed(0)
m = AutoModel.from_config(c, trust_remote_code=True, torch_dtype=torch.float32)
m.save_pretrained(tiny)
print("tiny model params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 1))
del m
model = AutoModel.from_pretrained(tiny, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, use_flash_attn=False).eval()
print("language model class:", type(model.language_model).__name__, "| num_image_token:", model.num_image_token)

from mpr_crosslocale.models import internvl25 as mod
added = mod.ensure_generation_mixin(model)
print("ensure_generation_mixin added GenerationMixin:", added, "->", type(model.language_model).__name__)
# tokenizer from the small files (same code as check_tokenizer_prompt.py)
from tokenizers import AddedToken
spec = importlib.util.spec_from_file_location("tokenization_internlm2", snap / "tokenization_internlm2.py")
tm = importlib.util.module_from_spec(spec); sys.modules["tokenization_internlm2"] = tm; spec.loader.exec_module(tm)
tc = json.loads((snap / "tokenizer_config.json").read_text()); tc.pop("auto_map", None); tc.pop("tokenizer_class", None)
tc["added_tokens_decoder"] = {int(k): AddedToken(**v) for k, v in tc["added_tokens_decoder"].items()}
tok = tm.InternLM2Tokenizer(vocab_file=str(snap / "tokenizer.model"), **tc)

ad = mod.InternVL25Adapter.__new__(mod.InternVL25Adapter)
ad.dtype = torch.float32; ad.model, ad.tokenizer = model, tok
ad.input_size, ad.min_num, ad.max_num, ad.use_thumbnail = 448, 1, 7, True
from mpr_crosslocale.inference.system_prompts import build_system_prompt
sysmsg, _, _ = build_system_prompt("english_fixed", "en")
row = {"image_paths": [str(image)], "question_raw": "Which input is left of the button? A: Name B: Email C: Phone D: Date", "gold_label": "B"}
out = ad.generate_one(row, "mpr_label_only_v1", {"do_sample": False, "num_beams": 1, "max_new_tokens": 2}, system_prompt=sysmsg, score_labels=True)
print("raw_output:", repr(out.raw_output), "| tiles:", out.num_patches_list, "| visual tokens:", out.visual_token_count, "| prompt tokens:", out.prompt_token_count)
print("label logprobs:", {k: round(v, 3) for k, v in out.label_summary.label_logprobs.items()})
assert out.rendered_prompt.startswith("<|im_start|>system\n" + sysmsg) and "书生" not in out.rendered_prompt
assert out.visual_token_count == sum(out.num_patches_list) * 256
print("OK: pinned remote code + adapter run end to end with random tiny weights (plumbing only; output is meaningless)")
