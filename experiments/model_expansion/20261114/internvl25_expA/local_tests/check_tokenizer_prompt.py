"""Local check (no weights): A-D single tokens and the rendered MCQ prompt of OpenGVLab/InternVL2_5-8B @ e9e4c0d.
Uses only small files of the pinned revision (tokenizer.model, tokenizer_config.json, added_tokens.json, special_tokens_map.json,
tokenization_internlm2.py, conversation.py, config.json) placed in SNAP, and the (patched) mpr-ivl adapter in REPO.
usage: python -I check_tokenizer_prompt.py <SNAP dir> <REPO dir>"""
import importlib.util, json, sys, types
from pathlib import Path

snap, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "01_code/src"))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

from tokenizers import AddedToken
tok_mod = load("tokenization_internlm2", snap / "tokenization_internlm2.py")
cfg = json.loads((snap / "tokenizer_config.json").read_text())
cfg.pop("auto_map", None); cfg.pop("tokenizer_class", None)
cfg["added_tokens_decoder"] = {int(k): AddedToken(**v) for k, v in cfg.get("added_tokens_decoder", {}).items()}
tok = tok_mod.InternLM2Tokenizer(vocab_file=str(snap / "tokenizer.model"), **cfg)
ids = {l: tok.encode(l, add_special_tokens=False) for l in "ABCD"}
print("label ids:", ids)
assert all(len(v) == 1 for v in ids.values())
# ids recorded in the earlier server run (label_token_ids in predictions.jsonl, InternVL2.5-8B REL full run, 2026-10-05)
assert {k: v[0] for k, v in ids.items()} == {"A": 290, "B": 309, "C": 289, "D": 302}, ids
print("matches the label_token_ids of the earlier server run: A 290, B 309, C 289, D 302")
print("IMG_CONTEXT id:", tok.convert_tokens_to_ids("<IMG_CONTEXT>"), "| <|im_end|> id:", tok.convert_tokens_to_ids("<|im_end|>"))

conv = load("conversation", snap / "conversation.py")
config = json.loads((snap / "config.json").read_text())
tmpl = conv.get_conv_template(config["template"])
model = types.SimpleNamespace(conv_template=tmpl, system_message=tmpl.system_message, num_image_token=256,
                              device="cpu", img_context_token_id=None)
from mpr_crosslocale.models.internvl25 import InternVL25Adapter
from mpr_crosslocale.inference.system_prompts import build_system_prompt
ad = InternVL25Adapter.__new__(InternVL25Adapter)
ad.model, ad.tokenizer = model, tok
sysmsg, _, _ = build_system_prompt("english_fixed", "en")
row = {"image_paths": ["x.png"], "question_raw": "Which field is left of the button? A: Name B: Email C: Phone D: Date"}
q = ad.build_question(row, "mpr_label_only_v1")
for label, sp in (("english_fixed", sysmsg), ("None (model default)", None)):
    rendered, input_ids, mask, eos, sep = ad._encode_question(q, sp, (3,))
    print(f"--- system_prompt = {label}\n{rendered}\n[prompt tokens {input_ids.shape[1]} (3 tiles x 256 image tokens included), eos id {eos}, sep {sep!r}]")
    if sp is not None:
        assert sysmsg in rendered and "书生" not in rendered, "english_fixed did not replace the default system message"
    else:
        assert "书生" in rendered
rendered, *_ = ad._encode_question(q, sysmsg, (3,))
assert rendered.startswith("<|im_start|>system\n" + sysmsg + "<|im_end|>\n<|im_start|>user\n<image>\n")
print("OK: english_fixed replaces the default Chinese system message; A-D are single tokens")
