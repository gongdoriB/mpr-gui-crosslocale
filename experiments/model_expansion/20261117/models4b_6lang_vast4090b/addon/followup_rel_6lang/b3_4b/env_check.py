"""Environment/identity check of one model (run from the repository root, venv active). Env: KEY, CONFIG, EXPA_REF, REL_REF (may be empty), CODE_REF.
  - model config (yaml) == model_config of the earlier 4-language run contract(s); english_fixed system prompt + template == the contract's
  - the repository's inference-code hashes (code_identity(), run_rel_T* excluded) == the earlier run's code_sha256
  - installed package versions == the earlier run's software_versions (torch, transformers, accelerate, torchvision, timm, sentencepiece, qwen-vl-utils, protobuf)
  - unless LOCAL_MOCK=1 or B2_PRE_MODEL=1: one RTX 4090 and the model snapshot in the local HF cache
Test hooks only: EXPECT_TORCH (replaces the torch version to compare), EXPECT_SKIP (comma list of packages not compared). Nothing is installed or changed."""
import importlib.metadata as md, json, os
from pathlib import Path
import yaml
from mpr_crosslocale.inference.run_rel_followup import code_identity
from mpr_crosslocale.inference.system_prompts import build_system_prompt

cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
for name in ("EXPA_REF", "REL_REF"):
    if not os.environ.get(name):
        continue
    ref = json.loads(Path(os.environ[name]).read_text())
    assert cfg == ref["model_config"], f"model config differs from {name} ({ref['model_config']['model_id']})"
    sp, _, tpl = build_system_prompt("english_fixed", "en")
    assert sp == ref["system_prompt"] and tpl == ref["prompt_template_version"], f"system prompt differs from {name}"
ref = json.loads(Path(os.environ["CODE_REF"]).read_text())
now = code_identity()
diff = sorted(k for k in set(now) | set(ref["code_sha256"]) if not k.startswith("run_rel_T") and ref["code_sha256"].get(k) != now.get(k))
assert not diff, f"inference code of the repository differs from the earlier {os.environ['KEY']} run: {diff}"
skip = os.environ.get("EXPECT_SKIP", "").split(",")
for pkg, want in ref["software_versions"].items():
    if pkg in ("python", "platform") or want is None or pkg in skip:
        continue
    if pkg == "torch" and os.environ.get("EXPECT_TORCH"):
        want = os.environ["EXPECT_TORCH"]
    have = md.version(pkg)
    assert have == want, f"{pkg} {have} differs from the earlier {os.environ['KEY']} run ({want}); this queue installs nothing"
print("model config, system prompt, inference-code hashes and package versions equal the earlier 4-language runs of", os.environ["KEY"])
if os.environ.get("LOCAL_MOCK") != "1" and not os.environ.get("B2_PRE_MODEL"):
    import torch
    name = torch.cuda.get_device_name(0)
    assert torch.cuda.device_count() == 1 and "4090" in name, name
    from huggingface_hub import snapshot_download
    snap = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    assert Path(snap, "config.json").is_file(), snap
    print("GPU", name, "| torch", torch.__version__, "| model", snap)
