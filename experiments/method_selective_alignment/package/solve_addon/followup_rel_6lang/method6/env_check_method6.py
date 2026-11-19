"""Environment/identity check of one model for the R_stem/R_opts 6-language run (repository root = cwd, venv active). Env: KEY, CONFIG, CONTRACT.
  - model config (yaml) == model_config of the earlier run contract; english_fixed system prompt + template == the contract's
  - the repository's inference-code hashes (code_identity()) == the earlier run's code_sha256 (keys of other runners run_rel_T*, run_rel_mixed*, run_rstem6* excluded)
  - installed package versions == the earlier run's software_versions (torch, transformers, accelerate, torchvision, timm, sentencepiece, qwen-vl-utils, protobuf)
  - unless LOCAL_MOCK=1 or PRE_MODEL=1: one RTX 4090 and the model snapshot in the local HF cache
Test hooks only: EXPECT_SKIP (comma list of packages not compared). Nothing is installed or changed."""
import importlib.metadata as md, json, os
from pathlib import Path
import yaml
from mpr_crosslocale.inference.run_rel_followup import code_identity
from mpr_crosslocale.inference.system_prompts import build_system_prompt

key = os.environ["KEY"]
cfg = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
ref = json.loads(Path(os.environ["CONTRACT"]).read_text())
assert cfg == ref["model_config"], f"model config differs from the earlier {key} run contract ({ref['model_config']['model_id']})"
sp, _, tpl = build_system_prompt("english_fixed", "en")
assert sp == ref["system_prompt"] and tpl == ref["prompt_template_version"], "system prompt differs from the earlier run contract"
now = code_identity()
other = ("run_rel_T", "run_rel_mixed", "run_rstem6")
diff = sorted(k for k in set(now) | set(ref["code_sha256"]) if not k.startswith(other) and ref["code_sha256"].get(k) != now.get(k))
assert not diff, f"inference code of the repository differs from the earlier {key} run: {diff}"
skip = os.environ.get("EXPECT_SKIP", "").split(",")
for pkg, want in ref["software_versions"].items():
    if pkg in ("python", "platform") or want is None or pkg in skip:
        continue
    have = md.version(pkg)
    assert have == want, f"{pkg} {have} differs from the earlier {key} run ({want})"
print("model config, system prompt, inference-code hashes and package versions equal the earlier run of", key)
if os.environ.get("LOCAL_MOCK") != "1" and not os.environ.get("PRE_MODEL"):
    import torch
    name = torch.cuda.get_device_name(0)
    assert torch.cuda.device_count() == 1 and "4090" in name, name
    from huggingface_hub import snapshot_download
    snap = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    assert Path(snap, "config.json").is_file(), snap
    print("GPU", name, "| torch", torch.__version__, "| model", snap)
