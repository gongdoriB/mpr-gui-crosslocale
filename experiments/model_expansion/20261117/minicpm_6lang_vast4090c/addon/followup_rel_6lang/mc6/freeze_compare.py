"""Compares `pip freeze` of the new venv with the pip freeze of the earlier MiniCPM run (mc6/reference/requirements.freeze.earlier_minicpm_20261116.txt, from
model_results_20261116/minicpm/unpacked/requirements.freeze.txt).
  python freeze_compare.py <reference freeze> <new freeze> <report out>
Key packages (the ones that decide model loading, preprocessing and numerics) must be identical: a difference exits 1 (the queue stops before any download).
Every other difference (version, missing, extra) is only listed in the report. The editable install line ("-e git+...") is ignored."""
import re, sys
from pathlib import Path

KEY = {"torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "tokenizers", "safetensors", "pillow", "numpy", "timm", "einops", "sentencepiece",
       "protobuf", "qwen-vl-utils", "pyyaml", "triton", "regex", "av"}


def parse(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(\S+)$", line)
        if not m:
            d[line] = "(unparsed line)"
            continue
        d[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return d


ref, new = parse(sys.argv[1]), parse(sys.argv[2])
key_diff, other_diff = [], []
for name in sorted(set(ref) | set(new)):
    a, b = ref.get(name), new.get(name)
    if a == b:
        continue
    (key_diff if name in KEY else other_diff).append(f"{name}: earlier {a} / now {b}")
lines = [f"packages earlier {len(ref)}, now {len(new)}, identical {sum(ref.get(k) == new.get(k) for k in set(ref) & set(new))}",
         f"KEY package differences ({len(key_diff)}):"] + ["  " + x for x in key_diff] + [f"other differences ({len(other_diff)}, report only):"] + ["  " + x for x in other_diff]
Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if key_diff else 0)
