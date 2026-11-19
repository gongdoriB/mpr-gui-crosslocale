import sys, torch, shutil, os
from transformers import AutoConfig, AutoModel
src, dst = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
cfg._attn_implementation = "sdpa"
torch.manual_seed(0)
m = AutoModel.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa")
m.save_pretrained(dst)
for f in os.listdir(src):
    if not os.path.exists(os.path.join(dst,f)) and f!="config.json": shutil.copy(os.path.join(src,f), dst)
print(sum(p.numel() for p in m.parameters())/1e6, "M params"); print(os.listdir(dst))
