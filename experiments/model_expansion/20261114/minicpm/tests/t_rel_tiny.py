import sys, torch
from types import SimpleNamespace
torch.cuda.is_available = lambda: True
import mpr_crosslocale.inference.run_cross_locale as rcl
orig = rcl._build_model_adapter
rcl._build_model_adapter = lambda args, cfg: orig(SimpleNamespace(attn_implementation=None, device_map="cpu"), cfg)
import mpr_crosslocale.inference.run_rel_followup as rf
rf.main(sys.argv[1:])
