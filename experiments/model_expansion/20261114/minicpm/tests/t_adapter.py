import sys, json
from mpr_crosslocale.models.minicpmv import MiniCPMVAdapter
from mpr_crosslocale.inference.system_prompts import build_system_prompt
a = MiniCPMVAdapter(model_id=sys.argv[1], dtype="float32", device_map="cpu")
row = {"question_raw":"Which element is at the top? A: x B: y C: z D: w","gold_label":"A","image_paths":[sys.argv[2]]}
sp = build_system_prompt("english_fixed","en")[0]
for sc in (True, False):
    o = a.generate_one(row, "mpr_label_only_v1", {"do_sample":False,"num_beams":1,"max_new_tokens":2,"temperature":None}, system_prompt=sp, score_labels=sc)
    print(repr(o.raw_output), o.prompt_token_count, o.output_token_count, o.num_patches_list, o.visual_token_count, o.label_summary and (o.label_summary.scored_predicted_label, o.label_summary.label_token_ids))
print(o.rendered_prompt[:400])
print(a.model.generation_config if hasattr(a.model,'generation_config') else None)
