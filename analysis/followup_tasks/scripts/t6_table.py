import pandas as pd, numpy as np
R="${TMPDIR}/work"
O="${PROJECT_ROOT}/analysis_plan_followup/"
rows=[]
d=pd.concat([pd.read_csv(R+"01_code/results/analysis/descriptive_bootstrap.csv"),pd.read_csv(R+"analysis_qwen3/results/descriptive_bootstrap_qwen3.csv")])
d=d[(d.outcome=="generation_correct")&(d.estimand=="matched_minus_mismatched")]
for _,r in d.iterrows():
    q=("6x6 matched bonus (all)" if r.scope=="overall" else "6x6 matched bonus dim="+r.level)
    rows.append((q,r.model,100*r.estimate,100*r.conf_low,100*r.conf_high,"pp","item bootstrap 5000 (existing)"))
b=pd.read_csv(O+"task6/rel_dependency_bonus_diff_item_bootstrap.csv")
for _,r in b.iterrows(): rows.append(("REL dependent-minus-independent matched bonus (raw)",r.model,r.diff_pp,r.lo,r.hi,"pp","item bootstrap 10000 seed42 (new, unadjusted)"))
glm={"qwen3":(0.07327609,0.03675991,0.11152344),"qwen":(0.12133774,0.07985848,0.16420508),"internvl":(0.12567546,0.08005911,0.17018511)}
for m,(e,l,h) in glm.items(): rows.append(("REL dependent-minus-independent matched bonus (GLMM-standardized)",m,100*e,100*l,100*h,"pp","GLMM, MVN 2000 (existing)"))
h=pd.read_csv(O+"task1/holm_ORCF.csv"); h=h[h.dependency=="dependent"]
for _,r in h.iterrows(): rows.append(("dependent "+r.contrast.replace("_minus_","-"),r.model,r.boot_est_pp,r.boot_lo,r.boot_hi,"pp","item bootstrap 10000 seed42"))
g=pd.read_csv(R+"analysis_qwen3/grounding_contrasts.csv"); g=g[(g.parse=="strict")&(g.gui_language=="all")&(g.contrast.isin(["R-O","C-O","F-R"]))]
for _,r in g.iterrows(): rows.append(("grounding "+r.contrast,"qwen3",r.estimate_pp,r.ci95_low,r.ci95_high,"pp","item bootstrap 10000 seed 20261103 (Qwen3 only)"))
t=pd.DataFrame(rows,columns=["quantity","model","estimate","ci_low","ci_high","unit","method"])
sg=lambda x:"+" if x>0 else "-"
ci=lambda r:"CI>0" if r.ci_low>0 else ("CI<0" if r.ci_high<0 else "CI includes 0")
t["sign"]=[sg(x) for x in t.estimate]; t["ci_class"]=[ci(r) for r in t.itertuples()]
agg=t.groupby("quantity",sort=False).agg(n_models=("model","nunique"),signs=("sign",lambda s:"".join(s)),all_same_sign=("sign",lambda s:s.nunique()==1),ci_classes=("ci_class",lambda s:" / ".join(s)))
agg["scope_note"]=["Qwen3 only (no grounding for other models)" if q.startswith("grounding") else "" for q in agg.index]
t.to_csv(O+"task6/sign_consistency_long.csv",index=False); agg.to_csv(O+"task6/sign_consistency_summary.csv")
pd.set_option("display.width",250,"display.max_colwidth",80); print(t.round(2).to_string()); print(agg.to_string())
