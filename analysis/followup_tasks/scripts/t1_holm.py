import numpy as np, pandas as pd, sys, os
P="${PROJECT_ROOT}/analysis_rel_pairs/"
R="${TMPDIR}/work"
O="${PROJECT_ROOT}/analysis_plan_followup/task1/"
q=pd.concat([pd.read_csv(P+'quartets_qwen3.csv'),pd.read_csv(P+'quartets_qwen25_internvl25.csv')])
# GLMM contrasts
g=pd.concat([pd.read_csv(R+'followup_rel_4lang/results/glmm/combined_contrasts.csv'),pd.read_csv(R+'analysis_qwen3/results/rel4_glmm/qwen3_contrasts.csv')])
CON=["R_minus_O","C_minus_O","F_minus_C","F_minus_R"]
B=10000
def holm(p):
    p=np.array(p);o=np.argsort(p);m=len(p);adj=np.empty(m);run=0
    for i,j in enumerate(o):
        run=max(run,(m-i)*p[j]);adj[j]=min(1,run)
    return adj
rows=[]
for m in ["qwen3","qwen","internvl"]:
  for dep in ["dependent","independent"]:
    s=q[(q.model==m)&(q.text_dependency==dep)]
    ids=sorted(s.parallel_id.unique()); ix={p:i for i,p in enumerate(ids)}
    k=len(ids); n=np.zeros(k); sums={c:np.zeros(k) for c in CON}
    for c in CON: sums[c]=s.groupby('parallel_id')[c].sum().reindex(ids).to_numpy(float)
    n=s.groupby('parallel_id').size().reindex(ids).to_numpy(float)
    rng=np.random.default_rng(42); idx=rng.integers(0,k,size=(B,k))   # same as bootstrap_rel.py (seed 42, fresh rng, B=10000)
    for c in CON:
        d=100*sums[c][idx].sum(1)/n[idx].sum(1); est=100*sums[c].sum()/n.sum()
        srt=np.sort(d)
        lo,hi=srt[int(0.025*(B-1))],srt[int(0.975*(B-1))]
        le=(d<=0).mean(); ge=(d>=0).mean(); p=min(1,2*min(le,ge))
        gr=g[(g.model==m)&(g.dependency==dep)&(g.contrast==c)].iloc[0]
        rows.append(dict(model=m,dependency=dep,contrast=c,items=k,quartets=int(n.sum()),boot_est_pp=est,boot_lo=lo,boot_hi=hi,boot_p_raw=p,boot_p_floor=("<%.4f"%(2/B)) if p==0 else "",
            glmm_OR=gr.odds_ratio,glmm_lo=gr.or_ci_low,glmm_hi=gr.or_ci_high,glmm_p_raw=gr.p_value_wald))
d=pd.DataFrame(rows)
d['boot_p_holm']=np.nan; d['glmm_p_holm']=np.nan
for m in d.model.unique():
    mk=(d.model==m)&(d.dependency=="dependent")
    d.loc[mk,'boot_p_holm']=holm(d.loc[mk,'boot_p_raw'].values)
    d.loc[mk,'glmm_p_holm']=holm(d.loc[mk,'glmm_p_raw'].values)
d.to_csv(O+'holm_ORCF.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(d.round(5).to_string())
# check vs previous
prev=pd.read_csv(R+'analysis_qwen3/rel4/bootstrap_rel_contrasts.csv')
for _,r in d.iterrows():
    pr=prev[(prev.model==r.model)&(prev.dependency==r.dependency)&(prev.target_gui=="all")&(prev.contrast==r.contrast)].iloc[0]
    assert abs(pr.estimate_pp-r.boot_est_pp)<2e-3 and abs(pr.ci_low_pp-r.boot_lo)<2e-3 and abs(pr.ci_high_pp-r.boot_hi)<2e-3,(r.model,r.contrast,pr.ci_low_pp,r.boot_lo)
print("matches previous bootstrap CIs")
