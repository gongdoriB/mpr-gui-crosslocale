"""Same estimator as analysis_qwen3_20261108/rel4/bootstrap_rel.py (10,000, seed 42, percentile int(q*(B-1))), InternVL3.5 only."""
import csv, numpy as np, sys
q=list(csv.DictReader(open(sys.argv[1])))
CON=["R_minus_O","C_minus_O","F_minus_C","F_minus_R","interaction"]; B=10000
def boot(rows,col):
    ids=sorted({r["parallel_id"] for r in rows}); ix={p:i for i,p in enumerate(ids)}
    s=np.zeros(len(ids)); n=np.zeros(len(ids))
    for r in rows: s[ix[r["parallel_id"]]]+=float(r[col]); n[ix[r["parallel_id"]]]+=1
    rng=np.random.default_rng(42); k=len(ids); idx=rng.integers(0,k,size=(B,k))
    d=np.sort(100*s[idx].sum(1)/n[idx].sum(1))
    return 100*s.sum()/n.sum(),d[int(0.025*(B-1))],d[int(0.975*(B-1))],len(ids),int(n.sum())
out=[]
for dep in ["dependent","independent"]:
  for grp in ["all","en","zh","th","ru"]:
    for c in (CON if grp=="all" else ["R_minus_O","F_minus_R"]):
      rows=[r for r in q if r["text_dependency"]==dep and (grp=="all" or r["gui_language"]==grp)]
      e,lo,hi,ni,nq=boot(rows,c)
      out.append(dict(model="internvl35",dependency=dep,target_gui=grp,contrast=c,estimate_pp=round(e,3),ci_low_pp=round(lo,3),ci_high_pp=round(hi,3),items=ni,quartets=nq,reps=B,seed=42))
w=csv.DictWriter(open(sys.argv[2],"w"),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for o in out: print(o["dependency"],o["target_gui"],o["contrast"],o["estimate_pp"],[o["ci_low_pp"],o["ci_high_pp"]])
