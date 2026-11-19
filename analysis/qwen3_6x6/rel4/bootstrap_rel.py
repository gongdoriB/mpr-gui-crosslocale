"""Item-cluster bootstrap (10,000, seed 42, percentile) for REL O/R/C/F contrasts, all 3 models.
Percentile index convention copied from followup_rel_4lang/analyze_rf_flips.py (int(q*(n-1)) on sorted draws)."""
import csv, json, numpy as np, collections
R="${TMPDIR}/work"; O="${TMPDIR}/work"
q=list(csv.DictReader(open(R+"/followup_rel_4lang/results/analysis_ready/quartet_contrasts.csv")))
q+=list(csv.DictReader(open(O+"/rel4/quartet_contrasts.csv")))
CON=["R_minus_O","C_minus_O","F_minus_C","F_minus_R","interaction"]
B=10000
def boot(rows, col):
    ids=sorted({r["parallel_id"] for r in rows}); ix={p:i for i,p in enumerate(ids)}
    s=np.zeros(len(ids)); n=np.zeros(len(ids))
    for r in rows: s[ix[r["parallel_id"]]]+=float(r[col]); n[ix[r["parallel_id"]]]+=1
    rng=np.random.default_rng(42); k=len(ids)
    draws=np.empty(B)
    idx=rng.integers(0,k,size=(B,k))
    draws=100*s[idx].sum(1)/n[idx].sum(1)
    d=np.sort(draws)
    est=100*s.sum()/n.sum()
    return est,d[int(0.025*(B-1))],d[int(0.975*(B-1))],len(ids),int(n.sum())
out=[]
models=sorted({r["model"] for r in q})
for m in models:
  for dep in ["dependent","independent"]:
    for grp,key in [("all",None)]+[(g,g) for g in ["en","zh","th","ru"]]:
      for c in (CON if grp=="all" else ["R_minus_O","F_minus_R"]):
        rows=[r for r in q if r["model"]==m and r["text_dependency"]==dep and (key is None or r["gui_language"]==key)]
        e,lo,hi,ni,nq=boot(rows,c)
        out.append(dict(model=m,dependency=dep,target_gui=grp,contrast=c,estimate_pp=round(e,3),ci_low_pp=round(lo,3),ci_high_pp=round(hi,3),items=ni,quartets=nq,reps=B,seed=42))
w=csv.DictWriter(open(O+"/rel4/bootstrap_rel_contrasts.csv","w"),fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for o in out:
    if o["target_gui"]=="all": print(o)
