import json, re, sys, csv, numpy as np, collections
from scipy import stats
sys.path.insert(0,"${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference/followup_rel_4lang/grounding_eval")
from point_format import point_in_box
R="${PROJECT_ROOT}/받은자료_20261109/실험결과/mpr-gui-crosslocale-followup-rel-4lang-inference"; O="${PROJECT_ROOT}/analysis_internvl35_6x6_20261112/grounding"
P=[json.loads(l) for l in open(O+"/raw/grounding_main_orcf_predictions.jsonl")]
STRICT=re.compile(r"\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)\Z")   # as run_qwen3_prompt_pilot.POINT
BARE=re.compile(r"\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\Z")
def score(raw,w,h,gt,lenient):
    m=STRICT.fullmatch(raw)
    if m is None and lenient: m=BARE.fullmatch(raw)
    if m is None: return 0,0
    p=tuple(map(int,m.groups()))
    if not all(0<=v<=1000 for v in p): return 0,0
    o=(p[0]*w/1000,p[1]*h/1000)
    return 1,int(point_in_box(o,gt))
mism=0; bare_n=0
for r in P:
    w,h=r["image_size"]
    v,hit=score(r["raw_output"],w,h,r["gt_bbox"],False)
    assert v==int(r["valid_output"]) and hit==int(r["point_in_gt_box"]), r["trial_id"]
    v2,hit2=score(r["raw_output"],w,h,r["gt_bbox"],True)
    r["s_valid"],r["s_hit"],r["l_valid"],r["l_hit"]=v,hit,v2,hit2
    bare_n+= (v==0 and v2==1)
print("strict recompute matches recorded for all",len(P),"rows; bare-accepted rows:",bare_n)
conds="ORCF"
out={"n_rows":len(P),"strict_recompute_matches_recorded":True,"bare_xy_rows_rescued":bare_n}
# validity per condition
vr=[]
for c in conds:
    rs=[r for r in P if r["condition"]==c]
    vr.append(dict(condition=c,n=len(rs),strict_valid=sum(r["s_valid"] for r in rs),lenient_valid=sum(r["l_valid"] for r in rs),strict_hit=sum(r["s_hit"] for r in rs),lenient_hit=sum(r["l_hit"] for r in rs),strict_hit_rate=round(sum(r["s_hit"] for r in rs)/len(rs),4),lenient_hit_rate=round(sum(r["l_hit"] for r in rs)/len(rs),4)))
out["validity_hits_by_condition"]=vr
# structures: item -> (query,gui) -> cond -> hit
def build(key):
    d=collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in P: d[r["parallel_id"]][(r["query_language"],r["gui_language"])][r["condition"]]=r[key]
    return d
items=sorted({r["parallel_id"] for r in P})
CON={"R-O":("R","O"),"C-O":("C","O"),"F-C":("F","C"),"F-R":("F","R")}
def item_means(d,con,gui=None):
    a,b=CON[con]; res=[]
    for it in items:
        vals=[pr[a]-pr[b] for k,pr in d[it].items() if (gui is None or k[1]==gui)]
        assert all(len(pr)==4 for pr in d[it].values())
        res.append(np.mean(vals))
    return np.array(res)
B=10000
def boot(x,seed=20261103):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(B,len(x)))
    m=100*x[idx].mean(1)
    return 100*x.mean(), np.percentile(m,[2.5,97.5]), np.percentile(m,[1.25,98.75])
rows=[]
for tag,key in [("strict","s_hit"),("lenient","l_hit")]:
    d=build(key)
    for con in CON:
        for gui in [None,"en","zh","th","ru"]:
            x=item_means(d,con,gui); e,ci95,ci975=boot(x)
            rows.append(dict(parse=tag,contrast=con,gui_language=gui or "all",items=len(x),pairs_per_item=12 if gui is None else 3,estimate_pp=round(e,3),ci95_low=round(ci95[0],3),ci95_high=round(ci95[1],3),ci975_low=round(ci975[0],3),ci975_high=round(ci975[1],3),reps=B,seed=20261103))
w=csv.DictWriter(open(O+"/grounding_contrasts.csv","w"),fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# pair-level 2x2 transitions O->R (strict, lenient)
for tag,key in [("strict","s_hit"),("lenient","l_hit")]:
    d=build(key); t=collections.Counter()
    for it in d:
        for pr in d[it].values(): t[(pr["O"],pr["R"])]+=1
    out["O_to_R_transitions_"+tag]={f"{a}->{b}":t[(a,b)] for a in (0,1) for b in (0,1)}
# accuracy by condition by gui
gb=[]
for tag,key in [("strict","s_hit"),("lenient","l_hit")]:
    for g in ["en","zh","th","ru"]:
        for c in conds:
            rs=[r for r in P if r["gui_language"]==g and r["condition"]==c]
            gb.append(dict(parse=tag,gui=g,condition=c,n=len(rs),hit_rate=round(sum(r[key] for r in rs)/len(rs),4),valid_rate=round(sum(r["s_valid" if tag=="strict" else "l_valid"] for r in rs)/len(rs),4)))
w=csv.DictWriter(open(O+"/grounding_by_gui_condition.csv","w"),fieldnames=list(gb[0]));w.writeheader();w.writerows(gb)
# E: task link
Q=list(csv.DictReader(open(O+"/rel4/quartet_contrasts.csv")))
rel={(q["parallel_id"],q["query_language"],q["gui_language"]):int(q["R_minus_O"]) for q in Q if q["text_dependency"]=="dependent"}
link=[]
for tag,key in [("strict","s_hit"),("lenient","l_hit")]:
    d=build(key); g=[];m=[];npairs=[]
    for it in items:
        vals=[(pr["R"]-pr["O"],rel[(it,)+k]) for k,pr in d[it].items() if (it,)+k in rel]
        if not vals: continue
        g.append(np.mean([v[0] for v in vals]));m.append(np.mean([v[1] for v in vals]));npairs.append(len(vals))
    g=np.array(g);m=np.array(m)
    pr_=stats.pearsonr(g,m)[0]; sp=stats.spearmanr(g,m)[0]
    rng=np.random.default_rng(20261103); n=len(g); idx=rng.integers(0,n,size=(B,n))
    bp=[];bs=[]
    gi=g[idx];mi=m[idx]
    gc=gi-gi.mean(1,keepdims=True);mc=mi-mi.mean(1,keepdims=True)
    with np.errstate(all="ignore"):
        bp=(gc*mc).sum(1)/np.sqrt((gc**2).sum(1)*(mc**2).sum(1))
    # spearman bootstrap via ranks of resampled (with ties -> average ranks)
    bs=np.array([stats.spearmanr(gi[i],mi[i])[0] for i in range(B)])
    link.append(dict(parse=tag,items=n,pairs_total=sum(npairs),mean_grounding_R_O_pp=round(100*g.mean(),3),mean_MC_R_O_pp=round(100*m.mean(),3),pearson=round(pr_,4),pearson_ci95=[round(float(np.nanpercentile(bp,2.5)),4),round(float(np.nanpercentile(bp,97.5)),4)],spearman=round(sp,4),spearman_ci95=[round(float(np.nanpercentile(bs,2.5)),4),round(float(np.nanpercentile(bs,97.5)),4)],reps=B,seed=20261103,nan_boot_pearson=int(np.isnan(bp).sum()),nan_boot_spearman=int(np.isnan(bs).sum())))
out["task_link"]=link
out["grounding_items"]=len(items); out["items_in_rel_dependent_quartets"]=len({k[0] for k in rel}&set(items))
json.dump(out,open(O+"/grounding_summary.json","w"),indent=1)
print(json.dumps(out,indent=1))
