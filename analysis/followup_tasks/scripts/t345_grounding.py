import json,re,sys,csv,collections,numpy as np
from scipy import stats
R="${TMPDIR}/work"
Q=R+"analysis_qwen3/"; O="${PROJECT_ROOT}/analysis_plan_followup/"
P=[json.loads(l) for l in open(Q+"raw/grounding_main_orcf_predictions.jsonl")]
STRICT=re.compile(r"\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)\Z")
sys.path.insert(0,R+"followup_rel_4lang/grounding_eval"); from point_format import point_in_box
def score(raw,w,h,gt):
    m=STRICT.fullmatch(raw)
    if m is None: return 0,0
    p=tuple(map(int,m.groups()))
    if not all(0<=v<=1000 for v in p): return 0,0
    return 1,int(point_in_box((p[0]*w/1000,p[1]*h/1000),gt))
H=collections.defaultdict(lambda: collections.defaultdict(dict)); V=collections.defaultdict(lambda: collections.defaultdict(dict)); AREA=collections.defaultdict(dict)
for r in P:
    w,h=r["image_size"]; v,hit=score(r["raw_output"],w,h,r["gt_bbox"])
    assert v==int(r["valid_output"]) and hit==int(r["point_in_gt_box"])
    k=(r["query_language"],r["gui_language"]); H[r["parallel_id"]][k][r["condition"]]=hit; V[r["parallel_id"]][k][r["condition"]]=v
    x0,y0,x1,y1=r["gt_bbox"]; AREA[r["parallel_id"]][r["gui_language"]]=(x1-x0)*(y1-y0)/(w*h)
items=sorted(H); B=10000; SEED=20261103
CON={"R-O":("R","O"),"C-O":("C","O"),"F-R":("F","R"),"F-C":("F","C")}
def con_scores(it,con,gui=None):
    a,b=CON[con]; return np.mean([pr[a]-pr[b] for k,pr in H[it].items() if gui is None or k[1]==gui])
def boot_mean(x,seed=SEED):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(B,len(x))); m=100*x[idx].mean(1)
    return 100*x.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
# ---------- Task 3
area_item={it:np.mean(list(AREA[it].values())) for it in items}
nvar=sum(len({round(v,9) for v in AREA[it].values()})>1 for it in items)
rat=[max(AREA[it].values())/min(AREA[it].values()) for it in items]
a=np.array([area_item[it] for it in items])
cuts=np.quantile(a,[1/3,2/3]); tert=np.digitize(a,cuts)  # 0,1,2
rows=[]; tinfo=[]
for t in range(3):
    sel=[i for i in range(len(items)) if tert[i]==t]
    tinfo.append(dict(tertile=t+1,n_items=len(sel),area_min=a[sel].min(),area_max=a[sel].max(),area_median=float(np.median(a[sel]))))
    for con in ["R-O","C-O","F-R","F-C"]:
        x=np.array([con_scores(items[i],con) for i in sel]); e,lo,hi=boot_mean(x)
        rows.append(dict(definition="item_mean_area",tertile=t+1,contrast=con,items=len(sel),estimate_pp=e,ci_low=lo,ci_high=hi,reps=B,seed=SEED))
    for c in "ORCF":
        hr=np.mean([np.mean([pr[c] for pr in H[items[i]].values()]) for i in sel]); vr=np.mean([np.mean([pr[c] for pr in V[items[i]].values()]) for i in sel])
        tinfo[-1]["hit_"+c]=hr; tinfo[-1]["valid_"+c]=vr
# difference tertile1 - tertile3 for R-O (independent resampling within tertile)
def diff13(con):
    s1=np.array([con_scores(items[i],con) for i in range(len(items)) if tert[i]==0]); s3=np.array([con_scores(items[i],con) for i in range(len(items)) if tert[i]==2])
    rng=np.random.default_rng(SEED); d=100*(s1[rng.integers(0,len(s1),(B,len(s1)))].mean(1)-s3[rng.integers(0,len(s3),(B,len(s3)))].mean(1))
    return 100*(s1.mean()-s3.mean()),np.percentile(d,2.5),np.percentile(d,97.5)
for con in ["R-O","C-O","F-R"]:
    e,lo,hi=diff13(con); rows.append(dict(definition="item_mean_area",tertile="T1(small)-T3(large)",contrast=con,items=len(items),estimate_pp=e,ci_low=lo,ci_high=hi,reps=B,seed=SEED))
# alternative: item x gui cell area tertile; each cell = 3 pairs; cluster bootstrap over items
cells=[(it,g) for it in items for g in ["en","zh","th","ru"]]
ca=np.array([AREA[it][g] for it,g in cells]); ct=np.digitize(ca,np.quantile(ca,[1/3,2/3]))
for t in range(3):
    for con in ["R-O","C-O","F-R"]:
        # per item sum & count of cells in tertile (pairs weighted equally; 3 pairs per cell)
        s=np.zeros(len(items));n=np.zeros(len(items))
        for (it,g),tt in zip(cells,ct):
            if tt==t:
                i=items.index(it); a_,b_=CON[con]
                for k,pr in H[it].items():
                    if k[1]==g: s[i]+=pr[a_]-pr[b_]; n[i]+=1
        rng=np.random.default_rng(SEED); idx=rng.integers(0,len(items),(B,len(items))); d=100*s[idx].sum(1)/n[idx].sum(1)
        rows.append(dict(definition="item_x_gui_cell_area",tertile=t+1,contrast=con,items=int((n>0).sum()),estimate_pp=100*s.sum()/n.sum(),ci_low=np.percentile(d,2.5),ci_high=np.percentile(d,97.5),reps=B,seed=SEED,pairs=int(n.sum())))
def wr(path,rows):
    keys=[]; [keys.append(k) for r in rows for k in r if k not in keys]
    w=csv.DictWriter(open(path,"w"),fieldnames=keys);w.writeheader();w.writerows(rows)
wr(O+"task3/grounding_by_area.csv",rows); wr(O+"task3/area_tertile_info.csv",tinfo)
json.dump(dict(items=len(items),items_with_area_varying_across_gui=nvar,median_max_over_min_area_ratio=float(np.median(rat)),max_ratio=float(max(rat)),cut_item_mean_area=cuts.tolist(),cell_cut=np.quantile(ca,[1/3,2/3]).tolist()),open(O+"task3/area_summary.json","w"),indent=1)
# ---------- Task 4
Qr=list(csv.DictReader(open(Q+"rel4/quartet_contrasts.csv")))
rel={(q["parallel_id"],q["query_language"],q["gui_language"]):(int(q["R_minus_O"]),int(q["y_R"])) for q in Q_ if False} if False else None
rel={(q["parallel_id"],q["query_language"],q["gui_language"]):(int(q["R_minus_O"]),int(q["y_R"]),int(q["y_O"])) for q in Qr if q["text_dependency"]=="dependent" and q["model"]=="qwen3"}
G=[];M=[];HR=[];MR=[]
for it in items:
    ks=sorted(k for k in H[it] if (it,)+k in rel); assert len(ks)==12
    G.append([H[it][k]["R"]-H[it][k]["O"] for k in ks]); M.append([rel[(it,)+k][0] for k in ks])
    HR.append([H[it][k]["R"] for k in ks]); MR.append([rel[(it,)+k][1] for k in ks])
G=np.array(G,float);M=np.array(M,float);HR=np.array(HR);MR=np.array(MR)
g=G.mean(1);m=M.mean(1); r_obs=np.corrcoef(g,m)[0,1]
assert abs(r_obs-0.0394)<5e-4, r_obs
rng=np.random.default_rng(42); nI=len(items)
def split_half(X,S,rng):
    hs=np.empty((S,nI,2))
    for s in range(S):
        perm=np.argsort(rng.random((nI,12)),axis=1)   # independent random 6/6 split per item
        Xp=np.take_along_axis(X,perm,axis=1); hs[s,:,0]=Xp[:,:6].mean(1); hs[s,:,1]=Xp[:,6:].mean(1)
    return hs
def sb_from(hs):
    a=hs[...,0];b=hs[...,1]; ac=a-a.mean(-1,keepdims=True); bc=b-b.mean(-1,keepdims=True)
    with np.errstate(all="ignore"): r=(ac*bc).sum(-1)/np.sqrt((ac**2).sum(-1)*(bc**2).sum(-1))
    return r,2*r/(1+r)
S=2000
hx=split_half(G,S,rng); hy=split_half(M,S,rng)
rx,sx=sb_from(hx); ry,sy=sb_from(hy)
def summ(v): v=v[~np.isnan(v)]; return dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()),p2_5=float(np.percentile(v,2.5)),p97_5=float(np.percentile(v,97.5)),n=int(len(v)))
relx=float(np.nanmean(sx)); rely=float(np.nanmean(sy))
dis=r_obs/np.sqrt(relx*rely)
# bootstrap with reliabilities refit on resampled items (use first 200 stored splits)
S2=200; hx2=hx[:S2]; hy2=hy[:S2]
rb=np.random.default_rng(SEED); dd=[];rr=[];rxs=[];rys=[]
for chunk in range(100):
    idx=rb.integers(0,nI,(100,nI))
    for b in range(100):
        ii=idx[b]; gi=g[ii];mi=m[ii]
        with np.errstate(all="ignore"):
            r=np.corrcoef(gi,mi)[0,1]
            _,sxb=sb_from(hx2[:,ii,:]); _,syb=sb_from(hy2[:,ii,:])
        rx_=np.nanmean(sxb); ry_=np.nanmean(syb); rr.append(r); rxs.append(rx_); rys.append(ry_)
        dd.append(r/np.sqrt(rx_*ry_) if rx_>0 and ry_>0 else np.nan)
dd=np.array(dd);rr=np.array(rr)
# plug-in CI: observed-r bootstrap CI divided by sqrt(relx*rely)
plug=[np.percentile(rr,2.5)/np.sqrt(relx*rely),np.percentile(rr,97.5)/np.sqrt(relx*rely)]
res=dict(items=nI,pairs_per_item=12,r_obs_pearson=r_obs,spearman=float(stats.spearmanr(g,m)[0]),
 grounding_item_score=dict(mean_pp=100*g.mean(),sd=float(g.std(ddof=1)),share_items_zero=float((g==0).mean())),
 mc_item_score=dict(mean_pp=100*m.mean(),sd=float(m.std(ddof=1)),share_items_zero=float((m==0).mean())),
 splithalf_reps=S,seed=42,grounding_r_half=summ(rx),grounding_SB=summ(sx),mc_r_half=summ(ry),mc_SB=summ(sy),
 disattenuated_r_plugin_point=float(dis),disattenuated_boot_reliability_refit=dict(point=float(dis),ci_low=float(np.nanpercentile(dd,2.5)),ci_high=float(np.nanpercentile(dd,97.5)),n_nan=int(np.isnan(dd).sum()),B=len(dd),seed=SEED,splits_used_per_boot=S2),
 disattenuated_ci_plugin_reliabilities=plug,
 observed_r_boot_ci=[float(np.percentile(rr,2.5)),float(np.percentile(rr,97.5))],
 ceiling=dict(share_items_hitR_all12=float((HR==1).all(1).mean()),n_items_hitR_all12=int((HR==1).all(1).sum()),mean_item_hitR=float(HR.mean()),
   share_items_MC_R_correct_all12=float((MR==1).all(1).mean()),n_items_MC_R_all12=int((MR==1).all(1).sum()),mean_item_MC_R=float(MR.mean())))
json.dump(res,open(O+"task4/task_link_reliability.json","w"),indent=1); print(json.dumps(res,indent=1))
# ---------- Task 5 (grounding side)
D=json.load(open(R+"followup_rel_4lang/grounding_eval/annotations_20261103/grounding_gt_draft_170.json"))["annotations"]
RS=json.load(open(R+"followup_rel_4lang/grounding_eval/annotations_20261103/audit/review_state.json"))["reviews"]
cnt=collections.Counter(); 
for it in items:
    for gui in ["en","zh","th","ru"]:
        a_=D[it][gui]; cnt[(a_["bbox_provenance"]["review_status"],a_["reference_equivalence"])]+=1
conf_cells={(it,gui) for it in items for gui in ["en","zh","th","ru"] if D[it][gui]["bbox_provenance"]["review_status"]=="confirmed"}
full_conf=[it for it in items if all((it,g_) in conf_cells for g_ in ["en","zh","th","ru"])]
rs_cnt=collections.Counter((v["annotation"]["bbox_provenance"]["review_status"],v["annotation"]["reference_equivalence"],v["resolution"]) for v in RS.values())
rs_main=sum(1 for k in RS if k.rsplit("::",1)[0] in set(items))
t5=dict(main_items=len(items),gt_cells=len(items)*4,gt_cell_review_status_x_refequiv={f"{a}|{b}":n for (a,b),n in cnt.items()},cells_bbox_confirmed=len(conf_cells),items_all4_confirmed=len(full_conf),
  review_state_json_records=len(RS),review_state_records_in_main158=rs_main,review_state_status_refequiv_resolution={"|".join(k):n for k,n in rs_cnt.items()})
# R-O on confirmed cells only (exploratory; confirmation = annotator confirmed own bbox; reference_equivalence is 'unreviewed' everywhere)
def cell_boot(cellset,con="R-O"):
    s=np.zeros(len(items));n=np.zeros(len(items)); a_,b_=CON[con]
    for i,it in enumerate(items):
        for k,pr in H[it].items():
            if (it,k[1]) in cellset: s[i]+=pr[a_]-pr[b_]; n[i]+=1
    rng=np.random.default_rng(SEED); idx=rng.integers(0,len(items),(B,len(items))); d=100*s[idx].sum(1)/np.maximum(n[idx].sum(1),1)
    return dict(contrast=con,items=int((n>0).sum()),pairs=int(n.sum()),estimate_pp=100*s.sum()/n.sum(),ci_low=float(np.percentile(d,2.5)),ci_high=float(np.percentile(d,97.5)))
allcells={(it,g_) for it in items for g_ in ["en","zh","th","ru"]}
t5["sens"]=[dict(subset="all_cells",**cell_boot(allcells,c)) for c in ["R-O","C-O","F-R"]]+[dict(subset="bbox_confirmed_cells",**cell_boot(conf_cells,c)) for c in ["R-O","C-O","F-R"]]+[dict(subset="bbox_not_confirmed_cells",**cell_boot(allcells-conf_cells,c)) for c in ["R-O","C-O","F-R"]]
json.dump(t5,open(O+"task5/review_counts_and_sensitivity.json","w"),indent=1,default=float); print(json.dumps(t5,indent=1,default=float))
print(open(O+"task3/area_summary.json").read()); print(open(O+"task3/grounding_by_area.csv").read()); print(open(O+"task3/area_tertile_info.csv").read())
