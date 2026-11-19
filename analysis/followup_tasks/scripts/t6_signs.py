import json,csv,numpy as np,collections,pandas as pd
S="${TMPDIR}/work"
R=S+"zip_1108/mpr-gui-crosslocale-followup-rel-4lang-inference/"; O="${PROJECT_ROOT}/analysis_plan_followup/"
dep={r["parallel_id"]:r["text_dependency"] for r in csv.DictReader(open(R+"annotation/rel_text_dependency/outputs/rel_annotations_integrated_366.csv"))}
LANG=["en","zh","fr","ru","ja","th"]
# per item matched / mismatched mean (generation_correct) for REL
acc={}
def add(m,pid,q,g,y):
    acc.setdefault(m,{}).setdefault(pid,{})[(q,g)]=y
for m,f in [("qwen","ex6/qwen/results/raw/vast_6x6_qwen2_5_vl_7b.jsonl"),("internvl","ex6/internvl/results/raw/vast_6x6_internvl2_5_8b.jsonl")]:
    for l in open(S+f):
        if '"dimension": "rel"' not in l: continue
        r=json.loads(l)
        if r["dimension"]!="rel": continue
        add(m,r["parallel_id"],r["question_language"],r["gui_language"],int(r["parsed_generated_label"]==r["gold_label"]))
for r in csv.DictReader(open(R+"analysis_qwen3/derived/qwen3_glmm.csv")):
    if r["dimension"]=="rel": add("qwen3",r["parallel_id"],r["question_language"],r["gui_language"],int(r["generation_correct"]))
B=10000; rows=[]
for m in ["qwen3","qwen","internvl"]:
    bon={}
    for pid,d in acc[m].items():
        assert len(d)==36,(m,pid,len(d))
        mt=np.mean([v for (q,g),v in d.items() if q==g]); mm=np.mean([v for (q,g),v in d.items() if q!=g]); bon[pid]=mt-mm
    assert set(bon)==set(dep),(m,len(bon))
    D=np.array([bon[p] for p in sorted(bon) if dep[p]=="dependent"]); I=np.array([bon[p] for p in sorted(bon) if dep[p]=="independent"])
    rng=np.random.default_rng(42)
    bd=D[rng.integers(0,len(D),(B,len(D)))].mean(1); bi=I[rng.integers(0,len(I),(B,len(I)))].mean(1); df=100*(bd-bi)
    rows.append(dict(model=m,n_dep=len(D),n_ind=len(I),dep_bonus_pp=100*D.mean(),ind_bonus_pp=100*I.mean(),diff_pp=100*(D.mean()-I.mean()),lo=np.percentile(df,2.5),hi=np.percentile(df,97.5),p_two_sided=min(1,2*min((df<=0).mean(),(df>=0).mean())),reps=B,seed=42))
t=pd.DataFrame(rows); t.to_csv(O+"task6/rel_dependency_bonus_diff_item_bootstrap.csv",index=False); print(t.round(3).to_string())
