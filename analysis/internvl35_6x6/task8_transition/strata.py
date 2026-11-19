import csv, json, sys, collections
R, OUT = sys.argv[1], sys.argv[2]
g = collections.defaultdict(dict)
for l in open(R + "/analysis_qwen3/raw/grounding_main_orcf_predictions.jsonl"):
    r = json.loads(l); g[r["pair_id"]][r["condition"]] = int(r["point_in_gt_box"])
rows=[]
for q in csv.DictReader(open(R + "/analysis_qwen3/rel4/quartet_contrasts.csv")):
    if q["pair_id"] in g:
        gg=g[q["pair_id"]]; rows.append(dict(item=q["parallel_id"],gui=q["gui_language"],qry=q["query_language"],gO=gg["O"],gR=gg["R"],mO=int(q["y_O"]),mR=int(q["y_R"])))
base=[r for r in rows if r["mO"]==0 and r["gR"]==1]
w=csv.writer(open(OUT+"/glmm_input.csv","w")); w.writerow(["item","gui","qry","gfix","mR"])
for r in base: w.writerow([r["item"],r["gui"],r["qry"],int(r["gO"]==0),r["mR"]])
out={}
for gui in ["en","zh","th","ru"]:
    s=[r for r in base if r["gui"]==gui]
    a=[r["mR"] for r in s if r["gO"]==0]; b=[r["mR"] for r in s if r["gO"]==1]
    out[gui]={"fixed":[round(sum(a)/len(a),3) if a else None,len(a)],"already_ok":[round(sum(b)/len(b),3) if b else None,len(b)]}
print(json.dumps(out,indent=1)); json.dump(out,open(OUT+"/by_gui.json","w"),indent=1)
