"""Pair-level link (Notion 261006 plan): do grounding O->R corrections coincide with REL O->R corrections?
Unit = item x (query, gui) pair, Qwen3, text-dependent REL items that also have grounding (strict parser)."""
import csv, json, sys, random, collections
R, OUT = sys.argv[1], sys.argv[2]
g = collections.defaultdict(dict)
for l in open(R + "/analysis_qwen3/raw/grounding_main_orcf_predictions.jsonl"):
    r = json.loads(l); g[r["pair_id"]][r["condition"]] = int(r["point_in_gt_box"])
rows = []
for q in csv.DictReader(open(R + "/analysis_qwen3/rel4/quartet_contrasts.csv")):
    if q["pair_id"] in g and len(g[q["pair_id"]]) == 4:
        gg = g[q["pair_id"]]
        rows.append(dict(item=q["parallel_id"], dep=q["text_dependency"], gui=q["gui_language"],
                         gO=gg["O"], gR=gg["R"], mO=int(q["y_O"]), mR=int(q["y_R"])))
def state(o, r): return {(0,0):"0->0",(0,1):"0->1",(1,0):"1->0",(1,1):"1->1"}[(o, r)]
res = {"n_pairs": len(rows), "n_items": len({r["item"] for r in rows}),
       "by_dependency": dict(collections.Counter(r["dep"] for r in rows))}
dep = [r for r in rows if r["dep"] == "dependent"]
tab = collections.Counter((state(r["gO"], r["gR"]), state(r["mO"], r["mR"])) for r in dep)
res["dependent_crosstab_grounding_x_mc"] = {f"{a} | {b}": n for (a, b), n in sorted(tab.items())}
def p_fix(sub):  # P(MC 0->1 | MC O wrong)
    s = [r for r in sub if r["mO"] == 0]
    return (sum(r["mR"] for r in s) / len(s) if s else None), len(s)
gfix = [r for r in dep if r["gO"] == 0 and r["gR"] == 1]
gstay = [r for r in dep if r["gO"] == 1 and r["gR"] == 1]
gfail = [r for r in dep if r["gR"] == 0]
res["mc_fix_rate_given_grounding_fixed"] = p_fix(gfix)
res["mc_fix_rate_given_grounding_already_ok"] = p_fix(gstay)
res["mc_fix_rate_given_grounding_R_wrong"] = p_fix(gfail)
# item-cluster bootstrap of difference: fixed vs already-ok
items = sorted({r["item"] for r in dep}); by = collections.defaultdict(list)
for r in dep: by[r["item"]].append(r)
def diff(sample):
    a = [r for it in sample for r in by[it] if r["gO"] == 0 and r["gR"] == 1 and r["mO"] == 0]
    b = [r for it in sample for r in by[it] if r["gO"] == 1 and r["gR"] == 1 and r["mO"] == 0]
    if not a or not b: return None
    return sum(r["mR"] for r in a)/len(a) - sum(r["mR"] for r in b)/len(b)
rng = random.Random(20261108); boots = []
for _ in range(5000):
    d = diff([rng.choice(items) for _ in items])
    if d is not None: boots.append(d)
boots.sort()
res["diff_fixed_minus_alreadyok"] = {"estimate": diff(items), "ci95": [boots[int(.025*len(boots))], boots[int(.975*len(boots))]], "reps_valid": len(boots), "seed": 20261108}
json.dump(res, open(OUT + "/transition_link.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
