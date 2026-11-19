"""Own read-only streaming numerical audit; never imports or executes archived code."""
import collections as co, hashlib, json, pathlib, resource, zipfile
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
ZIP=OUT.parents[1]/'internvl35_prep.zip'
P='internvl35_prep/results_internvl35_20261109/'
META=['model_id','model_revision','processor_revision','code_commit','run_id','experiment_fingerprint','precision','attn_implementation','batch_size','processor_profile','software_versions','hardware','label_scoring_method']
CON=('O','R','C','F'); CONTRASTS=[('R','O'),('F','C'),('C','O'),('F','R')]
z=zipfile.ZipFile(ZIP)
def audit_stream(suffix, consumer):
 counts=co.Counter(); ids=set(); meta={k:co.Counter() for k in META}; ts=[]; h=hashlib.sha256(); runtime=[]; inference_runtime=[]; examples=[]
 with z.open(P+suffix) as f:
  for line in f:
   h.update(line); r=json.loads(line); counts['rows']+=1
   k=r.get('input_id'); counts['duplicate_input_ids']+=k in ids; ids.add(k)
   counts['status:'+str(r.get('status'))]+=1
   for field in ['parse_success','generation_scoring_disagreement','reused_prediction','source_issue','primary_analysis_eligible']:
    counts[field+':'+(str(bool(r[field])) if field in r else 'MISSING')]+=1
   gold=r.get('gold_label',r.get('gold_answer')); gen=r.get('parsed_generated_label'); score=r.get('scored_predicted_label')
   counts['labels_differ']+=gen!=score
   counts['parsed_nonnull_labels_differ']+=gen is not None and gen!=score
   if gen!=score and len(examples)<5: examples.append({k:r.get(k) for k in ['input_id','gold_label','generated_text','parsed_generated_label','scored_predicted_label','correct','parse_success','generation_scoring_disagreement']})
   if gold is not None:
    g=gen==gold; s=score==gold
    counts['generation_correct_recomputed']+=g; counts['scoring_correct_recomputed']+=s
    for field,v in [('generation_correct',g),('scoring_correct',s),('correct',g)]:
     if field in r: counts[field+'_differs_from_recomputed_generation' if field=='correct' else field+'_field_mismatch']+=r[field]!=v
   for k in META: meta[k][json.dumps(r.get(k),sort_keys=True,ensure_ascii=False)]+=1
   if r.get('timestamp'): ts.append(r['timestamp'])
   if isinstance(r.get('runtime_ms'),(int,float)): runtime.append(r['runtime_ms'])
   if isinstance(r.get('inference_runtime_ms'),(int,float)): inference_runtime.append(r['inference_runtime_ms'])
   consumer(r)
 return {'path':P+suffix,'uncompressed_sha256':h.hexdigest(),'counts':dict(counts),'unique_input_ids':len(ids),'label_difference_examples':examples,'inference_runtime_ms':{'n':len(inference_runtime),'sum':sum(inference_runtime),'min':min(inference_runtime) if inference_runtime else None,'max':max(inference_runtime) if inference_runtime else None},'metadata':{k:dict(v) for k,v in meta.items()},'timestamp_min':min(ts) if ts else None,'timestamp_max':max(ts) if ts else None,'runtime_ms':{'n':len(runtime),'sum':sum(runtime),'min':min(runtime) if runtime else None,'max':max(runtime) if runtime else None,'median':float(np.median(runtime)) if runtime else None}}
# 6x6: compact sums and item cell coverage only.
a=co.defaultdict(lambda:[0,0,0]); coverage=co.defaultdict(set); dims=co.Counter(); cells=co.Counter()
def six(r):
 g=r['parsed_generated_label']==r['gold_label']; s=r['scored_predicted_label']==r['gold_label']
 q,u,d=r['question_language'],r['gui_language'],r['dimension']; matched=q==u
 coverage[r['parallel_id']].add((q,u)); dims[d]+=1; cells[q+'->'+u]+=1
 for filt in ['all']+(['excluding_th'] if 'th' not in (q,u) else []):
  for dim in ['all',d]:
   x=a[(filt,dim,'matched' if matched else 'mismatch')];x[0]+=1;x[1]+=g;x[2]+=s
six_audit=audit_stream('full_6x6.jsonl',six)
gaps={}
for filt,dim,mat in a:
 key=filt+'/'+dim
 if key in gaps: continue
 x=a[(filt,dim,'matched')];y=a[(filt,dim,'mismatch')]
 gaps[key]={'matched_n':x[0],'mismatch_n':y[0]}
 for i,name in [(1,'generation'),(2,'scoring')]:
  gaps[key][name]={'matched_successes':x[i],'mismatch_successes':y[i],'matched_accuracy':x[i]/x[0],'mismatch_accuracy':y[i]/y[0],'matched_minus_mismatch_pp':100*(x[i]/x[0]-y[i]/y[0])}
six_summary=json.loads(z.read(P+'full_6x6_summary.json'))
six_audit.update({'unique_items':len(coverage),'item_cell_counts':dict(co.Counter(map(len,coverage.values()))),'dimensions':dict(dims),'cells':dict(cells),'raw_gaps':gaps,'supplied_summary_selected':{k:six_summary[k] for k in ['overall_accuracy','matched_mean','mismatch_mean','matched_mismatch_delta','parsing_failure_rate','scoring_generation_disagreement_rate']}})
for mode in ['generation','scoring']:
 c=six_audit['counts'][mode+'_correct_recomputed'];six_audit[mode+'_overall']={'successes':c,'n':six_audit['counts']['rows'],'accuracy':c/six_audit['counts']['rows']}
del coverage
# REL compact quartet records and reused-prediction provenance.
groups=co.defaultdict(dict); rel_ids={}; flags=co.Counter()
def rel(r):
 gold=r['gold_answer']; g=int(r.get('parsed_generated_label')==gold)
 row=(r['parallel_id'],r['query_language'],r['gui_language'],r['text_dependency'],r['status'],bool(r.get('source_issue')),g)
 if r['condition'] in groups[r['pair_id']]: flags['duplicate_pair_condition']+=1
 groups[r['pair_id']][r['condition']]=row
 flags['primary_analysis_flag_mismatch']+=r['primary_analysis']!=(r['query_language']!=r['gui_language'])
 flags['eligible_flag_mismatch']+=r['primary_analysis_eligible']!=(r['query_language']!=r['gui_language'] and not r.get('source_issue'))
 rel_ids[r['input_id']]=(r.get('prediction_source_input_id'),bool(r.get('reused_prediction')),r.get('parsed_generated_label'),r.get('scored_predicted_label'))
rel_audit=audit_stream('full_rel/results.jsonl',rel)
pred_ids={}
def pred(r):pred_ids[r['input_id']]=(r.get('parsed_generated_label'),r.get('scored_predicted_label'))
pred_audit=audit_stream('full_rel/predictions.jsonl',pred)
for iid,(source,reused,g,s) in rel_ids.items():
 flags['source_missing_from_predictions']+=source not in pred_ids
 flags['source_label_mismatch']+=source in pred_ids and pred_ids[source]!=(g,s)
 flags['nonreused_input_missing_predictions']+=not reused and iid not in pred_ids
quartets=[]; exclude=co.Counter()
for pair,rs in groups.items():
 if set(rs)!=set(CON):exclude['incomplete']+=1;continue
 anchor=rs['O'];pid,q,u,dep=anchor[:4]
 if any(v[:4]!=anchor[:4] for v in rs.values()):exclude['metadata_inconsistent']+=1;continue
 if any(v[4]!='success' for v in rs.values()):exclude['failure']+=1;continue
 if q==u:exclude['matched']+=1;continue
 if any(v[5] for v in rs.values()):exclude['source_issue_mismatch_quartets']+=1;continue
 quartets.append((pid,q,u,dep,np.array([rs[c][6] for c in CON],dtype=np.int64)))

def stats(rows,boot=False):
 n=len(rows); sums=sum((r[4] for r in rows),np.zeros(4,dtype=np.int64));out={'quartets':n,'items':len(set(r[0] for r in rows)),'accuracy':{c:{'successes':int(sums[i]),'n':n,'accuracy':float(sums[i]/n)} for i,c in enumerate(CON)},'contrasts':{}}
 for ca,cb in CONTRASTS:
  ia,ib=CON.index(ca),CON.index(cb); corrected=sum(r[4][ia]==1 and r[4][ib]==0 for r in rows); introduced=sum(r[4][ia]==0 and r[4][ib]==1 for r in rows)
  out['contrasts'][ca+'-'+cb]={'net_successes':int(sums[ia]-sums[ib]),'n':n,'difference_pp':float(100*(sums[ia]-sums[ib])/n),'errors_corrected':int(corrected),'errors_introduced':int(introduced)}
 out['interaction_pp']=float(100*(sums[3]-sums[2]-sums[1]+sums[0])/n)
 if boot:
  clusters=co.defaultdict(lambda:np.zeros(9,dtype=np.int64))
  for r in rows:
   v=clusters[r[0]];v[0]+=1;v[1:5]+=r[4]
   v[5:9]+=np.array([r[4][CON.index(a)]-r[4][CON.index(b)] for a,b in CONTRASTS])
  arr=np.stack([clusters[k] for k in sorted(clusters)]);rng=np.random.default_rng(42);draws=[]
  for start in range(0,5000,100):
   sampled=arr[rng.integers(0,len(arr),size=(min(100,5000-start),len(arr)))].sum(axis=1)
   draws.append(100*sampled[:,5:9]/sampled[:,[0]])
  ci=np.percentile(np.concatenate(draws),[2.5,97.5],axis=0)
  for j,(ca,cb) in enumerate(CONTRASTS):out['contrasts'][ca+'-'+cb]['exploratory_unadjusted_cluster_bootstrap_95ci_pp']=ci[:,j].tolist()
 return out
rel_stats={}
for dep in ['all','dependent','independent']:
 rows=[r for r in quartets if dep=='all' or r[3]==dep]
 out=stats(rows,True);out['by_gui_language']={x:stats([r for r in rows if r[2]==x]) for x in sorted(set(r[2] for r in rows))};out['by_question_language']={x:stats([r for r in rows if r[1]==x]) for x in sorted(set(r[1] for r in rows))};out['by_question_to_gui_direction']={q+'->'+u:stats([r for r in rows if r[1]==q and r[2]==u]) for q,u in sorted(set((r[1],r[2]) for r in rows))};rel_stats[dep]=out
supplied=json.loads(z.read(P+'full_rel/summary.json'));diff=[]
for dep in ['dependent','independent']:
 a=rel_stats[dep];b=supplied['primary_descriptive_summary'][dep]
 if a['quartets']!=b['complete_unflagged_mismatch_quartets'] or a['items']!=b['items']:diff.append(dep+' denominators')
 for c in CON:
  if abs(a['accuracy'][c]['accuracy']-b['generation_accuracy'][c])>1e-12:diff.append(dep+' accuracy '+c)
 for c,v in a['contrasts'].items():
  for k in ['difference_pp','errors_corrected','errors_introduced']:
   if abs(v[k]-b['paired_contrasts'][c][k])>1e-10:diff.append(dep+' contrast '+c+' '+k)
 if abs(a['interaction_pp']-b['interaction_pp'])>1e-10:diff.append(dep+' interaction')
report={'provenance':{'archive':str(ZIP),'numpy':np.__version__,'script':str(pathlib.Path(__file__).resolve()),'archived_code_executed':False},'six_by_six':six_audit,'rel_results':rel_audit,'rel_predictions':pred_audit,'rel_integrity':dict(flags),'rel_quartet_exclusions':dict(exclude),'rel_filter':'Exactly O/R/C/F; all status success; consistent parallel_id/query/gui/dependency; query_language != gui_language; no source_issue in any condition. Outcome recomputed parsed_generated_label == gold_answer. Unparseable labels count wrong.','rel_primary':rel_stats,'rel_summary_disagreements':diff,'bootstrap':{'draws':5000,'seed':42,'rng':'numpy default_rng PCG64','unit':'parallel_id within dependency stratum, all eligible rows retained per item','estimand':'ratio of summed paired success differences to summed quartet counts; percentage points','CI':'2.5 and 97.5 percentile; exploratory/unadjusted; no p-values; only overall and dependency subgroup CIs','caution':'Item resampling assesses item variation conditional on observed languages, labels, model/run and filtering; source issues excluded, not adjudicated.'},'six_endpoint_definition_evidence':{'path':'internvl35_prep/mpr-gui-crosslocale/01_code/src/mpr_crosslocale/inference/run_cross_locale.py','correct':'scored == row[gold_label]','generation_scoring_disagreement':'parsed is not None and parsed != scored','consequence':'41 parse failures are excluded from disagreement flag by definition; all have correct scored label; summary uses scored accuracy.'},'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'six':{k:six_audit[k] for k in ['counts','generation_overall','scoring_overall','unique_items','item_cell_counts']},'six_raw_gaps':gaps,'rel_integrity':dict(flags),'rel_exclusions':dict(exclude),'rel_summary_disagreements':diff,'rel_primary':{k:{x:v[x] for x in ['quartets','items','accuracy','contrasts']} for k,v in rel_stats.items()},'peak_rss_bytes':report['peak_rss_bytes_macos']},indent=2))
