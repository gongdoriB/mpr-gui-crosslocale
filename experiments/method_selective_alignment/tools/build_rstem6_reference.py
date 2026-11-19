"""reference/rstem6_<model>_labels.jsonl.gz for the method6 gate: pair_id, condition, question_sha256, label (parsed_generated_label) of the R_stem/R_opts 6-language run.
Usage: python3 -I build_rstem6_reference.py <rstem6_conditions.jsonl> <out_dir> <model>=<predictions.jsonl> ..."""
import gzip, hashlib, json, sys
from pathlib import Path
cond, out = sys.argv[1], Path(sys.argv[2])
q = {}
for l in open(cond, encoding='utf-8'):
    r = json.loads(l); q[(r['pair_id'], r['condition'])] = r['question']
for spec in sys.argv[3:]:
    model, pred = spec.split('=', 1)
    recs, seen = [], set()
    for l in open(pred, encoding='utf-8'):
        d = json.loads(l)
        if d.get('status') != 'success':
            continue
        k = (d['pair_id'], d['condition'])
        if k in seen:
            continue
        seen.add(k)
        question = q[k]
        assert question in d['rendered_prompt'], k
        recs.append({'pair_id': k[0], 'condition': k[1], 'question_sha256': hashlib.sha256(question.encode('utf-8')).hexdigest(), 'label': d['parsed_generated_label']})
    recs.sort(key=lambda r: (r['pair_id'], r['condition']))
    assert len(recs) == 18960, (model, len(recs))
    with gzip.GzipFile(out / f'rstem6_{model}_labels.jsonl.gz', 'wb', mtime=0) as f:
        for r in recs:
            f.write((json.dumps(r, separators=(',', ':')) + '\n').encode())
    print(model, len(recs))
