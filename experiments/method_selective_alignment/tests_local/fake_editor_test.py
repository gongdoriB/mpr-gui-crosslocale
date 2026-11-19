"""Exercise edit application for all 6 languages with a fake editor that selects real inventory strings and rewrites options."""
import json, sys, importlib.util, collections
from pathlib import Path
spec = importlib.util.spec_from_file_location('re6', '../package/method6/run_edit6.py'); re6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(re6)
repo = Path('../repo').resolve(); sys.path[:0] = [str(repo), str(repo / '01_code/src')]
from mpr_crosslocale.methods.selective_alignment import schemas
re6.widen_languages(schemas)
from followup_rel_4lang.method_dev import editor_matrix as matrix, stem_refinement_2x2 as refinement
from mpr_crosslocale.methods.selective_alignment.search import quoted_referents
rows = [json.loads(l) for l in open('../package/inputs/method6_inputs.jsonl', encoding='utf-8')]
inv = {r['image_sha256']: r for r in map(json.loads, open('mock_inventories6.jsonl', encoding='utf-8'))}
def fake(stage, role, system, user, image=None):
    data = json.JSONDecoder().raw_decode(user[user.index('{'):])[0]
    if stage.startswith('extract'):
        try: refs = quoted_referents({'question_stem': data['fields']['question_stem'], 'options': {k: '' for k in 'ABCD'}})
        except ValueError: refs = []
        raw = json.dumps({'referents': [{k: r[k] for k in ('field', 'text', 'occurrence')} for r in refs]}, ensure_ascii=False)
    elif stage.startswith('selective'):
        vs = data['target_gui_visible_strings']
        raw = json.dumps({'translations': {r['span_id']: (vs[i % len(vs)] if vs else None) for i, r in enumerate(data['referents'])}}, ensure_ascii=False)
    else:
        vs = data['target_gui_visible_strings']; req = json.JSONDecoder().raw_decode(user.split('already fixed: ', 1)[1])[0]
        out = {'translations': {r['span_id']: (vs[0] if vs else None) for r in data.get('referents', [])},
               'options': {k: f'[{data["target_language"]}] ' + data['options'][k] for k in 'ABCD'}}
        raw = json.dumps({k: out[k] for k in req}, ensure_ascii=False)
    return {'raw_output': raw, 'runtime_ms': 0, 'cache_hit': False}
c = collections.Counter(); changed = collections.Counter()
for r in rows[::7]:
    ex = refinement.extract(r, fake)
    for cell in ('S', 'H'):
        res = matrix.rewrite(r, inv[r['image_sha256']], ex['referents'], fake, 'D', cell)
        c[(cell, r['gui_language'], res['status'])] += 1
        changed[(cell, r['gui_language'])] += res['output']['question_stem'] != r['question_stem']
for k in sorted(c): print(k, c[k])
print('stem changed', dict(sorted(changed.items())))
