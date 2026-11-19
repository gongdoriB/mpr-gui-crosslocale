"""Compare B200 method-dev inputs (4 languages) with our 6-language O rows."""
import json, sys, glob
from collections import Counter
b200 = {}
for f in glob.glob(sys.argv[1] + '/**/ppocrv5_cuda/inputs.jsonl', recursive=True):
    if '/infer_' in f:
        continue
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); b200[r['pair_id']] = r
ours = {}
for l in open(sys.argv[2], encoding='utf-8'):
    r = json.loads(l)
    if r['condition'] == 'O':
        ours[r['pair_id']] = r
print('b200 pairs', len(b200), 'ours O pairs', len(ours))
k = next(iter(b200)); print('b200 keys', sorted(b200[k])); print('example b200 pair', k, '| ours example', next(iter(ours)))
common = set(b200) & set(ours); print('common', len(common))
c = Counter()
for p in common:
    a, o = b200[p], ours[p]
    opts = o['options'] if isinstance(o['options'], dict) else json.loads(o['options'])
    c['stem_eq'] += a['question_stem'] == o['question_stem']
    c['opts_eq'] += a['options'] == opts
    c['img_eq'] += a['image_sha256'] == o['image_sha256']
    c['path_eq'] += a['image_path'] == o['image_path']
print(dict(c))
only = set(b200) - set(ours); print('b200 only', len(only), list(only)[:3])
inv = 0; langs = Counter(); last = None
for f in glob.glob(sys.argv[1] + '/**/ppocrv5_cuda/ocr/attempt_0001/inventories.jsonl', recursive=True):
    for l in open(f, encoding='utf-8'):
        last = json.loads(l); inv += 1; langs[last.get('gui_language')] += 1
print('inventories', inv, dict(langs)); print('inv keys', sorted(last))
