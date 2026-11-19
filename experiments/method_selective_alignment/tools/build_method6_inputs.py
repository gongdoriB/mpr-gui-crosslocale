"""Build 6-language method inputs (method schema) from rstem6 O rows. Gold answers kept in a separate labels file."""
import hashlib, json, sys
from collections import Counter
src, out_inputs, out_labels = sys.argv[1], sys.argv[2], sys.argv[3]
rows, labels = [], []
for l in open(src, encoding='utf-8'):
    r = json.loads(l)
    if r['condition'] != 'O':
        continue
    opts = r['options'] if isinstance(r['options'], dict) else json.loads(r['options'])
    assert list(opts) == list('ABCD')
    rows.append({'pair_id': r['pair_id'], 'parallel_id': r['parallel_id'], 'source_language': r['query_language'],
                 'gui_language': r['gui_language'], 'image_path': r['image_path'], 'image_sha256': r['image_sha256'],
                 'question_stem': r['question_stem'], 'options': opts, 'option_order': list('ABCD')})
    labels.append({'pair_id': r['pair_id'], 'gold_answer': r['gold_answer'], 'split': r['split']})
rows.sort(key=lambda r: r['pair_id']); labels.sort(key=lambda r: r['pair_id'])
assert len(rows) == 4740 and len({r['pair_id'] for r in rows}) == 4740
for path, data in ((out_inputs, rows), (out_labels, labels)):
    with open(path, 'w', encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
print(json.dumps({'rows': len(rows), 'items': len({r['parallel_id'] for r in rows}),
                  'directions': len({(r['source_language'], r['gui_language']) for r in rows}),
                  'images': len({r['image_sha256'] for r in rows}),
                  'images_by_gui': dict(Counter(l for l in {(r['image_sha256'], r['gui_language']) for r in rows} for l in [l[1]])),
                  'inputs_sha256': sha(out_inputs), 'labels_sha256': sha(out_labels), 'source_sha256': sha(src)}, indent=1))
