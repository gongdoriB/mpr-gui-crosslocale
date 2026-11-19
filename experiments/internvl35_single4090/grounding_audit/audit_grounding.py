"""Read-only, bounded audit of the supplied archive. No archived code is executed."""
import collections
import hashlib
import json
import pathlib
import re
import zipfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'internvl35_prep.zip'
PREFIX = 'internvl35_prep/results_internvl35_20261109/'
CONDITIONS = ['O', 'R', 'C', 'F']
CONTRASTS = [('R', 'O'), ('C', 'O'), ('F', 'C'), ('F', 'R')]
POINT = re.compile(r'\s*\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)\s*')

def rows(z, path):
    with z.open(PREFIX + path) as f:
        for line in f:
            yield json.loads(line)

def estimates(pairs):
    by_item = collections.defaultdict(lambda: np.zeros(5))
    for pair in pairs:
        a = by_item[pair['O']['parallel_id']]
        a[0] += 1
        for j, c in enumerate(CONDITIONS):
            a[j + 1] += pair[c]['point_in_gt_box']
    a = np.array(list(by_item.values()))
    total = a.sum(axis=0)
    rng = np.random.default_rng(42)
    # 100 draws per chunk: bounded memory; retain only 5000 x 4 estimates.
    boot = []
    for _ in range(50):
        sample = a[rng.integers(0, len(a), size=(100, len(a)))].sum(axis=1)
        boot.append(sample[:, 1:] / sample[:, :1])
    boot = np.concatenate(boot)
    result = {'items': len(a), 'pairs': int(total[0]),
              'accuracy': dict(zip(CONDITIONS, (total[1:] / total[0]).tolist())), 'contrasts': {}}
    for c, b in CONTRASTS:
        ix, iy = CONDITIONS.index(c), CONDITIONS.index(b)
        result['contrasts'][f'{c}-{b}'] = {
            'difference_pp': float(100 * (total[ix + 1] - total[iy + 1]) / total[0]),
            'ci95_pp': np.quantile(100 * (boot[:, ix] - boot[:, iy]), [.025, .975]).tolist()}
    return result

def main():
    report = {'method': 'Exploratory paired item-cluster percentile bootstrap; 5000 replicates, seed 42; no multiplicity adjustment; invalid outputs count as misses.',
              'numpy_version': np.__version__}
    with zipfile.ZipFile(ARCHIVE) as z:
        pairs = collections.defaultdict(dict)
        ids = set()
        count = collections.Counter()
        invalid = []
        for r in rows(z, 'grounding_main_orcf_literal/predictions.jsonl'):
            assert r['trial_id'] not in ids, r['trial_id']
            ids.add(r['trial_id'])
            m = POINT.fullmatch(r['raw_output'])
            point = [int(x) for x in m.groups()] if m else None
            if point is not None and max(point) > 1000:
                point = None
            valid = point is not None
            hit = False
            if valid:
                x, y = [point[i] * r['image_size'][i] / 1000 for i in (0, 1)]
                b = r['gt_bbox']
                hit = b[0] <= x <= b[2] and b[1] <= y <= b[3]
                assert np.allclose([x, y], r['point_original_pixels'])
            assert valid == r['valid_output'] and hit == r['point_in_gt_box'], r['trial_id']
            assert point == r['point_1000']
            count[(r['condition'], 'rows')] += 1
            count[(r['condition'], 'valid')] += valid
            count[(r['condition'], 'hit')] += hit
            if not valid:
                invalid.append({'trial_id': r['trial_id'], 'raw_output': r['raw_output']})
            pairs[r['pair_id']][r['condition']] = r
        assert all(set(v) == set(CONDITIONS) for v in pairs.values())
        report['rows'] = len(ids)
        report['duplicates'] = 0
        report['scoring_mismatches'] = 0
        report['invalid_outputs'] = invalid
        summary = json.loads(z.read(PREFIX + 'grounding_main_orcf_literal/summary.json'))
        report['supplied_summary_matches'] = all(summary['conditions'][c][k] == count[(c, key)] for c in CONDITIONS for k, key in [('count', 'rows'), ('valid', 'valid'), ('hit', 'hit')])
        report['overall'] = estimates(list(pairs.values()))
        report['by_gui'] = {gui: estimates([v for v in pairs.values() if v['O']['gui_language'] == gui]) for gui in ['en','zh','th','ru']}
        # Join the same question/language pairs, using REL's primary eligibility.
        rel = collections.defaultdict(dict)
        review_counts = collections.Counter()
        for r in rows(z, 'full_rel/results.jsonl'):
            review_counts['human_reviewed_true'] += bool(r.get('human_reviewed'))
            review_counts['screenshot_verified_true'] += bool(r.get('screenshot_verified'))
            if r['pair_id'] in pairs and not r['matched'] and not r.get('source_issue') and r['text_dependency'] == 'dependent' and r['status'] == 'success':
                rel[r['pair_id']][r['condition']] = r['generation_correct']
        report['rel_review_true_rows'] = dict(review_counts)
        joined = {key: r for key, r in rel.items() if set(r) == set(CONDITIONS)}
        tab = collections.Counter()
        for key, r in joined.items():
            if not r['O']:
                g = pairs[key]
                tab[f"g{int(g['O']['point_in_gt_box'])}{int(g['R']['point_in_gt_box'])}_relR{int(r['R'])}"] += 1
        report['joint_exploratory'] = {'complete_dependent_unflagged_pairs': len(joined), 'items': len({pairs[k]['O']['parallel_id'] for k in joined}), 'among_rel_O_wrong': dict(tab), 'note': 'Separate tasks; this table does not identify a causal mediation mechanism.'}
        # Verify source snapshots against the execution contract.
        contract = json.loads(z.read(PREFIX + 'grounding_main_orcf_literal/run_contract.json'))
        plan = json.loads(z.read(PREFIX + 'grounding_main_orcf_literal/input_plan.json'))
        planned = {t['trial_id']: t for group in plan for t in group['trials']}
        assert set(planned) == ids
        for pair in pairs.values():
            for r in pair.values():
                assert all(r[k] == v for k, v in planned[r['trial_id']].items())
        report['plan_membership_and_metadata_match'] = True
        repo = ROOT / 'source/mpr-gui-crosslocale'
        gt = json.loads((repo / 'followup_rel_4lang/grounding_eval/annotations_20261103/grounding_gt_draft_170.json').read_text())
        pilots = set(gt['gt_assembly']['pilot_item_ids'])
        cells = [cell for item, locales in gt['annotations'].items() if item not in pilots for cell in locales.values()]
        report['main_gt_review_metadata'] = {
            'cells': len(cells),
            'bbox_review_status': dict(collections.Counter(c['bbox_provenance']['review_status'] for c in cells)),
            'reference_equivalence': dict(collections.Counter(c['reference_equivalence'] for c in cells)),
            'options_equivalence': dict(collections.Counter(c['options_equivalence'] for c in cells))}
        qwen = json.loads((repo / 'followup_rel_4lang/artifacts/qwen3_single4090_20261107/grounding_main_orcf/summary.json').read_text())
        report['qwen3_summary_only_comparison'] = {
            gui: {c: {'internvl35': summary['by_gui'][gui][c]['hit'] / summary['by_gui'][gui][c]['count'],
                      'qwen3': qwen['by_gui'][gui][c]['hit'] / qwen['by_gui'][gui][c]['count']}
                  for c in CONDITIONS} for gui in ['en', 'zh', 'th', 'ru']}
        report['comparison_limit'] = 'Qwen3 values from supplied summary; not a new paired between-model test. Model/processor/tokenization/batch differ.'
        hashes = {}
        for name, wanted in contract['implementation_sha256'].items():
            p = repo / ('01_code/src/mpr_crosslocale/models' if name == 'internvl25.py' else 'followup_rel_4lang/grounding_eval') / name
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest() == wanted
        p = repo / 'followup_rel_4lang/grounding_eval/run_internvl_grounding.py'
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest() == contract['runner_sha256']
        report['source_hashes_match_contract'] = hashes
    (ROOT / 'grounding_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
