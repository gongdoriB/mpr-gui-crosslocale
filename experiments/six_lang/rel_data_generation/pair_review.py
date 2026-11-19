"""Undirected P1/P2 review queue. Assistant findings are stored separately.

Run: python3 pair_review.py
This never edits QAS, mappings, conditions, or human-review flags.
"""
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANGUAGES = ('en', 'zh', 'th', 'ru')
PAIRS = list(combinations(LANGUAGES, 2))
STATUSES = ('pass', 'extraction_error', 'alignment_error', 'source_issue')
REVIEW_FILE = ROOT / 'annotation/pair_review_assistant_20260923.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def make_queue(audits):
    """Merge both directions; take the higher risk priority; omit self/P3 pairs."""
    directed = {}
    for r in audits:
        key = (r['parallel_id'], r['query_language'], r['gui_language'])
        if key in directed:
            raise ValueError(f'Duplicate directed audit: {key}')
        directed[key] = r
    queue = []
    for pid in sorted({r['parallel_id'] for r in audits}):
        for a, b in PAIRS:
            ab, ba = directed.get((pid, a, b)), directed.get((pid, b, a))
            if not ab and not ba:
                continue
            if not ab or not ba:
                raise ValueError(f'Missing reverse audit: {pid} {a}-{b}')
            for left, right in [('query_fields', 'gui_fields'), ('query_extraction', 'gui_extraction')]:
                if ab[left] != ba[right] or ab[right] != ba[left]:
                    raise ValueError(f'Inconsistent reverse evidence: {pid} {a}-{b} {left}')
            if ab['text_dependency'] != ba['text_dependency']:
                raise ValueError(f'Inconsistent dependency: {pid} {a}-{b}')
            priority = min(ab['review_priority'], ba['review_priority'])
            if priority not in (1, 2):
                continue
            sides = {
                a: {'fields': ab['query_fields'], 'extraction': ab['query_extraction'], 'image_path': ba['image_path']},
                b: {'fields': ab['gui_fields'], 'extraction': ab['gui_extraction'], 'image_path': ab['image_path']},
            }
            record = {
                'review_id': f'{pid}::{a}-{b}', 'parallel_id': pid,
                'language_pair': f'{a}-{b}', 'languages': [a, b],
                'text_dependency': ab['text_dependency'], 'review_priority': priority,
                'merged_direction_ids': [ab['pair_id'], ba['pair_id']],
                'direction_priorities': {f'{a}->{b}': ab['review_priority'], f'{b}->{a}': ba['review_priority']},
                'review_flags': sorted(set(ab['review_flags']) | set(ba['review_flags'])),
                'source_issues': [r['source_issue'] for r in (ab, ba) if r.get('source_issue')],
                'extraction_methods': ab['extraction_methods'], 'sides': sides,
                'count_definition': ab['count_definition'],
            }
            record['source_fingerprint'] = digest(record)
            queue.append(record)
    return queue


def attach_reviews(queue, reviews):
    index = {}
    image_hashes = {}
    for r in reviews:
        if r['review_id'] in index:
            raise ValueError('Duplicate review: ' + r['review_id'])
        if r['status'] not in STATUSES or not r['reason'].strip():
            raise ValueError('Invalid review: ' + r['review_id'])
        index[r['review_id']] = r
    for q in queue:
        r = index.get(q['review_id'])
        images_unchanged = True
        for evidence in r.get('visual_evidence', []) if r else []:
            path = ROOT / evidence['image_path']
            if path.is_file() and evidence.get('sha256'):
                if str(path) not in image_hashes:
                    image_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                images_unchanged &= image_hashes[str(path)] == evidence['sha256']
        q['assistant_review'] = r if r and r['source_fingerprint'] == q['source_fingerprint'] and images_unchanged else None
        q['review_state'] = 'assistant_reviewed' if q['assistant_review'] else ('stale' if r else 'unreviewed')
    return queue


def summarize(queue):
    def counts(rows):
        return {'total': len(rows), 'priority_1': sum(r['review_priority'] == 1 for r in rows),
                'priority_2': sum(r['review_priority'] == 2 for r in rows),
                'judgments': dict(Counter(r['assistant_review']['status'] if r.get('assistant_review') else r['review_state'] for r in rows))}
    return {
        **counts(queue), 'unique_samples': len({r['parallel_id'] for r in queue}),
        'by_pair': {f'{a}-{b}': counts([r for r in queue if r['language_pair'] == f'{a}-{b}']) for a, b in PAIRS},
        'by_dependency': {d: counts([r for r in queue if r['text_dependency'] == d]) for d in ('dependent', 'independent')},
        'screenshots_inspected_pairs': sum(bool(r.get('assistant_review') and r['assistant_review'].get('visual_evidence')) for r in queue),
        'scope': 'Assistant first-pass review of source text and extracted references; screenshots inspected only where explicitly recorded. Not human sign-off or a full gold-answer audit.',
        'priority_3_included': False, 'directions_merged': True,
    }


def export_pair_review(out, audits=None):
    out = Path(out)
    queue = make_queue(audits if audits is not None else read_jsonl(out / 'object_extraction_by_pair.jsonl'))
    attach_reviews(queue, read_jsonl(REVIEW_FILE) if REVIEW_FILE.exists() else [])
    write_jsonl(out / 'review_6pairs_p12.jsonl', queue)
    summary = summarize(queue)
    (out / 'review_6pairs_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    template = (ROOT / 'pair_review_template.html').read_text(encoding='utf-8')
    data = json.dumps({'schema': 'cuai-pair-review-v1', 'records': queue}, ensure_ascii=False).replace('<', '\\u003c')
    (out / '6개언어조합_추출검수.html').write_text(template.replace('__REVIEW_DATA__', data), encoding='utf-8')
    return summary


if __name__ == '__main__':
    for directory in ('variants', 'independent', 'rel_all_366'):
        result = export_pair_review(ROOT / 'data' / directory)
        print(directory, json.dumps(result, ensure_ascii=False))
