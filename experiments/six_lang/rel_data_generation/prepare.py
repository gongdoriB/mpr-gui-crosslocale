"""Rebuild the four-language study from immutable local annotation sources."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LANGS = ('en', 'zh', 'th', 'ru', 'fr', 'ja')
ACTIVE = ROOT / 'mpr-gui-crosslocale-rel-annotation-ui'
sys.path.insert(0, str(ACTIVE / '01_code/src'))
from mpr_crosslocale.data.options import parse_question_options


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def candidates(text):
    # Offsets are Python Unicode code points; only the inner quoted text is selected.
    pattern = r'“([^”]+)”|«([^»]+)»|「([^」]+)」|"([^"\n]+)"|(?<!\w)\x27(.+?)\x27(?!\w)'
    result = []
    for m in re.finditer(pattern, text):
        group = next(i for i in range(1, 6) if m.group(i) is not None)
        result.append({'start': m.start(group), 'end': m.end(group),
                       'text': m.group(group), 'ref_id': None})
    return result


def main():
    manifest_path = ACTIVE / 'annotation/rel_text_dependency/data/pilot_manifest.json'
    labels_path = ACTIVE / 'annotation/rel_text_dependency/outputs/rel_annotations_integrated_366.csv'
    manifest = json.loads(manifest_path.read_text())
    labels = list(csv.DictReader(labels_path.open(encoding='utf-8-sig')))
    assert len(labels) == len({r['parallel_id'] for r in labels}) == 366
    source_hash = hashlib.sha256(manifest_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')).hexdigest()
    assert {r['manifest_sha256'] for r in labels} == {source_hash}, 'annotation belongs to another manifest'
    keep = {r['parallel_id'] for r in labels if r['text_dependency'] == 'dependent' and r['review_flag'] == '0'}
    assert len(keep) == 302
    qa = {lang: [json.loads(s) for s in (HERE / 'sources/qas' / f'rel_el_{lang}.jsonl').read_text().splitlines() if s.strip()] for lang in LANGS}
    items, reviews, originals, plan, assets = [], [], [], [], {}
    for item in manifest['items']:
        if item['parallel_id'] not in keep:
            continue
        item = dict(item)
        item['original_display_order'] = item['display_order']
        item['display_order'] = len(items) + 1
        item.pop('ko_helper', None)
        item['locales'] = {lang: item['locales'][lang] for lang in LANGS}
        review = {'parallel_id': item['parallel_id'], 'parallel_semantics_reviewed': False, 'reviewer': '', 'notes': '', 'locales': {}}
        for lang, loc in item['locales'].items():
            row = qa[lang][loc['source_line'] - 1]
            parsed = parse_question_options(row['question'])
            assert parsed.option_parse_status == 'ok'
            assert parsed.question_stem == loc['question_stem'] and parsed.options == loc['options']
            assert row['image_path'].removeprefix('../images/') == loc['asset']
            assert row['answer'] in 'ABCD' and len(row['answer']) == 1
            loc['gold_answer'] = row['answer']
            fields = {'question_stem': loc['question_stem'], **loc['options']}
            review['locales'][lang] = {'span_reviewed': False, 'gui_labels_reviewed': False,
                'image_asset': loc['asset'], 'fields': fields,
                'spans': {field: candidates(text) for field, text in fields.items()}, 'gui_labels': {}}
            assets[loc['asset']] = {'asset': loc['asset'], 'locale': lang,
                'url': f"https://huggingface.co/datasets/{manifest['source']['hf_repo']}/resolve/{manifest['source']['hf_revision']}/{loc['asset']}"}
        items.append(item)
        reviews.append(review)
        for q in LANGS:
            for g in LANGS:
                loc = item['locales'][q]
                pair = {'pair_id': f"{item['parallel_id']}::{q}::{g}", 'parallel_id': item['parallel_id'],
                    'state_key': item['state_key'], 'query_language': q, 'gui_language': g,
                    'matched': q == g, 'image_path': 'assets/images/' + item['locales'][g]['asset'],
                    'gold_answer': item['locales'][g]['gold_answer'],
                    'source_query_gold_answer': loc['gold_answer'],
                    'gold_agrees_across_locales': loc['gold_answer'] == item['locales'][g]['gold_answer']}
                originals.append({**pair, 'condition': 'O', 'question_stem': loc['question_stem'],
                    'options': loc['options'], 'option_order': loc['option_order'],
                    'status': 'original_text_only_pending_semantic_review_and_assets'})
                for c, a, b in [('O', 0, 0), ('R', 1, 0), ('C', 0, 1), ('F', 1, 1)]:
                    plan.append({**pair, 'condition': c, 'referent_alignment': a, 'context_localization': b,
                        'primary_analysis': q != g, 'status': 'pending_annotation'})
    assert len(items) == 302 and len(originals) == 302*36 and len(plan) == 302*36*4
    assert {i['parallel_id'] for i in items} == keep
    output = {'schema_version': 'followup-orcf-1', 'languages': list(LANGS), 'source': manifest['source'],
              'selection': {'dimension': 'rel', 'text_dependency': 'dependent', 'sample_size': 302}, 'items': items}
    dump(HERE / 'data/manifest_302_6lang.json', output)
    # Never overwrite human annotation on rebuild.
    review_path = HERE / 'annotation/referents_review.jsonl'
    if not review_path.exists():
        jsonl(review_path, reviews)
    jsonl(HERE / 'data/original_pairs.jsonl', originals)
    jsonl(HERE / 'data/condition_plan.jsonl', plan)
    dump(HERE / 'data/assets_required.json', list(assets.values()))
    dump(HERE / 'config.json', {'languages': list(LANGS), 'conditions': ['O', 'R', 'C', 'F'],
        'models': ['Qwen/Qwen2.5-VL-7B-Instruct', 'OpenGVLab/InternVL2_5-8B'],
        'primary_population': 'query_language != gui_language', 'matched_role': 'separate_reference',
        'intervention_fields': ['question_stem', 'A', 'B', 'C', 'D'],
        'contrasts': ['R-O', 'F-C', 'C-O', 'F-R'], 'interaction': '(F-C)-(R-O)',
        'analysis': {'outcome': 'generation_correct', 'confidence_level': 0.95,
            'pair_keys': ['model', 'parallel_id', 'query_language', 'gui_language'],
            'repeated_structure': ['parallel_id', 'state_key'], 'language_controls': ['query_language', 'gui_language']}})
    dump(HERE / 'data/preparation_report.json', {'items': len(items), 'languages': list(LANGS),
        'locale_records': len(items)*6, 'language_pairs': 36, 'original_pairs': len(originals),
        'primary_pairs': 302*30, 'primary_condition_rows_per_model': 302*30*4, 'matched_condition_rows_per_model': 302*6*4,
        'planned_rows_per_model': len(plan), 'planned_rows_two_models': len(plan)*2,
        'required_images': len(assets), 'cross_locale_gold_disagreements': sum(not r['gold_agrees_across_locales'] for r in originals),
        'r_c_f_ready': False, 'annotation_required': True,
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest_path, labels_path, *sorted((HERE/'sources/qas').glob('*.jsonl'))]}})
    print('Prepared 302 items / 6 languages / 10,872 pairs / 43,488 planned condition rows per model.')


if __name__ == '__main__':
    main()
