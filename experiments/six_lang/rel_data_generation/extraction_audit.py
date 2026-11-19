"""Pair-level reference extraction evidence, independent of inference outcomes.

Counts describe text mentions, NOT visually detected objects or verified entities.
"""
from collections import Counter
import html
import json
from pathlib import Path

FIELDS = ('question_stem','A','B','C','D')


def locale_stats(mapping, locale, quote_extractor):
    mentions=[];per_field={};raw_quotes={}
    for field in FIELDS:
        text=mapping['fields'][locale][field]
        spans=mapping['spans'][locale][field]
        per_field[field]=len(spans)
        raw_quotes[field]=len(quote_extractor(text))
        for span in spans:
            assert text[span['start']:span['end']]==span['text']
            mentions.append({'field':field,**span})
    return {
        'reference_occurrence_count':len(mentions),
        'unique_reference_text_count':len({s['text'].strip() for s in mentions}),
        'question_reference_count':per_field['question_stem'],
        'option_reference_count':sum(per_field[f] for f in 'ABCD'),
        'counts_by_field':per_field,'raw_quote_counts_by_field':raw_quotes,
        'raw_quote_count':sum(raw_quotes.values()),'mentions':mentions,
    }


def export_extraction_audit(out, rows, groups, mappings, quote_extractor):
    """Attach compact counts and emit a one-row-per-item/language-pair audit."""
    out=Path(out)
    stats={m['parallel_id']:{l:locale_stats(m,l,quote_extractor) for l in m['fields'] if m['spans'][l] is not None} for m in mappings}
    byid={m['parallel_id']:m for m in mappings}
    audits=[];by_pair={}
    for group in groups:
        pid=group['parallel_id'];q=group['query_language'];g=group['gui_language']
        a,b=stats[pid][q],stats[pid][g]
        raw_different=[f for f in FIELDS if a['raw_quote_counts_by_field'][f]!=b['raw_quote_counts_by_field'][f]]
        aligned_different=[f for f in FIELDS if a['counts_by_field'][f]!=b['counts_by_field'][f]]
        qroles={(s['field'],s['ref_id']) for s in a['mentions']}
        groles={(s['field'],s['ref_id']) for s in b['mentions']}
        flags=[]
        if group.get('source_issue'):flags.append('known_source_issue')
        if aligned_different or qroles!=groles:flags.append('aligned_reference_mismatch')
        if raw_different:flags.append('raw_quote_count_difference')
        if a['unique_reference_text_count']!=b['unique_reference_text_count']:flags.append('unique_text_count_difference')
        if not a['reference_occurrence_count'] and not b['reference_occurrence_count']:flags.append('no_reference_spans')
        mapping=byid[pid]
        compact={
            'query_reference_count':a['reference_occurrence_count'],
            'gui_reference_count':b['reference_occurrence_count'],
            'query_unique_text_count':a['unique_reference_text_count'],
            'gui_unique_text_count':b['unique_reference_text_count'],
            'query_raw_quote_count':a['raw_quote_count'],'gui_raw_quote_count':b['raw_quote_count'],
            'aligned_counts_equal':not aligned_different,'role_ids_equal':qroles==groles,
            'review_flags':flags,
        }
        priority=1 if any(f in flags for f in ['known_source_issue','aligned_reference_mismatch']) else 2 if flags and flags!=['no_reference_spans'] else 3
        audit={
            'pair_id':f'{pid}::{q}::{g}','parallel_id':pid,'query_language':q,'gui_language':g,
            'text_dependency':group.get('text_dependency',mapping.get('text_dependency','dependent')),
            'count_definition':'reference text occurrences in question and A-D, including repeats; not image object detections',
            **compact,'query_extraction':a,'gui_extraction':b,
            'aligned_count_difference_fields':aligned_different,'raw_quote_difference_fields':raw_different,
            'extraction_methods':mapping['methods'],'source_issue':group.get('source_issue'),
            'review_priority':priority,'review_status':'unreviewed',
            'query_fields':mapping['fields'][q],'gui_fields':mapping['fields'][g],
            'image_path':group['image_path'],
        }
        group['object_extraction']=compact
        by_pair[pid,q,g]=compact
        audits.append(audit)
    for row in rows:
        row['object_extraction']=dict(by_pair[row['parallel_id'],row['query_language'],row['gui_language']])
        context=stats[row['parallel_id']][row['context_language']]
        refs=stats[row['parallel_id']][row['reference_language']]
        row['object_extraction']['condition_reference_count']=context['reference_occurrence_count']
        row['object_extraction']['condition_unique_text_count']=refs['unique_reference_text_count']
    focus=sorted((r for r in audits if r['query_language']=='en' and r['gui_language']=='zh'),key=lambda r:(r['review_priority'],r['parallel_id']))
    for name,records in [('object_extraction_by_pair.jsonl',audits),('review_en_zh.jsonl',focus)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    summary={'pair_count':len(audits),'en_zh_sample_count':len(focus),
        'counts_are':'text reference occurrences; repeated mentions counted; unique text count is NOT semantic object count',
        'en_zh_flags':dict(Counter(f for r in focus for f in r['review_flags'])),
        'language_pairs':{f'{q}->{g}':{'samples':len(rs),'query_reference_occurrences':sum(r['query_reference_count'] for r in rs),
            'gui_reference_occurrences':sum(r['gui_reference_count'] for r in rs),
            'raw_quote_count_difference_samples':sum('raw_quote_count_difference' in r['review_flags'] for r in rs),
            'aligned_mismatch_samples':sum('aligned_reference_mismatch' in r['review_flags'] for r in rs)}
            for q,g in sorted({(r['query_language'],r['gui_language']) for r in audits})
            for rs in [[r for r in audits if r['query_language']==q and r['gui_language']==g]]},
        'caution':'Equal counts do not prove semantic alignment. English-Chinese review does not verify Thai or Russian.'}
    (out/'object_extraction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_review_html(out,focus)
    # Keep manually authored findings separate from regenerated extraction data.
    from pair_review import export_pair_review
    export_pair_review(out,audits)


def write_review_html(out,focus):
    esc=html.escape
    cards=[]
    for r in focus:
        lists=[]
        for label,key in [('영어','query_extraction'),('중국어','gui_extraction')]:
            s=r[key]
            details=''.join(f"<li><b>{esc(m['field'])}</b> [{esc(m['ref_id'])}] {esc(m['text'])}</li>" for m in s['mentions']) or '<li>추출 구간 없음</li>'
            lists.append(f"<section><h3>{label}: 추출 {s['reference_occurrence_count']}회 / 서로 다른 표기 {s['unique_reference_text_count']}개</h3><p>질문 {s['question_reference_count']} · 보기 {s['option_reference_count']} · 원시 따옴표 {s['raw_quote_count']}</p><ul>{details}</ul></section>")
        note=' · '.join(r['review_flags']) or '추출 수량 차이 없음 (미검수)'
        textcols=[]
        for key in ['query_fields','gui_fields']:
            textcols.append('<section>'+''.join(f'<p><b>{f}</b> {esc(r[key][f])}</p>' for f in FIELDS)+'</section>')
        issue=f"<p>{esc(r['source_issue']['reason'])}</p>" if r['source_issue'] else ''
        cards.append(f"<article data-priority='{r['review_priority']}'><h2>{esc(r['parallel_id'])} <small>{esc(r['text_dependency'])}</small></h2><p>우선순위 {r['review_priority']} · {esc(note)}</p>{issue}<div class='cols'>{''.join(lists)}</div><details><summary>원문 질문·보기와 중국어 화면</summary><div class='cols'>{''.join(textcols)}</div><img loading='lazy' src='../../{esc(r['image_path'],quote=True)}'></details></article>")
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>영어→중국어 객체 추출 검수</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182b40}}article{{background:white;padding:22px;margin:20px 0;border:1px solid #ccd6e0;border-radius:8px}}h2{{font-size:20px}}h3{{font-size:17px}}small{{font-weight:400}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}li{{overflow-wrap:anywhere}}img{{max-width:400px;max-height:650px}}select{{font:inherit;padding:8px}}@media(max-width:700px){{.cols{{grid-template-columns:1fr}}}}</style>
<p><a href="6개언어조합_추출검수.html">최신 검수 화면: 6개 비방향성 언어 조합 · 우선순위 1·2 · 판정 저장</a></p>
<h1>영어→중국어 객체 추출 검수 (이전 화면)</h1><p>{len(focus)}문항 · 원문, 추출 위치·표현, 언어별 개수를 함께 기록했습니다. 동일 표현이 여러 번 나오면 추출 횟수에 각각 포함됩니다. 서로 다른 표기 수는 실제 객체 수와 다를 수 있습니다.</p>
<p>원시 따옴표 수는 보완 전 표기이고, 추출 수는 따옴표 누락·순서 차이 등을 보완한 뒤의 값입니다. 수량 일치는 의미 대응의 정확성을 보장하지 않습니다. 이 화면에서 영어→중국어를 확인해도 태국어·러시아어가 검증된 것으로 처리하지 않습니다.</p>
<label>표시 <select id="priority"><option value="all">전체</option><option value="1">우선순위 1: 원본 문제·대응 불일치</option><option value="2">우선순위 2: 추출 표기 차이</option><option value="3">우선순위 3: 나머지</option></select></label><span id="count"></span>
{''.join(cards)}<script>const s=document.querySelector('#priority');function filter(){{let n=0;document.querySelectorAll('article').forEach(a=>{{a.hidden=s.value!=='all'&&a.dataset.priority!==s.value;if(!a.hidden)n++}});document.querySelector('#count').textContent=' '+n+'문항';}}s.onchange=filter;filter();</script></html>'''
    (out/'영어_중국어_추출검수.html').write_text(page,encoding='utf-8')
