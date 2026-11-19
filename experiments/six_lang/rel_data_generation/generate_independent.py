"""Extend REL ORCF to the 64 original independent items without relabeling."""
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

from prepare import ACTIVE, parse_question_options, dump, jsonl
from generate_variants import HERE, LANGS, LANGS4, FRJA_LANGS, FIELDS, quoted, locate, replace, make_viewer, apply_frja, check_spans

RULES = {}


def rule(i, fields, en, zh, th, ru):
    for field in fields.split(','):
        RULES[i,field] = dict(zip(LANGS4,[en,zh,th,ru]))


rule(0,'question_stem',['Wi-Fi'],['无线网络'],['Wi-Fi'],['Wi-Fi'])
rule(13,'A',['AI'],['AI'],['AI'],['AI'])
rule(13,'D',['Settings'],['设置'],['การตั้งค่า'],['Настройки'])
rule(14,'question_stem',['Music','Calendar'],['音乐','日历'],['เพลง','ปฏิทิน'],['Музыка','Календарь'])
rule(17,'question_stem',['Aa'],['Aa'],['Aa'],['Aa'])
rule(21,'question_stem',['09:30'],['09:30'],['09:30'],['09:30'])
rule(22,'C',['Start planting today!'],['开始种植今天！'],['เริ่มต้นปลูกวันนี้!'],['Начните выращивать деревья сегодня!'])
rule(22,'D',['Plant'],['种植'],['ปลูก'],['Посадить'])
rule(25,'A',['Reports'],['报告'],['รายงาน'],['отчетов'])
rule(27,'A',['Calendar'],['日历'],['ปฏิทิน'],['Календарь'])
rule(27,'B',['Via'],['Via'],['Via'],['Via'])
rule(27,'C',['Close all'],['关闭所有'],['ปิดทั้งหมด'],['×'])
rule(27,'D',['Deep clean'],['深度清理'],['ตัวทำความสะอาด'],['Глубокая очистка'])
rule(33,'question_stem',['+86 (755) 2393 7079','+86 162 2450 2526'],['+86 (755) 2393 7079','+86 162 2450 2526'],['+86 (755) 2393 7079','+86 162 2450 2526'],['+86 (755) 2393 7079','+86 162 2450 2526'])
rule(35,'question_stem',['Sinus Rhythm'],['窦性心律'],['จังหวะเต้นของเนื้อเยื่อหัวใจกำเนิดไฟฟ้า (Sinus)'],['Синусовый ритм'])
rule(40,'question_stem',['+86 162 6746 0018'],['+86 162 6746 0018'],['+86 162 6746 0018'],['+86 162 6746 0018'])
rule(41,'question_stem',['+86 184 0408 5448'],['+86 184 0408 5448'],['+86 184 0408 5448'],['+86 184 0408 5448'])
rule(44,'question_stem',['Calendar'],['日历'],['ปฏิทิน'],['Календарь'])
rule(46,'question_stem',['0.90KM','Pool Swim','375M'],['0.90公里','泳池游泳','375米'],['0.90กม.','ว่ายน้ำในสระ','375ม.'],['0.90 км','Бассейн','375 м'])
rule(46,'A,B,C,D',['Pool Swim','375M'],['泳池游泳','375米'],['ว่ายน้ำในสระ','375ม.'],['Бассейн','375 м'])
rule(51,'A',['Copy'],['复制'],['คัดลอก'],['Копировать'])
rule(51,'C',['Messages'],['信息'],['จัดความ'],['Сообщения'])
rule(51,'D',['Mail'],['邮件'],['เมล'],['Почта'])
rule(53,'question_stem',['Calls'],['通话'],['การโทร'],['Вызовы'])
rule(54,'A',['Cancel'],['取消'],['ยกเลิก'],['Отменить'])
rule(54,'B',['To:'],['收件人：'],['ถึง:'],['Кому:'])
rule(56,'D',['72'],['71'],['70'],['80'])
rule(59,'question_stem',['Calculator'],['计算器'],['เครื่องคิดเลข'],['Калькулятор'])

WHOLE_OPTIONS = {0,5,6,7,12,15,16,19,21,26,31,32,40,41,44,47,55,58,59,62}
WHOLE_FIELDS = {56:'AB'}
# Quotes around visual shapes/control descriptions are not GUI-label spans.
# Preserve these descriptions as context instead of translating icons as labels.
VISUAL_CONTEXT = {(11,f) for f in FIELDS} | {(28,'C')} | {(34,f) for f in FIELDS} | {(60,f) for f in FIELDS} | {(61,f) for f in FIELDS}

ISSUES = {
    22: {'languages':['zh'], 'reason':'Coin label differs: 569 in Chinese vs 589 in other source locales; verify screen correspondence.'},
    27: {'languages':['ru','th'], 'reason':'Close all vs × and Deep clean vs cleaner are heterogeneous label/icon descriptions.'},
    34: {'languages':['ru'], 'reason':'Russian specifies 02:00...04:11 whereas other locales refer generically to the progress bar; preserved as context, no invented reference alignment.'},
    51: {'languages':['th'], 'reason':'Messages option has Thai source label จัดความ; possible source typo, kept verbatim.'},
    53: {'languages':['zh'], 'reason':'Chinese digit and search phrase differ (9 / 拼音9键搜索 vs 2 / Supports smart search).'},
    60: {'languages':['th','ru','zh'], 'reason':'Thai B/C have empty quoted labels; other locales mix visual descriptions and labels. No fabricated replacement spans.'},
    63: {'languages':['zh'], 'reason':'Chinese question reverses target/reference (three-dot relative to share), while options retain the original referent; requires adjudication.'},
}


FRJA_ISSUES = {
    22: {'languages':['ja'], 'reason':'Japanese option D is 始める (start), not Plant (6-language extension).'},
    27: {'languages':['fr','ja'], 'reason':'Close all is only an x mark in French/Japanese (6-language extension).'},
    56: {'languages':['fr','ja'], 'reason':'Number differs across locales: EN 72, FR 77, JA 74 (6-language extension).'},
}


def extract(i, fields):
    spans={l:{} for l in LANGS};methods={};excluded=set()
    for f in FIELDS:
        auto=None
        if (i,f) in RULES:
            assert f'ind:{i}:{f}' in __import__('generate_variants').FRJA,('rule without fr/ja',i,f)
            ss={l:locate(fields[l][f],RULES[i,f][l]) for l in LANGS4};method='explicit_parallel_role_mapping'
        elif (i,f) in VISUAL_CONTEXT:
            auto=lambda l:[];method='visual_or_generic_description_not_label'
        elif f!='question_stem' and (i in WHOLE_OPTIONS or f in WHOLE_FIELDS.get(i,'')):
            auto=lambda l:(quoted(fields[l][f]) or [{'start':0,'end':len(fields[l][f]),'text':fields[l][f]}]);method='whole_option_label'
        else:
            auto=lambda l:quoted(fields[l][f]);method='parallel_quoted_references'
        if auto is not None: ss={l:auto(l) for l in LANGS4}
        fm=apply_frja('ind',i,f,fields,ss,excluded)
        if fm is None:
            for l in FRJA_LANGS: ss[l]=auto(l)
        check_spans(i,f,fields,ss,excluded)
        for l in LANGS:
            if ss.get(l) is None: spans[l][f]=None; continue
            end=0
            for s in sorted(ss[l],key=lambda x:x['start']):
                assert end<=s['start']<s['end']<=len(fields[l][f])
                assert fields[l][f][s['start']:s['end']]==s['text'] and s['text'].strip()
                end=s['end']
            for n,s in enumerate(ss[l]):s['ref_id']=f'{f}:{s.pop("role",n)}'
            spans[l][f]=ss[l]
        methods[f]=method if fm is None else method+'+'+fm
    for l in excluded: spans[l]=None
    return spans,methods,sorted(excluded)


def enrich(rows, groups, maps, dependency):
    byid={m['parallel_id']:m for m in maps}
    lookup={}
    for m in maps:
        m.update(text_dependency=dependency,dimension='rel')
    for group in groups:
        pid=group['parallel_id'];mapping=byid[pid]
        has_refs=any(mapping['spans']['en'][f] for f in FIELDS)
        seen={};equivalence={}
        for c in 'ORCF':
            key=json.dumps(group['conditions'][c],sort_keys=True,ensure_ascii=False)
            equivalence[c]=seen.setdefault(key,c)
        group.update(text_dependency=dependency,dimension='rel',has_reference_spans=has_refs,
                     reference_edit_changes_text=group['conditions']['R']!=group['conditions']['O'],
                     equivalent_to_condition=equivalence,unique_condition_texts=len(seen))
        lookup[pid,group['query_language'],group['gui_language']]=group
    for row in rows:
        group=lookup[row['parallel_id'],row['query_language'],row['gui_language']]
        row.update(text_dependency=dependency,dimension='rel',has_reference_spans=group['has_reference_spans'],
                   reference_edit_changes_text=group['reference_edit_changes_text'],
                   equivalent_to_condition=group['equivalent_to_condition'][row['condition']],
                   unique_condition_texts=group['unique_condition_texts'])


def export(out,rows,groups,maps,source):
    out.mkdir(exist_ok=True)
    from extraction_audit import export_extraction_audit
    export_extraction_audit(out,rows,groups,maps,quoted)
    jsonl(out/'conditions_6lang.jsonl',rows);jsonl(out/'quartets_6lang.jsonl',groups);jsonl(out/'reference_mappings.jsonl',maps)
    report={'items':len(maps),'languages':list(LANGS),'pairs':len(groups),'condition_rows':len(rows),
        'items_by_dependency':dict(Counter(m['text_dependency'] for m in maps)),
        'rows_by_dependency':dict(Counter(r['text_dependency'] for r in rows)),
        'rows_by_condition':dict(Counter(r['condition'] for r in rows)),
        'mismatch_condition_rows':sum(r['primary_analysis'] for r in rows),
        'items_without_reference_spans':[m['parallel_id'] for m in maps if not any(m['spans']['en'][f] for f in FIELDS)],
        'R_equals_O_mismatch_pairs':sum(not g['reference_edit_changes_text'] and g['query_language']!=g['gui_language'] for g in groups),
        'unique_condition_texts_within_pairs':sum(g['unique_condition_texts'] for g in groups),
        'known_source_issue_pairs':sum(bool(g['source_issue']) for g in groups),
        'human_reviewed':False,'screenshot_verified_all':False,'source':source}
    dump(out/'generation_report.json',report)
    dump(out/'source_issues.json',[{'parallel_id':m['parallel_id'],**m['source_issue']} for m in maps if m.get('source_issue')])
    make_viewer(groups,report,out)
    return report


def main(download=False):
    source_path=ACTIVE/'annotation/rel_text_dependency/data/pilot_manifest.json'
    labels_path=ACTIVE/'annotation/rel_text_dependency/outputs/rel_annotations_integrated_366.csv'
    source=json.loads(source_path.read_text());labels=list(csv.DictReader(labels_path.open(encoding='utf-8-sig')))
    assert len(labels)==len({r['parallel_id'] for r in labels})==366
    digest=hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert {r['manifest_sha256'] for r in labels}=={digest}
    selected={r['parallel_id'] for r in labels if r['text_dependency']=='independent' and r['review_flag']=='0'}
    items=[copy.deepcopy(x) for x in source['items'] if x['parallel_id'] in selected]
    assert len(items)==64
    qa={l:[json.loads(s) for s in (HERE/'sources/qas'/f'rel_el_{l}.jsonl').read_text().splitlines()] for l in LANGS}
    provenance={'annotation_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),'manifest_sha256':digest,**source['source']}
    assets=[];rows=[];groups=[];maps=[]
    for i,item in enumerate(items):
        item.pop('ko_helper',None);item['original_display_order']=item['display_order'];item['display_order']=i+1
        item['text_dependency']='independent';item['locales']={l:item['locales'][l] for l in LANGS}
        for l,loc in item['locales'].items():
            raw=qa[l][loc['source_line']-1];parsed=parse_question_options(raw['question'])
            assert parsed.option_parse_status=='ok' and parsed.question_stem==loc['question_stem'] and parsed.options==loc['options']
            assert raw['image_path'].removeprefix('../images/')==loc['asset'] and raw['answer'] in ['A','B','C','D']
            loc['gold_answer']=raw['answer'];a=loc['asset']
            assets.append({'asset':a,'locale':l,'url':f"https://huggingface.co/datasets/{source['source']['hf_repo']}/resolve/{source['source']['hf_revision']}/{a}"})
        fields={l:{'question_stem':loc['question_stem'],**loc['options']} for l,loc in item['locales'].items()}
        spans,methods,excluded=extract(i,fields)
        values={l:{s['ref_id']:s['text'] for f in FIELDS for s in spans[l][f]} for l in LANGS if l not in excluded}
        maps.append({'parallel_id':item['parallel_id'],'fields':fields,'spans':spans,'methods':methods,
            'label_provenance':'parallel_qas_text_proxy','human_reviewed':False,'screenshot_verified':False,'source_issue':ISSUES.get(i),'excluded_locales':excluded})
        for q in LANGS:
            for g in LANGS:
                if q in excluded or g in excluded: continue
                issue=ISSUES.get(i);issue=issue if issue and q!=g and {q,g}&set(issue['languages']) else None
                extra=FRJA_ISSUES.get(i)
                if issue is None and extra and q!=g and {q,g}&set(extra['languages']): issue=extra
                gold=item['locales'][g]['gold_answer'];qgold=item['locales'][q]['gold_answer']
                pair={'pair_id':f"{item['parallel_id']}::{q}::{g}",'parallel_id':item['parallel_id'],'state_key':item['state_key'],
                    'query_language':q,'gui_language':g,'matched':q==g,'image_path':'assets/images/'+item['locales'][g]['asset'],
                    'gold_answer':gold,'source_query_gold_answer':qgold,'gold_agrees_across_locales':gold==qgold}
                if gold!=qgold:issue={'languages':[q,g],'reason':'Parallel source gold answer disagreement.'}
                conditions={}
                for c,context,refs in [('O',q,q),('R',q,g),('C',g,q),('F',g,g)]:
                    t={f:replace(fields[context][f],spans[context][f],values[refs]) for f in FIELDS}
                    assert all(t.values())
                    if c=='O':assert t==fields[q]
                    if c=='F':assert t==fields[g]
                    conditions[c]=t
                    rows.append({**pair,'condition':c,'status':'generated_parallel_reference_variants',
                        'question_stem':t['question_stem'],'options':{f:t[f] for f in 'ABCD'},'option_order':item['locales'][q]['option_order'],
                        'question':t['question_stem']+' '+' '.join(f'{f}: {t[f]}' for f in 'ABCD'),
                        'context_language':context,'reference_language':refs,'referent_alignment':int(c in 'RF'),'context_localization':int(c in 'CF'),
                        'primary_analysis':q!=g,'screenshot_verified':False,'human_reviewed':False,'label_provenance':'parallel_qas_text_proxy',
                        'source_issue':issue,'eligible_without_source_adjudication':issue is None,
                        'changed_fields_from_O':[f for f in FIELDS if t[f]!=fields[q][f]]})
                groups.append({**pair,'conditions':conditions,'source_issue':issue})
    folder=HERE/'data/independent';folder.mkdir(exist_ok=True)
    dump(folder/'manifest_64_6lang.json',{'schema_version':'followup-orcf-1','languages':list(LANGS),'source':source['source'],
        'selection':{'dimension':'rel','text_dependency':'independent','sample_size':64},'items':items})
    dump(folder/'assets_required.json',assets)
    enrich(rows,groups,maps,'independent')
    report=export(folder,rows,groups,maps,provenance)
    dep=HERE/'data/variants'
    read=lambda name:[json.loads(s) for s in (dep/name).read_text().splitlines()]
    dr,dg,dm=read('conditions_6lang.jsonl'),read('quartets_6lang.jsonl'),read('reference_mappings.jsonl')
    enrich(dr,dg,dm,'dependent')
    assert not ({m['parallel_id'] for m in dm}&selected)
    assert len(dm)+len(maps)==366
    combined=export(HERE/'data/rel_all_366',dr+rows,dg+groups,dm+maps,provenance)
    dep_manifest=json.loads((HERE/'data/manifest_302_6lang.json').read_text())
    for item in dep_manifest['items']:item['text_dependency']='dependent'
    all_items=sorted(dep_manifest['items']+items,key=lambda x:x['original_display_order'])
    for n,item in enumerate(all_items,1):item['display_order']=n
    dump(HERE/'data/rel_all_366/manifest_366_6lang.json',{'schema_version':'followup-orcf-1','languages':list(LANGS),
        'source':source['source'],'selection':{'dimension':'rel','sample_size':366,'text_dependency':['dependent','independent']},'items':all_items})
    config=json.loads((HERE/'config.json').read_text())
    config.update(text_dependency_groups=['dependent','independent'],dimension='rel',
        conditions_file='data/rel_all_366/conditions_6lang.jsonl',manifest_file='data/rel_all_366/manifest_366_6lang.json',
        image_path_base='followup_rel_6lang',condition_equivalence_field='equivalent_to_condition',
        note='Preserve original dependency annotations. Identical conditions are not independent observations. Generated references are source-QAS proxies, not fully screenshot-verified.')
    dump(HERE/'data/rel_all_366/config.json',config)
    dump(HERE/'data/rel_all_366/assets_required.json',json.loads((HERE/'data/assets_required.json').read_text())+assets)
    if download:
        from fetch_images import fetch
        with ThreadPoolExecutor(max_workers=8) as pool:failures=[r for r in pool.map(fetch,assets) if r]
        dump(folder/'image_download_report.json',{'required':len(assets),'available':len(assets)-len(failures),'failed':failures})
        assert not failures,failures
    print(json.dumps({'independent':report,'combined':combined},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download-images',action='store_true')
    main(p.parse_args().download_images)
