"""Generate concrete ORCF texts from parallel references, retaining provenance.

Generated texts are NOT represented as human-reviewed or screenshot-verified.
Source QAS and the manually reviewable annotation file remain immutable.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
LANGS4 = ('en', 'zh', 'th', 'ru')
FRJA_LANGS = ('fr', 'ja')
LANGS = LANGS4 + FRJA_LANGS
# fr/ja surfaces aligned to the English role order (frja_alignment.json); None = no counterpart -> locale excluded for that item.
FRJA = json.loads((Path(__file__).resolve().parent/'frja_alignment.json').read_text(encoding='utf-8'))
FIELDS = ('question_stem', 'A', 'B', 'C', 'D')
QUOTES = re.compile(r'“([^”]+)”|«([^»]+)»|「([^」]+)」|"([^"\n]+)"|(?<![A-Za-z0-9])\x27(.+?)\x27(?![A-Za-z0-9])')


def quoted(text):
    result = []
    for m in QUOTES.finditer(text):
        n = next(i for i in range(1, 6) if m.group(i) is not None)
        result.append({'start': m.start(n), 'end': m.end(n), 'text': m.group(n)})
    return result


def locate(text, surfaces):
    spans = []
    for n, surface in enumerate(surfaces):
        matches = [m for m in re.finditer(re.escape(surface), text)
                   if not any(m.start() < s['end'] and s['start'] < m.end() for s in spans)]
        if not matches:
            raise ValueError(f'Cannot locate {surface!r} in {text!r}')
        m = matches[0]
        spans.append({'start': m.start(), 'end': m.end(), 'text': surface, 'role': n})
    return spans


# Explicit role alignment, rather than blindly pairing quotation order.
# Each list is in semantic-role order and can differ from sentence order.
OVERRIDES = {}


def override(i, fields, en, zh, th, ru):
    for field in fields.split(','):
        OVERRIDES[i, field] = dict(zip(LANGS4, [en, zh, th, ru]))


override(27, 'question_stem', ['London'], ['伦敦'], ['ลอนดอน'], ['Лондон'])
override(34, 'question_stem', ['Notifications'], ['通知'], ['การแจ้งเตือน'], ['Уведомления'])
override(50, 'question_stem', ['Clear 202 MB','Cache Junk'], ['清理 202 MB','缓存垃圾'], ['ล้างข้อมูล 202 MB','แคชที่ไม่มีประโยชน์'], ['Удалить 202 МБ','Очистка кэша'])
override(50, 'A,B,C,D', ['Cache Junk'], ['缓存垃圾'], ['แคชที่ไม่มีประโยชน์'], ['Очистка кэша'])
override(70, 'question_stem,A,B,C,D', ['Cancel'], ['取消'], ['ยกเลิก'], ['Отменить'])
override(117, 'question_stem', ['Text Size','Display & Brightness'], ['文字大小','显示与亮度'], ['ขนาดข้อความ','จอภาพและความสว่าง'], ['Размер текста','Экран и яркость'])
override(125, 'question_stem', ['Recorder'], ['录音机'], ['บันทึกเสียง'], ['Диктофон'])
override(144, 'question_stem', ['PHOTO'], ['照片'], ['รูปภาพ'], ['ФОТО'])
override(172, 'question_stem', ['Repeat','Set reminder'], ['重复提醒','提醒时间'], ['วนซ้ำ','ตั้งการเตือน'], ['Повтор','Установите напоминание'])
override(223, 'question_stem', ['Thursday'], ['周四'], ['วันพฤหัสบดี'], ['Чт, 12'])
override(287, 'question_stem', ['1x','Natural Light'], ['1x','自然光'], ['1x','แสงไฟธรรมชาติ'], ['1x','ЕСТЕСТВЕННЫЙ СВЕТ'])
override(296, 'question_stem', ['Paris'], ['巴黎'], ['ปารีส'], ['Париж'])
override(299, 'A,B,C,D', ['Voicemail'], ['语音信箱'], ['บริการรับฝากข้อ...'], ['Голосовая поч...'])

# Missing quote marks in otherwise parallel option fields.
override(14, 'B', ['My Files'], ['文件管理'], ['ไฟล์ของฉัน'], ['Мои файлы'])
override(80, 'C', ['Files'], ['文件'], ['แอปไฟล์'], ['Файлы'])
override(81, 'D', ['10-day forecast'], ['10日天气预报'], ['พยากรณ์อากาศ 10 วัน'], ['Прогноз на 10 дней'])
override(87, 'D', ['Do Not Disturb'], ['勿扰模式'], ['ไม่ได้เล่นอยู่'], ['Не беспокоить'])
override(93, 'B', ['Year'], ['年'], ['ปี'], ['Год'])
override(99, 'B', ['Home'], ['通勤'], ['เริ่ม'], ['Регулярно'])
override(99, 'C', ['Default'], ['默认'], ['ค่าเริ่มต้น'], ['По умолчанию'])
override(99, 'D', ['Transit'], ['公交'], ['ขนส่งสาธารณะ'], ['Транспорт'])
override(108, 'A', ['10-DAY FORECAST'], ['10日天气预报'], ['พยากรณ์อากาศ 10 วัน'], ['ПРОГНОЗ НА 10 ДНЕЙ'])
override(108, 'B', ['AIR QUALITY'], ['空气质量'], ['คุณภาพอากาศ'], ['КАЧЕСТВО ВОЗДУХА'])
override(119, 'A', ['Tap to capture'], ['触屏拍照'], ['แตะเพื่อถ่ายภาพ'], ['Снимок касанием'])
override(119, 'B', ['Shooting methods'], ['拍摄方式'], ['วิธีการถ่าย'], ['методов съемки'])
override(119, 'C', ['Tap anywhere in the viewfinder'], ['点击预览画面拍摄照片'], ['แตะที่ใดก็ได้ในช่องมองภาพ'], ['Коснитесь в любом месте видоискателя'])
override(123, 'A', ['Search'], ['搜索'], ['ค้นหาเพลง'], ['Поиск треков'])
override(123, 'B', ['Recent'], ['最近播放'], ['ล่าสุด'], ['Недавние'])
override(123, 'C', ['Local'], ['本地音乐'], ['ในเครื่อง'], ['На устройстве'])
override(123, 'D', ['Favorites'], ['收藏'], ['รายการโปรด'], ['Избранные'])
override(127, 'C', ['About Blood Oxygen'], ['关于血氧'], ['เกี่ยวกับออกซิเจนในเลือด'], ['Подробнее о кислороде в крови'])
override(127, 'D', ['Education'], ['教育'], ['การศึกษา'], ['Информация'])
override(130, 'A', ['10-day forecast'], ['10日天气预报'], ['การพยากรณ์อากาศ 10 วัน'], ['Прогноз на 10 дней'])
override(130, 'B', ['air quality map'], ['空气质量地图'], ['แผนที่คุณภาพอากาศ'], ['Качество воздуха на карте'])
override(130, 'C', ['daily summary'], ['每日概要'], ['สรุปรายวัน'], ['Ежедневная сводка'])
override(133, 'D', ['Phone Manager'], ['手机管家'], ['ตัวจัดการโทรศัพท์'], ['Диспетчера телефона'])
override(145, 'B', ['Recorder'], ['录音'], ['โปรแกรมบันทึก'], ['Диктофон'])
override(154, 'D', ['Share'], ['分享'], ['แชร์'], ['Поделиться'])
override(165, 'D', ['Recents'], ['最近'], ['ล่าสุด'], ['Recents'])
override(176, 'A', ['Create Channel'], ['创建频道'], ['สร้างช่อง'], ['СОЗДАТЬ КАНАЛ'])
override(182, 'B', ['Maps'], ['地图'], ['แผนที่'], ['Карты'])
override(193, 'B', ['No harmful apps found'], ['未发现任何有害应用'], ['ไม่พบแอปที่เป็นอันตราย'], ['Вредоносные приложения не найдены'])
override(193, 'C', ['Learn more'], ['了解详情'], ['ดูข้อมูลเพิ่มเติม'], ['Узнать больше'])
override(210, 'A', ['Back'], ['返回'], ['ย้อนกลับ'], ['Назад'])
override(210, 'B', ['Edit'], ['编辑'], ['แก้ไข'], ['Править'])
override(210, 'C', ['Ringtones'], ['铃声'], ['เสียงเรียกเข้า'], ['Рингтоны'])
override(210, 'D', ['Pick a song'], ['选取歌曲'], ['เลือกเพลง'], ['Выбор песни'])
override(211, 'B', ['Tomorrow 20:00'], ['明天 20:00'], ['พรุ่งนี้ 20:00'], ['Завтра 20:00'])
override(211, 'C', ['Search for tasks'], ['搜索待办'], ['ค้นหางาน'], ['Поиск задач'])
override(218, 'C', ['History'], ['历史记录'], ['ประวัติการเข้าชม'], ['История'])
override(218, 'D', ['Your videos'], ['我的视频'], ['วิดีโอที่ชอบ'], ['ваших видео'])
override(221, 'A', ['Load events from 2027'], ['加载2027年日程'], ['โหลดกำหนดการจาก 2027'], ['Загрузить данные из 2027'])
override(221, 'B', ['Home'], ['主页'], ['หน้าแรก'], ['Home'])
override(221, 'C', ['Day'], ['日期'], ['วัน'], ['День'])
override(274, 'D', ['AI scene enhancement'], ['AI场景增强'], ['การเพิ่มประสิทธิภาพฉาก AI'], ['Улучшение сцены ИИ: вкл.'])
override(278, 'B', ['Radar'], ['雷达'], ['เรดาร์'], ['Радар'])
override(283, 'C', ['Maps'], ['地图'], ['แผนที่'], ['Карты'])
for field, en, zh, th, ru in zip('ABCD', ['Tuesday','Wednesday','Friday','Sunday'], ['周二','周三','周五','周日'], ['วันอังคาร','วันพุธ','วันศุกร์','วันอาทิตย์'], ['Вт, 10','Ср, 11','Пт, 13','Вс, 8']):
    override(223, field, [en], [zh], [th], [ru])
for field, en, zh, th, ru in zip('ABCD', ['Sun 8','Wed 11','Fri 13','Tue 10'], ['8周日','11周三','13周五','10周二'], ['อา. 8','พ. 11','ศ. 13','อ. 10'], ['Вс 8','Ср 11','Пт 13','Вт 10']):
    override(159, field, [en], [zh], [th], [ru])
for field, en, zh, th, ru in zip('ABCD', ['D','W','M','Y'], ['日','周','月','年'], ['D','W','M','Y'], ['ДН','НЕД','МЕС','ГОД']):
    override(164, field, [en], [zh], [th], [ru])
for field in 'ABCD':
    # Three semantic roles; Russian does not retain the English/Chinese role order in C.
    override(247, field, ['Search','No Orders','Learn More'], ['搜索','无订单','进一步了解'], ['ค้นหา','ไม่มีคำสั่งซื้อ','เรียนรู้เพิ่มเติม'], ['Поиск','Нет заказов','Подробнее'])

# These options name interface entries even without quote marks.
FULL_LABEL_ITEMS = set(map(int, '''2 3 5 9 10 12 16 18 20 21 22 23 26 27 30 37 42 43 44 45 48 51 52 55 56 57 58 59 62 64 68 69 72 73 77 78 79 83 85 88 89 94 97 98 100 101 102 104 105 106 107 110 112 117 118 120 121 125 126 128 129 131 132 134 138 139 140 141 143 144 147 149 151 152 156 157 161 168 170 173 174 175 181 182 185 187 189 190 191 192 196 197 198 202 204 207 215 217 225 228 230 232 236 238 241 243 253 255 257 261 264 267 268 269 270 271 272 275 280 282 284 285 288 289 294 295 296 297'''.split()))
EXTRA_LABEL_FIELDS = {
    6:'BCD',11:'B',14:'ABD',33:'ACD',34:'BD',60:'A',65:'AD',66:'ABD',67:'ABCD',74:'ABD',75:'ABC',
    76:'ABCD',80:'AD',86:'ABCD',93:'AC',103:'AB',109:'ABC',133:'ABC',135:'ABC',145:'A',150:'ABCD',
    154:'ABC',163:'ABCD',164:'ABCD',165:'ABC',166:'ABC',176:'A',184:'ABC',186:'ABD',193:'A',
    194:'ABCD',195:'ACD',199:'ABC',201:'ABD',205:'AB',216:'ABC',218:'B',220:'ABCD',231:'ABCD',
    251:'ABD',274:'B',278:'ACD',283:'AB',291:'AB',
}


def label_span(text, lang):
    spans = quoted(text)
    if len(spans) == 1:
        return spans
    # Remove descriptive wrappers, retaining their original offsets/context.
    rules = {
        'en': (r'^(?:The |the |A |An )', r'\s+(?:button|option|section title|section|label|text|icon|tab|link|field|checkbox|dropdown|toggle switch|toggle|progress bar|challenge|countdown text)$'),
        'zh': (r'^', r'(?:按钮|选项|部分|标签|文本|图标|链接|字段|复选框|下拉菜单|开关)$'),
        'th': (r'^(?:ปุ่ม|ตัวเลือก|ส่วน|ป้าย|ข้อความ|ไอคอน|แท็บ|ลิงก์|ช่องทำเครื่องหมาย|รายการเลือก)\s*', r'$'),
        'ru': (r'^(?:Кнопка|кнопка|Опция|опция|Раздел|раздел|Метка|метка|Текст|текст|Значок|значок|Ссылка|ссылка|Флажок|Выпадающий список)\s+', r'\s+(?:текст|значок|кнопка|раздел)$'),
    }
    from frja_wrap_rules import FRJA_WRAP
    rules.update(FRJA_WRAP)
    start, end = 0, len(text)
    prefix, suffix = rules[lang]
    m = re.search(prefix, text)
    if m: start = m.end()
    m = re.search(suffix, text[:end], re.I if lang == 'en' else 0)
    if m and m.start() > start: end = m.start()
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end-1].isspace(): end -= 1
    if start == end:
        start, end = 0, len(text)  # e.g. the actual label is simply "Link".
    return [{'start': start, 'end': end, 'text': text[start:end]}]


def apply_frja(kind, index, field, fields, spans, excluded):
    """Fill fr/ja spans. Explicit FRJA entries win; otherwise the same automatic rule as the 4 languages."""
    key = f'{kind}:{index}:{field}'
    if key in FRJA:
        for l in FRJA_LANGS:
            surf = FRJA[key][l]
            if surf is None:
                excluded.add(l); spans[l] = None
            else:
                spans[l] = locate(fields[l][field], surf)
        return 'frja_explicit_alignment'
    return None


def check_spans(index, field, fields, spans, excluded):
    live = [l for l in LANGS if l not in excluded or spans.get(l) is not None]
    live = [l for l in live if spans.get(l) is not None]
    counts = {len(spans[l]) for l in live}
    if len(counts) != 1:
        raise ValueError(f'Unresolved role count: {index} {field} {spans}')
    return live


def select_spans(index, fields, kind='dep'):
    result = {l: {} for l in LANGS}
    methods = {}
    excluded = set()
    for field in FIELDS:
        if (index, field) in OVERRIDES:
            assert f'{kind}:{index}:{field}' in FRJA, ('override without fr/ja alignment', index, field)
            spans = {l: locate(fields[l][field], OVERRIDES[index,field][l]) for l in LANGS4}
            method = 'explicit_parallel_role_mapping'
            auto = None
        elif field != 'question_stem' and index in FULL_LABEL_ITEMS:
            auto = lambda l: (quoted(fields[l][field]) or [{'start':0,'end':len(fields[l][field]),'text':fields[l][field]}])
            method = 'whole_option_label'
        elif field != 'question_stem' and field in EXTRA_LABEL_FIELDS.get(index,''):
            auto = lambda l: label_span(fields[l][field], l)
            method = 'unquoted_option_label_with_wrapper_rule'
        elif index == 224 and field != 'question_stem':
            auto = lambda l: locate(fields[l][field], [s.strip() for s in fields[l][field].split('>')])
            method = 'navigation_path_labels'
        else:
            auto = lambda l: quoted(fields[l][field])
            method = 'parallel_quoted_references'
        if auto is not None:
            spans = {l: auto(l) for l in LANGS4}
        frja_method = apply_frja(kind, index, field, fields, spans, excluded)
        if frja_method is None:
            assert auto is not None
            for l in FRJA_LANGS:
                spans[l] = auto(l)
        live = check_spans(index, field, fields, spans, excluded)
        for l in LANGS:
            if spans.get(l) is None:
                result[l][field] = None
                continue
            cursor = 0
            for s in sorted(spans[l], key=lambda x:x['start']):
                if not (cursor <= s['start'] < s['end'] <= len(fields[l][field])) or not s['text'].strip():
                    raise ValueError(f'Invalid/empty/overlapping span: {index} {l} {field} {s}')
                cursor = s['end']
            for n, s in enumerate(spans[l]):
                s['ref_id'] = f'{field}:{s.get("role",n)}'
                s.pop('role', None)
            result[l][field] = spans[l]
        methods[field] = method if frja_method is None else method + '+' + frja_method
    for l in excluded:
        result[l] = None
    return result, methods, sorted(excluded)


def replace(text, spans, values):
    for s in sorted(spans, key=lambda x:x['start'], reverse=True):
        text = text[:s['start']] + values[s['ref_id']] + text[s['end']:]
    return text


# Observed differences in the supplied parallel corpus, not introduced translations.
SOURCE_ISSUES = {
    24: {'languages':['zh'], 'reason':'Hotel name and rating differ from other locales; check screenshot-equivalent referent.'},
    29: {'languages':['zh'], 'reason':'App vault is paired with 桌面布局 (home screen layout); semantic correspondence needs adjudication.'},
    87: {'languages':['th'], 'reason':'Do Not Disturb option is paired with ไม่ได้เล่นอยู่ (Not Playing).'},
    99: {'languages':['en','zh','th','ru'], 'reason':'Home / 通勤 / เริ่ม / Регулярно do not denote a straightforward translation-equivalent label.'},
    124: {'languages':['ru'], 'reason':'Standard recording 2 vs Standard recording 1; verify intended screen instance.'},
    218: {'languages':['th'], 'reason':'Your videos is paired with liked videos in the Thai option.'},
    238: {'languages':['th'], 'reason':'Thai A-D are Sat/Sun/Mon/Tue; other locales use Mon/Tue/Wed/Thu. Thai screenshot confirms shifted weekday labels. Cross-locale A-D cannot be assumed semantically aligned.'},
}


# Added for the 6-language extension; kept separate so 4-language rows stay byte-identical.
FRJA_SOURCE_ISSUES = {
    87: {'languages':['ja'], 'reason':'Japanese option D is 再生停止中 (not playing), not Do Not Disturb (6-language extension).'},
    99: {'languages':['fr','ja'], 'reason':'Home/Transit option labels are not straightforward translation equivalents; Japanese 交通状況 means traffic conditions (6-language extension).'},
    221: {'languages':['fr'], 'reason':'French option A is Actualiser (refresh), not Load events from 2027 (6-language extension).'},
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    manifest_path = HERE/'data/manifest_302_6lang.json'
    manifest = json.loads(manifest_path.read_text())
    originals = [json.loads(s) for s in (HERE/'data/original_pairs.jsonl').read_text().splitlines()]
    lookup = {(r['parallel_id'],r['query_language'],r['gui_language']):r for r in originals}
    mappings, rows, groups = [], [], []
    for i,item in enumerate(manifest['items']):
        fields = {l:{'question_stem':item['locales'][l]['question_stem'],**item['locales'][l]['options']} for l in LANGS}
        spans,methods,excluded = select_spans(i,fields,'dep')
        values = {l:{s['ref_id']:s['text'] for f in FIELDS for s in spans[l][f]} for l in LANGS if l not in excluded}
        assert values['en'], f'No references extracted from item {i}'
        mapping = {'parallel_id':item['parallel_id'],'fields':fields,'spans':spans,'methods':methods,
                   'label_provenance':'parallel_qas_text_proxy','screenshot_verified':False,'human_reviewed':False,
                   'source_issue':SOURCE_ISSUES.get(i),'excluded_locales':excluded}
        mappings.append(mapping)
        for q in LANGS:
            for g in LANGS:
                if q in excluded or g in excluded:
                    continue  # no counterpart for a referenced name in this locale
                original = lookup[item['parallel_id'],q,g]
                conditions = {}
                issue = SOURCE_ISSUES.get(i)
                relevant_issue = issue if issue and ({q,g}&set(issue['languages'])) and q!=g else None
                extra = FRJA_SOURCE_ISSUES.get(i)
                if relevant_issue is None and extra and ({q,g}&set(extra['languages'])) and q!=g:
                    relevant_issue = extra
                for c,context,refs in [('O',q,q),('R',q,g),('C',g,q),('F',g,g)]:
                    transformed = {f:replace(fields[context][f],spans[context][f],values[refs]) for f in FIELDS}
                    for f in FIELDS:
                        # Assert unchanged non-reference context, even for reversed role order.
                        assert all(s['text']==fields[context][f][s['start']:s['end']] for s in spans[context][f])
                    if c=='O': assert transformed==fields[q]
                    if c=='F': assert transformed==fields[g]
                    text = transformed['question_stem']+' '+ ' '.join(f'{f}: {transformed[f]}' for f in 'ABCD')
                    row = {**original,'condition':c,'status':'generated_parallel_reference_variants',
                        'question_stem':transformed['question_stem'],'options':{f:transformed[f] for f in 'ABCD'},
                        'question':text,'context_language':context,'reference_language':refs,
                        'referent_alignment':int(c in 'RF'),'context_localization':int(c in 'CF'),
                        'primary_analysis':q!=g,'screenshot_verified':False,'human_reviewed':False,
                        'label_provenance':'parallel_qas_text_proxy','source_issue':relevant_issue,
                        'eligible_without_source_adjudication':relevant_issue is None,
                        'changed_fields_from_O':[f for f in FIELDS if transformed[f]!=fields[q][f]]}
                    rows.append(row)
                    conditions[c] = transformed
                groups.append({'parallel_id':item['parallel_id'],'query_language':q,'gui_language':g,
                    'image_path':original['image_path'],'gold_answer':original['gold_answer'],
                    'conditions':conditions,'source_issue':relevant_issue})
    out = HERE/'data/variants'
    out.mkdir(exist_ok=True)
    from extraction_audit import export_extraction_audit
    export_extraction_audit(out, rows, groups, mappings, quoted)
    for name,records in [('conditions_6lang.jsonl',rows),('quartets_6lang.jsonl',groups),('reference_mappings.jsonl',mappings)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    report = {'items':302,'languages':list(LANGS),'pairs':len(groups),'condition_rows':len(rows),
        'rows_by_condition':dict(Counter(r['condition'] for r in rows)),
        'mismatch_condition_rows':sum(r['primary_analysis'] for r in rows),
        'reference_occurrences':sum(len(s) for m in mappings for l in LANGS if m['spans'][l] for s in m['spans'][l].values()),
        'excluded_item_locales':[{'parallel_id':m['parallel_id'],'locales':m['excluded_locales']} for m in mappings if m['excluded_locales']],
        'items_with_no_question_references':[m['parallel_id'] for m in mappings if not m['spans']['en']['question_stem']],
        'known_source_issue_items':len(set(SOURCE_ISSUES)|set(FRJA_SOURCE_ISSUES)),'known_source_issue_pairs':sum(bool(g['source_issue']) for g in groups),
        'R_equals_O_mismatch_pairs':sum(g['query_language']!=g['gui_language'] and g['conditions']['O']==g['conditions']['R'] for g in groups),
        'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'human_reviewed':False,'screenshot_verified_all':False,
        'construction':'O=source original; R=source context+target parallel references; C=target parallel context+source references; F=target parallel original',
        'limitations':'GUI labels use existing localized QAS as proxies. No claim of exhaustive screenshot-label or translation validation. Known semantic mismatches are flagged in every affected row.'}
    write_json(out/'generation_report.json',report)
    write_json(out/'source_issues.json',[{'parallel_id':manifest['items'][i]['parallel_id'],**v} for i,v in SOURCE_ISSUES.items()])
    write_json(out/'source_issues_frja.json',[{'parallel_id':manifest['items'][i]['parallel_id'],**v} for i,v in FRJA_SOURCE_ISSUES.items()])
    make_viewer(groups, report, out)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def make_viewer(groups, report, out):
    payload = json.dumps(groups,ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>O · R · C · F 전체 변형본</title>
<style>body{font:16px/1.6 system-ui;margin:32px;color:#16283e;background:#f6f8fa}h1{margin-bottom:4px}header,p{max-width:1100px}select,button,input{font:inherit;padding:8px;margin:4px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd3dc;padding:14px;text-align:left;vertical-align:top}th{background:#e9eef4}td:first-child{font-weight:700;width:36px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:24px}img{max-width:100%;max-height:75vh;object-fit:contain}mark{background:#fff0aa}small{color:#526174}#warning{color:#8c3b0a}details{margin-top:8px}li{margin:3px 0}@media(max-width:900px){.layout{display:block}}</style>
<h1>O · R · C · F 전체 변형본</h1><p>302문항 · 4개 언어 · 4,832개 언어쌍 · 19,328개 실제 문장. 기존 병렬 QAS의 참조 표현으로 생성했습니다. 화면 레이블의 전체 검수는 완료하지 않았습니다.</p>
<label>문항 <select id="item"></select></label><label>질의 <select id="q"></select></label><label>GUI <select id="g"></select></label><button id="prev">이전</button><button id="next">다음</button>
<p id="info"></p><p id="warning"></p><div class="layout"><table><thead><tr><th>조건</th><th>질문 및 보기</th></tr></thead><tbody id="body"></tbody></table><aside><img id="screen"><p><small>네 조건에 동일하게 사용하는 GUI 화면</small></p></aside></div>
<script>const data=__DATA__;const ids=[...new Set(data.map(x=>x.parallel_id))];const el=id=>document.getElementById(id);const esc=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
ids.forEach((id,i)=>el('item').add(new Option((i+1)+'. '+id,id)));['q','g'].forEach(k=>['en','zh','th','ru','fr','ja'].forEach(l=>el(k).add(new Option(l,l))));el('g').value='zh';
function draw(){const r=data.find(x=>x.parallel_id===el('item').value&&x.query_language===el('q').value&&x.gui_language===el('g').value);el('body').innerHTML=['O','R','C','F'].map(c=>{const t=r.conditions[c];return '<tr><td>'+c+'</td><td>'+esc(t.question_stem)+'<details open><summary>보기</summary>'+['A','B','C','D'].map(k=>'<div>'+k+'. '+esc(t[k])+'</div>').join('')+'</details></td></tr>'}).join('');el('screen').src='../../'+r.image_path;el('info').textContent=r.parallel_id+' | '+r.query_language+' → '+r.gui_language+' | 원본 정답: '+r.gold_answer;el('warning').textContent=r.source_issue?'원본 대응 주의: '+r.source_issue.reason:'';}
['item','q','g'].forEach(k=>el(k).onchange=draw);el('prev').onclick=()=>{el('item').selectedIndex=Math.max(0,el('item').selectedIndex-1);draw()};el('next').onclick=()=>{el('item').selectedIndex=Math.min(ids.length-1,el('item').selectedIndex+1);draw()};draw();</script></html>'''.replace('__DATA__',payload)
    page = page.replace('302문항 · 4개 언어 · 4,832개 언어쌍 · 19,328개 실제 문장.',
        f"{report['items']:,}문항 · 6개 언어 · {report['pairs']:,}개 언어쌍 · {report['condition_rows']:,}개 실제 문장.")
    page = page.replace("(i+1)+'. '+id,id", "(i+1)+'. '+(data.find(x=>x.parallel_id===id).text_dependency||'dependent')+' | '+id,id")
    page = page.replace("+' | 원본 정답: '+r.gold_answer", "+' | '+(r.text_dependency||'dependent')+' | 원본 정답: '+r.gold_answer+(r.reference_edit_changes_text===false?' | R=O (실제 치환 변화 없음)':'')")
    (out/'전체_변형본_보기.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
