"""Local CPU (onnxruntime) PP-OCRv5 coverage check: do the screen names used in R_stem questions appear in OCR strings?

Approximate check only: ONNX CPU, not the Paddle CUDA path of the method. Not part of any result table.
Usage: python ocr_coverage_check.py <repo_root_with_images> <rstem6_conditions.jsonl> <b200_extract_dir> <out.json> [n_per_lang]
"""
import glob, json, re, sys, unicodedata, random
from collections import defaultdict
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion, EngineType

root, cond_path, b200_dir, out_path = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 20
REC = {'en': 'en', 'zh': 'ch', 'ja': 'ch', 'th': 'th', 'ru': 'eslav', 'fr': 'latin'}
QUOTES = re.compile(r"'([^']{1,60})'|\"([^\"]{1,60})\"|«([^»]{1,60})»|「([^」]{1,60})」|“([^”]{1,60})”|‘([^’]{1,60})’|『([^』]{1,60})』")
norm = lambda s: ' '.join(unicodedata.normalize('NFKC', s).split()).casefold()

names = defaultdict(set)  # image_sha -> quoted spans that R_stem introduced (absent from the O stem of the same pair)
meta = {}
rows = defaultdict(dict)
for l in open(cond_path, encoding='utf-8'):
    r = json.loads(l)
    if r['condition'] in ('O', 'R_stem'):
        rows[r['pair_id']][r['condition']] = r
quoted = lambda s: {next(g for g in m.groups() if g) for m in QUOTES.finditer(s)}
for pid, d in rows.items():
    r = d['R_stem']
    new = {q for q in quoted(r['question_stem']) - quoted(d['O']['question_stem']) if q.strip() == q and q in r['question_stem'] and q not in d['O']['question_stem']}
    names[r['image_sha256']] |= new
    meta[r['image_sha256']] = (r['gui_language'], r['image_path'])
b200 = {}
for f in glob.glob(b200_dir + '/**/ppocrv5_cuda/ocr/attempt_0001/inventories.jsonl', recursive=True):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); b200[r['image_sha256']] = r['visible_strings']

engines = {}
def engine(lang):
    if lang not in engines:
        engines[lang] = RapidOCR(params={
            'Global.text_score': 0, 'Global.use_cls': False, 'Global.log_level': 'error',
            'Det.ocr_version': OCRVersion('PP-OCRv5'), 'Det.model_type': ModelType('mobile'), 'Det.lang_type': LangDet('ch'),
            'Det.engine_type': EngineType('onnxruntime'), 'Det.limit_side_len': 960, 'Det.limit_type': 'max',
            'Rec.ocr_version': OCRVersion('PP-OCRv5'), 'Rec.model_type': ModelType('mobile'), 'Rec.lang_type': LangRec(REC[lang]),
            'Rec.engine_type': EngineType('onnxruntime')})
    return engines[lang]

def hit(name, strings):
    t = norm(name)
    return any(t == norm(s) or (len(t) >= 2 and t in norm(s)) for s in strings)

random.seed(20261119)
by_lang = defaultdict(list)
for sha, (lang, path) in meta.items():
    if names[sha]:
        by_lang[lang].append(sha)
report = {}
for lang in ('fr', 'ja', 'en', 'zh', 'th', 'ru'):
    sample = sorted(by_lang[lang]); random.shuffle(sample); sample = sample[:n]
    tot = got_local = got_b200 = b200_n = 0
    examples = []
    for sha in sample:
        out = engine(lang)(f"{root}/{meta[sha][1]}")
        strings = list(out.txts or [])
        for nm in sorted(names[sha]):
            tot += 1
            h = hit(nm, strings); got_local += h
            if sha in b200:
                b200_n += 1; got_b200 += hit(nm, b200[sha])
            if not h and len(examples) < 6:
                examples.append(nm)
    report[lang] = {'images': len(sample), 'names': tot, 'local_onnx_hit': got_local / tot if tot else None,
                    'b200_paddle_hit': got_b200 / b200_n if b200_n else None, 'b200_names': b200_n, 'missed_examples': examples}
    print(lang, json.dumps(report[lang], ensure_ascii=False), flush=True)
json.dump(report, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
