"""PP-OCRv5 (Paddle CUDA via RapidOCR) screen inventories for the 948 images of the 6-language method run.

Screenshot-only: reads image_path / image_sha256 / gui_language from method6_inputs.jsonl, never the question text.
Uses the method branch reader unchanged (followup_rel_4lang/method_dev/gpu_ocr_inventory.CudaReader, ocr_inventory.transcribe)
with the config configs/gpu_ocr_6lang_v1.json (= gpu_ocr_screen_v1.json ppocrv5_cuda + fr -> latin_PP-OCRv5_rec_mobile, ja -> ch_PP-OCRv5_rec_mobile).
One reader per GUI language; every image sha256 is checked before OCR. Resumable (images already in the output are skipped).
Output: inventories6.jsonl (image_sha256, gui_language, image_path, status, visible_strings, original_strings, detections, errors, runtime_ms, ...).
Usage (.venv-ocr):  python run_ocr6.py --repo <method repo> --data-root <harness repo> --inputs <method6_inputs.jsonl> --config <gpu_ocr_6lang_v1.json> --model-dir <dir> --out <inventories6.jsonl> [--only-images file]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True, help='method repo (code)')
    ap.add_argument('--data-root', type=Path, required=True, help='harness repo root; image_path is relative to it')
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only-images', type=Path, default=None, help='optional file with image_sha256 per line (pilot)')
    a = ap.parse_args()
    repo = a.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / '01_code/src')]
    from followup_rel_4lang.method_dev.gpu_ocr_inventory import CudaReader, weights
    cfg = json.loads(a.config.read_text())['backends']['ppocrv5_cuda']
    cfg = {**cfg, 'model_dir': str(a.model_dir.resolve())}
    weights(cfg)                                   # every pinned model file + dictionary sha256, raises on a difference
    images = {}
    for l in a.inputs.read_text(encoding='utf-8').splitlines():
        r = json.loads(l)
        images.setdefault(r['image_sha256'], (r['gui_language'], r['image_path']))
    if a.only_images:
        keep = {l.strip() for l in a.only_images.read_text().splitlines() if l.strip()}
        images = {k: v for k, v in images.items() if k in keep}
    for k, (lang, p) in images.items():
        if lang not in cfg['recognizers']:
            raise ValueError('no recognizer for ' + lang)
        if sha(a.data_root / p) != k:
            raise ValueError('image sha256 differs: ' + p)
    done = set()
    if a.out.exists():
        done = {json.loads(l)['image_sha256'] for l in a.out.read_text(encoding='utf-8').splitlines() if l.strip()}
    todo = {k: v for k, v in images.items() if k not in done}
    print(f'images {len(images)}, done {len(done & set(images))}, to do {len(todo)}', flush=True)
    t0 = time.perf_counter(); n = 0
    with a.out.open('a', encoding='utf-8') as f:
        for lang in sorted({v[0] for v in todo.values()}):
            reader = CudaReader(lang, cfg)
            for k, (lg, p) in sorted(todo.items()):
                if lg != lang:
                    continue
                res = reader.inventory(a.data_root / p)
                f.write(json.dumps({'image_sha256': k, 'gui_language': lang, 'image_path': p,
                                    'recognizer': cfg['recognizers'][lang], **res}, ensure_ascii=False) + '\n'); f.flush()
                n += 1
                print(f'OCR {lang} {n}/{len(todo)} {res["status"]} {len(res["visible_strings"])} strings', flush=True)
            del reader
    rows = [json.loads(l) for l in a.out.read_text(encoding='utf-8').splitlines() if l.strip()]
    got = {r['image_sha256'] for r in rows}
    if not set(images) <= got or len(got) != len(rows):
        raise RuntimeError('inventory incomplete or duplicated')
    status = {}
    for r in rows:
        status[r['status']] = status.get(r['status'], 0) + 1
    print('COMPLETE', len(rows), status, f'{time.perf_counter() - t0:.0f}s this invocation', flush=True)


if __name__ == '__main__':
    main()
