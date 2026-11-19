"""Download the checksum-pinned PP-OCRv5 Paddle files of configs/gpu_ocr_6lang_v1.json into --model-dir (copy of the method branch's
prepare_gpu_ocr_models.py with the config path and model dir as arguments). Existing files are checked, never overwritten.
Usage: python prepare_models6.py <config.json> <model_dir>"""
import hashlib, json, sys
from pathlib import Path
import requests

cfg = json.loads(Path(sys.argv[1]).read_text())['backends']['ppocrv5_cuda']
folder = Path(sys.argv[2]); folder.mkdir(parents=True, exist_ok=True)
downloads = []
for name, spec in cfg['models'].items():
    downloads.extend((folder / name / f, spec['url'] + '/' + f, s) for f, s in spec['files'].items())
    if 'dictionary_url' in spec:
        downloads.append((folder / Path(spec['dictionary_url']).name, spec['dictionary_url'], spec['dictionary_sha256']))
for path, url, sha in downloads:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Existing model checksum differs: ' + str(path))
        print('CACHED', path.relative_to(folder), flush=True); continue
    tmp = path.with_suffix('.download')
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as e:
            print('retry', attempt + 1, url, e, flush=True)
            if attempt == 2:
                raise
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
        raise ValueError('Downloaded model checksum differs: ' + str(path))
    tmp.replace(path); print('DOWNLOADED', path.relative_to(folder), flush=True)
print('PP-OCRv5 files verified:', len(downloads), flush=True)
