"""6-language run of the final method editor (cells V-D-S and V-D-H) with SELF-EDIT: the evaluated model (--model) is its own text-only editor (editors6.py).

Uses the method branch code unchanged (method-selective-alignment-dev @ 533633c):
  stem_refinement_2x2.extract  -> stem GUI-name extraction, once per unique source stem (source_key, as run_editor_matrix_remaining.py)
  editor_matrix.rewrite(row, inventory, referents, backend, 'D', 'S'|'H') -> V-D-S (stem names only) / V-D-H (stem names + full option translation)
  editor_matrix_backend.TextEditor(kind 'V') for Qwen3-VL; editors6.SelfEditor for UI-TARS / InternVL3.5 (same prompts, greedy, per-call cache)
The only change is a runtime widening of schemas.validate_input from 4 to 6 languages (fr, ja added); every other check of that
function is kept (copied below and compared with the original source at start).

Inputs (no labels, no answers): method6_inputs.jsonl (4,740 rows, allowlisted fields only), inventories6.jsonl (one PP-OCRv5 inventory per image).
Outputs in --out: extractions.jsonl, rewrite_V-D-S.jsonl, rewrite_V-D-H.jsonl, cost.json, COMPLETE (+ cache/ of every editor call).
Usage (python of the model's solver venv; method repo code is put first on sys.path):
  python run_edit6.py --model KEY --repo <method repo> --inputs .. --inventories .. --out .. --editor-path <snapshot dir> [--mock] [--pairs file]
"""
import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

LANGS = ('en', 'zh', 'th', 'ru', 'fr', 'ja')
# Self-edit (decided 2026-11-19): the evaluated model edits its own questions. key -> editor spec (same pinned revisions as the solver runs)
EDITORS = {
    'qwen3_vl_8b':   {'model_id': 'Qwen/Qwen3-VL-8B-Instruct',     'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 'family': 'qwen3_vl'},
    'qwen3_vl_4b':   {'model_id': 'Qwen/Qwen3-VL-4B-Instruct',     'revision': 'ebb281ec70b05090aa6165b016eac8ec08e71b17', 'family': 'qwen3_vl'},
    'uitars_15_7b':  {'model_id': 'ByteDance-Seed/UI-TARS-1.5-7B', 'revision': '683d002dd99d8f95104d31e70391a39348857f4e', 'family': 'qwen2_5_vl'},
    'internvl35_8b': {'model_id': 'OpenGVLab/InternVL3_5-8B',      'revision': '9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79', 'family': 'internvl'},
}
CELLS = ('V-D-S', 'V-D-H')
EXPECTED_ORIGINAL = '''    if row['source_language'] not in ('en','zh','th','ru') or row['gui_language'] not in ('en','zh','th','ru'):'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    lines = [l for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    out = []
    for i, l in enumerate(lines):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print('WARNING: dropped a truncated last line of', path, flush=True)   # killed mid-write; the row is redone
            text = Path(path).read_text(encoding='utf-8')
            Path(path).write_text(text[:text.rstrip('\n').rfind('\n') + 1] if '\n' in text.rstrip('\n') else '', encoding='utf-8')
    return out


def widen_languages(schemas):
    src = inspect.getsource(schemas.validate_input)
    if EXPECTED_ORIGINAL not in src:
        raise RuntimeError('schemas.validate_input differs from the reviewed version; refusing to patch')
    keys = schemas.INPUT_KEYS

    def validate_input(row):
        if set(row) != keys:
            raise ValueError('Method input must contain only the allowlisted original fields')
        if row['source_language'] not in LANGS or row['gui_language'] not in LANGS:
            raise ValueError('Unknown language')
        if list(row['options']) != list('ABCD') or row['option_order'] != list('ABCD'):
            raise ValueError('Option keys/order must be A-D')
        if not isinstance(row['question_stem'], str) or any(not isinstance(v, str) for v in row['options'].values()):
            raise ValueError('All MCQ fields must be strings')
        return row
    schemas.validate_input = validate_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--inventories', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', choices=sorted(EDITORS), required=True, help='evaluated model = its own editor')
    ap.add_argument('--editor-path', default=None, help='local snapshot dir of that model (pinned revision)')
    ap.add_argument('--pairs', type=Path, default=None, help='optional file with one pair_id per line (pilot subset)')
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--min-free-gpu-gib', type=float, default=18.0)
    a = ap.parse_args()
    repo = a.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / '01_code/src')]
    import mpr_crosslocale
    if not Path(mpr_crosslocale.__file__).resolve().is_relative_to(repo):
        raise RuntimeError('mpr_crosslocale is not imported from the method repo: ' + mpr_crosslocale.__file__)
    from mpr_crosslocale.methods.selective_alignment import schemas
    widen_languages(schemas)                       # before the method modules bind validate_input
    from followup_rel_4lang.method_dev import editor_matrix as matrix, stem_refinement_2x2 as refinement
    if not Path(matrix.__file__).resolve().is_relative_to(repo):
        raise RuntimeError('method modules not imported from the method repo: ' + matrix.__file__)
    for mod in (matrix, refinement, matrix.pipeline):
        if getattr(mod, 'validate_input', schemas.validate_input) is not schemas.validate_input:
            raise RuntimeError('validate_input patch did not reach ' + mod.__name__)

    rows = read(a.inputs)
    if a.pairs:
        keep = {l.strip() for l in a.pairs.read_text().splitlines() if l.strip()}
        rows = [r for r in rows if r['pair_id'] in keep]
        if len(rows) != len(keep):
            raise ValueError('pilot pair ids missing from inputs')
    inv = {r['image_sha256']: r for r in read(a.inventories)}
    for r in rows:
        schemas.validate_input(r)
        i = inv.get(r['image_sha256'])
        if i is None or i['gui_language'] != r['gui_language']:
            raise ValueError('inventory missing or GUI language differs for ' + r['pair_id'])
    a.out.mkdir(parents=True, exist_ok=True)
    snapshot = {'inputs_sha256': sha(a.inputs), 'inventories_sha256': sha(a.inventories), 'rows': len(rows),
                'pairs_file_sha256': sha(a.pairs) if a.pairs else None, 'model': a.model, 'editor': EDITORS[a.model], 'cells': CELLS,
                'method_code_sha256': {p: sha(repo / p) for p in (
                    'followup_rel_4lang/method_dev/editor_matrix.py', 'followup_rel_4lang/method_dev/stem_refinement_2x2.py',
                    'followup_rel_4lang/method_dev/editor_matrix_backend.py', 'followup_rel_4lang/method_dev/two_stage_v2/pipeline.py',
                    'followup_rel_4lang/method_dev/two_stage_v2/prompts.py', 'followup_rel_4lang/method_dev/two_stage_v2/legacy.py',
                    'followup_rel_4lang/method_dev/two_stage_v2/backend.py', '01_code/src/mpr_crosslocale/methods/selective_alignment/schemas.py')},
                'runner_sha256': sha(__file__), 'editors6_sha256': sha(Path(__file__).with_name('editors6.py')), 'mock': a.mock}
    cpath = a.out / 'contract.json'
    snapshot = json.loads(json.dumps(snapshot))   # tuples -> lists, as stored
    if cpath.exists() and json.loads(cpath.read_text()) != snapshot:
        raise ValueError('contract.json differs from this run (inputs/code/editor changed); use a new --out')
    cpath.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1))

    config = {'seed': 42, 'min_free_gpu_gib': a.min_free_gpu_gib, 'output_tokens': matrix.LIMITS, 'max_input_tokens': 4096,
              'models': {'editor_vlm': {**{k: v for k, v in EDITORS[a.model].items() if k != 'family'}, 'family': EDITORS[a.model]['family'], 'path': a.editor_path}}}
    if not a.mock and (not a.editor_path or a.editor_path.rstrip('/').split('/')[-1] != EDITORS[a.model]['revision']):
        raise ValueError('--editor-path must be the snapshot folder of the pinned revision ' + EDITORS[a.model]['revision'])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from editors6 import make_editor
    backend = make_editor(EDITORS[a.model]['family'], config, repo, a.out / 'cache', a.mock)
    t0 = time.perf_counter()
    source_key = lambda row: schemas.digest(matrix.prompts.extraction_payload(row, stem_only=True))
    sources = {}
    for r in rows:
        sources.setdefault(source_key(r), r)
    ext_path = a.out / 'extractions.jsonl'
    done = {r['source_key'] for r in read(ext_path)} if ext_path.exists() else set()
    with ext_path.open('a', encoding='utf-8') as f:
        for n, (k, r) in enumerate(sorted(sources.items()), 1):
            if k in done:
                continue
            res = refinement.extract(r, backend)
            f.write(json.dumps({'source_key': k, 'source_language': r['source_language'], **res}, ensure_ascii=False) + '\n'); f.flush()
            print(f'EXTRACT {n}/{len(sources)} {res["status"]}', flush=True)
    extracted = {r['source_key']: r['referents'] for r in read(ext_path)}
    if not set(sources) <= set(extracted):
        raise RuntimeError('incomplete extraction')
    for cell in CELLS:
        p = a.out / f'rewrite_{cell}.jsonl'
        done = {r['pair_id'] for r in read(p)} if p.exists() else set()
        with p.open('a', encoding='utf-8') as f:
            for n, r in enumerate(rows, 1):
                if r['pair_id'] in done:
                    continue
                refs = extracted[source_key(r)]
                res = matrix.rewrite(r, inv[r['image_sha256']], refs, backend, 'D', cell[-1])
                out = res.pop('output')
                rec = {**r, 'question_stem': out['question_stem'], 'options': {k: out[k] for k in 'ABCD'}, 'cell': cell,
                       'modified_question': schemas.render(out), 'extracted_referents': refs,
                       'inventory_status': inv[r['image_sha256']]['status'], 'gold_target_used': False, **res}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                print(f'REWRITE {cell} {n}/{len(rows)} {res["status"]}', flush=True)
    for cell in CELLS:
        got = {r['pair_id'] for r in read(a.out / f'rewrite_{cell}.jsonl')}
        if got != {r['pair_id'] for r in rows}:
            raise RuntimeError('incomplete rewrite ' + cell)
    (a.out / 'cost.json').write_text(json.dumps({
        'wall_seconds_this_invocation': time.perf_counter() - t0, 'model_loads': backend.load_events,
        'new_editor_runtime_ms': sum(c['runtime_ms'] for c in backend.calls if not c['cache_hit']),
        'editor_calls': len(backend.calls), 'cache_hits': sum(c['cache_hit'] for c in backend.calls),
        'output_limit_reached': sum(bool(c.get('output_limit_reached')) for c in backend.calls)}, indent=1))
    (a.out / 'COMPLETE').write_text('complete\n')
    print('COMPLETE', a.out, flush=True)


if __name__ == '__main__':
    main()
