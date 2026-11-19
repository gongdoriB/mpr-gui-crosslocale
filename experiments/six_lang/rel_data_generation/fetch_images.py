"""Download only the 1,208 required screenshots at the source-locked revision."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def fetch(row):
    dest = HERE/'assets/images'/row['asset']
    if dest.exists() and dest.read_bytes()[:3] in (b'\xff\xd8\xff', b'\x89PN'):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix+'.part')
    r = subprocess.run(['curl', '-sfL', '--retry', '2', '--connect-timeout', '15', '--max-time', '90', row['url'], '-o', str(tmp)], capture_output=True)
    if r.returncode or not tmp.exists() or tmp.read_bytes()[:3] not in (b'\xff\xd8\xff', b'\x89PN'):
        tmp.unlink(missing_ok=True)
        return row['asset']
    tmp.replace(dest)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download source-locked images from an asset manifest.')
    parser.add_argument('--manifest', type=Path, default=Path('data/assets_required.json'),
                        help='Path relative to this script folder, or an absolute path.')
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else HERE/args.manifest
    assets = json.loads(manifest.read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = []
        for n, failed in enumerate(pool.map(fetch, assets), 1):
            if failed: failures.append(failed)
            if n % 100 == 0: print(f'{n}/{len(assets)} checked; failures={len(failures)}', flush=True)
    (manifest.parent/'image_download_report.json').write_text(json.dumps({'required': len(assets), 'available': len(assets)-len(failures), 'failed': failures}, indent=2)+'\n')
    print(f'Available: {len(assets)-len(failures)}/{len(assets)}')
    raise SystemExit(bool(failures))
