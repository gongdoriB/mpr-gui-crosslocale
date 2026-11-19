#!/usr/bin/env bash
# Build minicpm_addon_20261114.tar.gz from the addon tree. Usage: build_addon.sh <addon_dir> <out_dir>
set -euo pipefail
ADDON="$1"; OUT="$2"; mkdir -p "$OUT"
cd "$ADDON"
find . -name __pycache__ -prune -exec rm -rf {} \; -o -name .DS_Store -delete 2>/dev/null || true
find followup_rel_4lang -type f ! -name ADDON_SHA256SUMS | sort | xargs shasum -a 256 > followup_rel_4lang/minicpm/ADDON_SHA256SUMS
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls -czf "$OUT/minicpm_addon_20261114.tar.gz" followup_rel_4lang
cd "$OUT"; shasum -a 256 minicpm_addon_20261114.tar.gz | tee MINICPM_ADDON_SHA256.txt
