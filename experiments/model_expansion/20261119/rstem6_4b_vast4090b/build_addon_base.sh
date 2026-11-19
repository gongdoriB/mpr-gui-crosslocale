#!/usr/bin/env bash
# Builds rstem6_addon_20261118.tar.gz from addon/ (SHA256SUMS of addon/followup_rel_6lang/rstem6 is regenerated first). --sums-only: only the SHA256SUMS.
# The tar holds only followup_rel_6lang/rstem6/ (new folder, nothing else). A rebuilt tar changes the sha256 -> ADDON_SHA in queue_rstem6_c.sh must follow.
set -euo pipefail
cd "$(dirname "$0")"
A=addon/followup_rel_6lang/rstem6
find "$A" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
( cd "$A" && find . -type f ! -name SHA256SUMS | sort | sed 's|^\./||' | LC_ALL=C xargs shasum -a 256 > SHA256SUMS )
[ "${1:-}" = "--sums-only" ] && { echo "SHA256SUMS: $(wc -l < $A/SHA256SUMS) files"; exit 0; }
OUT=rstem6_addon_20261118.tar.gz
export COPYFILE_DISABLE=1
( cd addon && tar -cf - followup_rel_6lang/rstem6 | gzip -n -9 > "../$OUT" )
shasum -a 256 "$OUT"; tar -tzf "$OUT" | head -40
