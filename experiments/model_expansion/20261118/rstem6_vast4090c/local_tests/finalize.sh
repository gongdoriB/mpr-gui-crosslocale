#!/usr/bin/env bash
# Builds the addon tar and writes its sha256 into queue_rstem6_c.sh (ADDON_SHA=...). Then SHA256SUMS of the delivery. Run after the last change of any addon file.
set -euo pipefail
cd "$(dirname "$0")/.."
bash build_addon.sh > local_tests/build_addon_result.txt
SHA=$(shasum -a 256 rstem6_addon_20261118.tar.gz | cut -d' ' -f1)
sed -i.bak -E "s/^ADDON_SHA=[0-9a-f_A-Z]+/ADDON_SHA=$SHA/" queue_rstem6_c.sh && rm -f queue_rstem6_c.sh.bak
grep -n '^ADDON_SHA=' queue_rstem6_c.sh
bash -n queue_rstem6_c.sh
