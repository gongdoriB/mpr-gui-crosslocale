#!/usr/bin/env bash
# Local only: rebuild sixbysix_addon_20261114.tar.gz from addon/, write its sha256 into queue_6x6.sh, write SHA256SUMS. Run in this folder.
set -euo pipefail
cd "$(dirname "$0")"
NAME=sixbysix_addon_20261114.tar.gz
( cd addon/sixbysix_addon && find . -type f ! -name SHA256SUMS ! -name '.DS_Store' ! -name '*.pyc' | sed 's|^\./||' | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS )
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon sixbysix_addon
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1)
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_6x6.sh
shasum -a 256 "$NAME" queue_6x6.sh > SHA256SUMS
tar -tzf "$NAME"; cat SHA256SUMS
