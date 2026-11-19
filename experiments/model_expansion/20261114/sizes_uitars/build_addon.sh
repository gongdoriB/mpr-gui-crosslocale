#!/usr/bin/env bash
# Local only: rebuild the addon tar.gz from addon/ and write the sha256 into queue_sizes_uitars.sh. Run in this folder.
set -euo pipefail
cd "$(dirname "$0")"
NAME=model_expansion_sizes_uitars_20261114.tar.gz
( cd addon && find 01_code followup_rel_4lang/model_expansion -type f ! -name SHA256SUMS ! -name '.DS_Store' ! -name '*.pyc' | LC_ALL=C sort | xargs shasum -a 256 > followup_rel_4lang/model_expansion/SHA256SUMS )
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --numeric-owner -czf "$NAME" -C addon 01_code followup_rel_4lang
H=$(shasum -a 256 "$NAME" | cut -d' ' -f1)
echo "$H  $NAME" > ADDON_SHA256.txt
sed -i '' "s/^ADDON_SHA=.*/ADDON_SHA=$H/" queue_sizes_uitars.sh
tar -tzf "$NAME"; echo "$H"
