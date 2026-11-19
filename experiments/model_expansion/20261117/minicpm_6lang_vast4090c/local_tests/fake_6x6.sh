#!/usr/bin/env bash
# stand-in for queue_6x6.sh in the flow tests: MODELS=minicpm_v45 expected
[ "${MODELS:-}" = minicpm_v45 ] || { echo "fake 6x6: MODELS wrong"; exit 2; }
[ "${FORCE_SLOW_NET:-}" = 1 ] || { echo "fake 6x6: FORCE_SLOW_NET not passed"; exit 2; }
echo "KEEP_MODEL_CACHE=${KEEP_MODEL_CACHE:-unset}" > "$W/fake_6x6_env.txt"
echo "$(date -u +%FT%TZ) 6x6: queue started: $MODELS"
O="$W/mpr-minicpm/followup_rel_4lang/results/6x6_minicpm_v45"; D="$W/queue_results/6x6_minicpm_v45"
mkdir -p "$O" "$D"; echo '{}' > "$O/full_6x6.jsonl"
tar -czf "$D/6x6_minicpm_v45_test.tar.gz" -C "$(dirname "$O")" 6x6_minicpm_v45; (cd "$D" && sha256sum ./*.tar.gz > SHA256SUMS)
[ "${FAKE_6X6_FAIL:-0}" = 1 ] || touch "$D/COMPLETE"
echo "$(date -u +%FT%TZ) 6x6: QUEUE FINISHED"
