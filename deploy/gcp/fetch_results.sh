#!/usr/bin/env bash
# Pull a finished run's results from GCS and open the HTML report.
#
#   BUCKET=gs://my-bucket RUN_ID=vlbench-20260525-... ./deploy/gcp/fetch_results.sh
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET=gs://your-bucket}
RUN_ID=${RUN_ID:?set RUN_ID=<the run name printed by run_on_gce.sh>}
DEST=${DEST:-runs/$RUN_ID}

mkdir -p "$DEST"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "$DEST/"
echo "downloaded to $DEST"

REPORT="$DEST/report.html"
if [ -f "$REPORT" ]; then
  ( xdg-open "$REPORT" 2>/dev/null \
    || open "$REPORT" 2>/dev/null \
    || start "$REPORT" 2>/dev/null \
    || echo "open this in a browser: $REPORT" )
else
  echo "no report.html found; see $DEST/"
fi
