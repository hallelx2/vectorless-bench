#!/usr/bin/env bash
# All-in-one autonomous VM runner, driven entirely by instance metadata — no
# SSH/IAP needed (so it works cleanly when launched from a GitHub Actions
# workflow). On boot it: installs Docker + gcloud, pulls the staged bundle +
# .env from GCS, builds the image, downloads FinanceBench, runs the benchmark,
# uploads results AND this log to GCS, then powers off (the instance's
# max-run-duration is the deletion backstop).
#
# Required instance metadata attributes:
#   vlbench-staging  gs://bucket/_staging/<name>/   (bundle.tgz + env live here)
#   vlbench-results  gs://bucket/<name>/            (results + run.log go here)
#   vlbench-config   configs/financebench_threeway.yaml
set -uxo pipefail
LOG=/var/log/vlbench-run.log
exec > >(tee -a "$LOG") 2>&1

meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
STAGING_URI=$(meta vlbench-staging)
RESULTS_URI=$(meta vlbench-results)
CONFIG=$(meta vlbench-config)
LIMIT=$(meta vlbench-limit || echo 0)   # 0 = use the config's own limit

# Always push the log (for post-mortem) and stop the VM when we exit, however we
# exit. Deletion is handled by the instance's --max-run-duration backstop.
cleanup() {
  rc=$?
  echo "EXIT rc=$rc $(date -u)"
  gcloud storage cp "$LOG" "$RESULTS_URI/run.log" || true
  poweroff
}
trap cleanup EXIT

echo "START $(date -u)  staging=$STAGING_URI  results=$RESULTS_URI  config=$CONFIG"

# ── install Docker + gcloud (same approach as startup-script.sh) ──
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
if ! command -v gcloud >/dev/null 2>&1; then
  snap install google-cloud-cli --classic
fi

set -e
mkdir -p /opt/vlbench && cd /opt/vlbench
gcloud storage cp "$STAGING_URI/bundle.tgz" bundle.tgz
gcloud storage cp "$STAGING_URI/env" .env.staged
tar -xzf bundle.tgz
cp .env.staged .env

LIMARG=""
if [ "${LIMIT:-0}" != "0" ]; then LIMARG="--limit $LIMIT"; fi

docker compose build
docker compose run --rm --entrypoint python bench scripts/download_financebench.py
docker compose run --rm bench run --config "$CONFIG" $LIMARG --out /results

echo "RUN DONE $(date -u); uploading results to $RESULTS_URI/"
gcloud storage cp -r results/* "$RESULTS_URI/"
echo "UPLOADED $(date -u)"
# cleanup() trap uploads the log and powers off
