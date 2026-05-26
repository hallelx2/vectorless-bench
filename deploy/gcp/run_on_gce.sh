#!/usr/bin/env bash
# Provision a GCE VM, run the benchmark bundle on it unattended, push results to
# a GCS bucket, and (optionally) delete the VM. Orchestrated from your machine
# with the gcloud CLI; the VM only needs Docker + the Cloud CLI (installed by
# startup-script.sh).
#
# Prereqs (one time):
#   - gcloud auth login && gcloud config set project <PROJECT>
#   - a GCS bucket; the VM's service account needs roles/storage.objectAdmin on it
#   - a local .env with your API keys (OPENAI_API_KEY, VECTORLESS_*, etc.)
#
# Usage:
#   PROJECT=my-proj BUCKET=gs://my-bucket ./deploy/gcp/run_on_gce.sh
#
# Knobs (env vars): ZONE MACHINE NAME CONFIG ENV_FILE DISK DOWNLOAD_DOCS DELETE_AFTER
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT=<gcp project id>}
BUCKET=${BUCKET:?set BUCKET=gs://your-bucket}
ZONE=${ZONE:-us-central1-a}
MACHINE=${MACHINE:-e2-standard-4}
NAME=${NAME:-vlbench-$(date +%Y%m%d-%H%M%S)}
CONFIG=${CONFIG:-configs/financebench.yaml}
ENV_FILE=${ENV_FILE:-.env}
DISK=${DISK:-50GB}
DOWNLOAD_DOCS=${DOWNLOAD_DOCS:-1}   # fetch FinanceBench PDFs on the VM
DELETE_AFTER=${DELETE_AFTER:-1}     # delete the VM when done (saves money)

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
g() { gcloud --project="$PROJECT" "$@"; }
ssh_vm() { g compute ssh "$NAME" --zone="$ZONE" --command="$1"; }

echo ">> creating VM $NAME ($MACHINE, $ZONE)"
g compute instances create "$NAME" \
  --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="$DISK" --scopes=cloud-platform \
  --metadata-from-file=startup-script="$HERE/startup-script.sh"

echo ">> waiting for Docker on the VM"
for _ in $(seq 1 40); do
  if ssh_vm "sudo docker ps >/dev/null 2>&1 && test -f /var/run/vlbench-ready"; then
    echo "   ready"; break
  fi
  sleep 15
done

echo ">> shipping the bundle"
tar --exclude='.git' --exclude='runs' --exclude='data/financebench/docs' \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    -czf /tmp/vlbench.tgz -C "$REPO_ROOT" .
g compute scp /tmp/vlbench.tgz "$NAME":~ --zone="$ZONE"
if [ -f "$ENV_FILE" ]; then
  g compute scp "$ENV_FILE" "$NAME":~/vlbench.env --zone="$ZONE"
else
  echo "   (no $ENV_FILE — the VM will rely on whatever keys the run needs)"
fi

echo ">> building + running on the VM (this is the long part)"
ssh_vm "
  set -euxo pipefail
  rm -rf ~/vlbench && mkdir -p ~/vlbench && tar -xzf ~/vlbench.tgz -C ~/vlbench
  cd ~/vlbench
  [ -f ~/vlbench.env ] && cp ~/vlbench.env .env || true
  sudo docker compose build
  if [ '$DOWNLOAD_DOCS' = '1' ]; then
    sudo docker compose run --rm --entrypoint python bench scripts/download_financebench.py
  fi
  sudo docker compose run --rm bench run --config '$CONFIG' --out /results
  sudo gcloud storage cp -r results/* '$BUCKET/$NAME/'
"

echo ">> results uploaded to $BUCKET/$NAME/"
echo "   view:  BUCKET=$BUCKET RUN_ID=$NAME $HERE/fetch_results.sh"

if [ "$DELETE_AFTER" = "1" ]; then
  echo ">> deleting VM $NAME"
  g compute instances delete "$NAME" --zone="$ZONE" --quiet
fi
