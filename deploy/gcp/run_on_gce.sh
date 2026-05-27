#!/usr/bin/env bash
# Provision a GCE VM, run the benchmark bundle on it, bring results back, and
# delete the VM. Orchestrated from your machine with the gcloud CLI.
#
# Secrets are read from Secret Manager HERE (with your gcloud identity) into a
# .env that is shipped to the VM — so no keys are typed by hand and the VM needs
# no extra IAM. The VM itself only needs Docker (installed by startup-script.sh).
#
# Prereqs:
#   gcloud auth login   (and access to the project's `server-config` secret)
#
# Usage:
#   PROJECT=project-03250746-ec5b-4198-990 ./deploy/gcp/run_on_gce.sh
#
# Knobs: ZONE MACHINE NAME CONFIG SECRET BASE_URL DISK DOWNLOAD_DOCS DELETE_AFTER BUCKET
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT=<gcp project id>}
ZONE=${ZONE:-us-central1-a}
MACHINE=${MACHINE:-e2-standard-4}
NAME=${NAME:-vlbench-$(date +%Y%m%d-%H%M%S)}
CONFIG=${CONFIG:-configs/financebench_gemini.yaml}
SECRET=${SECRET:-server-config}
BASE_URL=${BASE_URL:-https://vectorless-server-2rzh3kctga-uc.a.run.app}
DISK=${DISK:-50GB}
DOWNLOAD_DOCS=${DOWNLOAD_DOCS:-1}
DELETE_AFTER=${DELETE_AFTER:-1}
BUCKET=${BUCKET:-}          # optional: also copy results to gs://...
LIMIT=${LIMIT:-}           # optional: override the config's question limit
DOCS=${DOCS:-}             # optional: cap how many FinanceBench PDFs to download
DOCS_ARG=""; [ -n "$DOCS" ] && DOCS_ARG="--limit $DOCS"
# network tag required for IAP SSH ingress on this project (firewall targets it)
TAGS=${TAGS:-dokploy}
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
g() { gcloud --project="$PROJECT" "$@"; }
# SSH is IAP-only on this network, so tunnel through IAP for ssh + scp.
IAP="--tunnel-through-iap"
# echo y: auto-accept PuTTY plink's host-key cache prompt on Windows (harmless
# leftover stdin for the remote command, which doesn't read it).
ssh_vm() { echo y | g compute ssh "$NAME" --zone="$ZONE" $IAP --command="$1"; }

echo ">> staging the vectorless SDK into the build context"
"$REPO_ROOT/deploy/vendor_sdk.sh"

echo ">> reading secrets from Secret Manager ($SECRET) into .env"
python "$REPO_ROOT/deploy/load_secrets.py" \
  --project "$PROJECT" --secret "$SECRET" --base-url "$BASE_URL" \
  --out "$REPO_ROOT/.env"

echo ">> creating VM $NAME ($MACHINE, $ZONE)"
g compute instances create "$NAME" \
  --zone="$ZONE" --machine-type="$MACHINE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="$DISK" --tags="$TAGS" \
  --metadata-from-file=startup-script="$HERE/startup-script.sh"

cleanup() {
  if [ "$DELETE_AFTER" = "1" ]; then
    echo ">> deleting VM $NAME"
    g compute instances delete "$NAME" --zone="$ZONE" --quiet || true
  fi
}
trap cleanup EXIT   # delete the VM even if the run fails

# pre-generate a passphrase-less SSH key so `gcloud compute ssh` is non-interactive
KEY="$HOME/.ssh/google_compute_engine"
[ -f "$KEY" ] || { mkdir -p "$HOME/.ssh"; ssh-keygen -t rsa -b 2048 -f "$KEY" -N "" -q; }

echo ">> waiting for Docker on the VM (and caching the host key)"
# On Windows gcloud uses PuTTY plink/pscp, which prompt to cache an unknown host
# key. Pipe 'y' so the first connect caches it (shared by later plink + pscp);
# subsequent calls then run non-interactively.
for _ in $(seq 1 40); do
  if echo y | g compute ssh "$NAME" --zone="$ZONE" $IAP \
       --command="sudo docker ps >/dev/null 2>&1 && test -f /var/run/vlbench-ready" \
       >/dev/null 2>&1; then
    echo "   ready"; break
  fi
  sleep 15
done

echo ">> shipping the bundle (incl. vendored SDK) + .env"
tar --exclude='.git' --exclude='runs' --exclude='data/financebench/docs' \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='vendor/*/.venv' \
    -czf "/tmp/vlbench-$NAME.tgz" -C "$REPO_ROOT" .
g compute scp $IAP "/tmp/vlbench-$NAME.tgz" "$NAME":~/vlbench.tgz --zone="$ZONE"
g compute scp $IAP "$REPO_ROOT/.env" "$NAME":~/vlbench.env --zone="$ZONE"

echo ">> building + running on the VM (the long part)"
ssh_vm "
  set -euxo pipefail
  rm -rf ~/vlbench && mkdir -p ~/vlbench && tar -xzf ~/vlbench.tgz -C ~/vlbench
  cd ~/vlbench && cp ~/vlbench.env .env
  sudo docker compose build
  if [ '$DOWNLOAD_DOCS' = '1' ]; then
    sudo docker compose run --rm --entrypoint python bench scripts/download_financebench.py $DOCS_ARG
  fi
  sudo docker compose run --rm bench run --config '$CONFIG' $LIMIT_ARG --out /results
"

echo ">> bringing results back to runs/vm-$NAME"
mkdir -p "$REPO_ROOT/runs"
g compute scp --recurse $IAP "$NAME":~/vlbench/results "$REPO_ROOT/runs/vm-$NAME" --zone="$ZONE"

if [ -n "$BUCKET" ]; then
  echo ">> also uploading results to $BUCKET/$NAME/"
  gcloud storage cp -r "$REPO_ROOT/runs/vm-$NAME/*" "$BUCKET/$NAME/" || true
fi

echo ">> done. open the report:"
find "$REPO_ROOT/runs/vm-$NAME" -name report.html | head -1
# VM deleted by the EXIT trap
