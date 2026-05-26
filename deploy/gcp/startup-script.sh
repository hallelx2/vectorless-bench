#!/usr/bin/env bash
# Runs once on VM boot (passed as metadata startup-script). Installs Docker and
# the Google Cloud CLI so the benchmark can run in containers and upload results
# to GCS using the VM's service-account identity (no keys baked in).
set -euxo pipefail

# Docker engine + compose plugin
curl -fsSL https://get.docker.com | sh

# Google Cloud CLI via snap (reliable on Ubuntu images); used to push results.
if ! command -v gcloud >/dev/null 2>&1; then
  snap install google-cloud-cli --classic
fi

# marker the orchestrator can poll for
touch /var/run/vlbench-ready
