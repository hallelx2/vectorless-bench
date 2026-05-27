#!/usr/bin/env bash
# Runs once on VM boot (passed as the instance startup-script). Installs Docker so
# the benchmark can run in containers. Secrets and result-shipping are handled
# from the orchestrator host (run_on_gce.sh), so the VM needs nothing else.
set -euxo pipefail

curl -fsSL https://get.docker.com | sh

# marker the orchestrator polls for
touch /var/run/vlbench-ready
