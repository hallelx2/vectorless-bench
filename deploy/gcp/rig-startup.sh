#!/usr/bin/env bash
# Boot script for the rig VMs (deploy/gcp/rig.sh). The VM's role comes from
# instance metadata: "engine" builds and runs the engine with its Postgres and
# the host probe; "bench" installs the harness, a pgvector Postgres and the
# probe. Idempotent: a reboot re-runs it and leaves a ready VM alone.
set -euxo pipefail
[ -f /var/run/vlb-ready ] && exit 0
md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
ROLE=$(md role); ENGINE_REF=$(md engine-ref); BENCH_REF=$(md bench-ref)
MODEL=$(md model); THINKING_BUDGET=$(md thinking-budget)
PROJECT=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git jq docker.io python3 python3-venv stress-ng sysstat
systemctl enable --now docker
curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

secret() { gcloud secrets versions access latest --secret="$1" --project="$PROJECT"; }

# The probe agent ships with the harness; both VMs run it from the bench repo.
git clone -q https://github.com/hallelx2/vectorless-bench.git /opt/vectorless-bench
git -C /opt/vectorless-bench checkout -q "$BENCH_REF"

if [ "$ROLE" = engine ]; then
  GO=1.25.1
  curl -fsSL https://go.dev/dl/go$GO.linux-amd64.tar.gz | tar -C /usr/local -xz
  export PATH=/usr/local/go/bin:$PATH HOME=/root GOPATH=/root/go
  git clone -q https://github.com/hallelx2/vectorless-engine.git /opt/vectorless-engine
  cd /opt/vectorless-engine && git checkout -q "$ENGINE_REF"
  SHA=$(git rev-parse --short=12 HEAD)
  go build -trimpath -ldflags "-X main.version=${ENGINE_REF}-${SHA}" -o /usr/local/bin/vls-server ./cmd/server

  docker run -d --restart unless-stopped --name vlb-pg -e POSTGRES_USER=vectorless -e POSTGRES_PASSWORD=vectorless \
    -e POSTGRES_DB=engine -p 127.0.0.1:5432:5432 postgres:16
  mkdir -p /var/lib/vectorless/documents
  # The selfhost config, with the document root moved and no retrieval cache
  # (a cached answer would time the cache, not the engine).
  sed "s|/data/documents|/var/lib/vectorless/documents|" selfhost/engine.yaml > /etc/vectorless-engine.yaml
  umask 077
  cat > /etc/vectorless-engine.env <<EOF
VLS_DATABASE_URL=postgres://vectorless:vectorless@127.0.0.1:5432/engine?sslmode=disable
VLS_AUTH_API_KEY=$(secret vlbench-engine-api-key)
VLS_TYPESAFE_API_KEY=$(secret vlbench-typesafe-api-key)
VLS_LLM_DRIVER=vertex
VLS_LLM_VERTEX_PROJECT=$PROJECT
VLS_LLM_VERTEX_LOCATION=global
VLS_LLM_VERTEX_MODEL=$MODEL
VLS_LLM_VERTEX_THINKING_BUDGET=$THINKING_BUDGET
EOF
  cat > /etc/systemd/system/vls-server.service <<'EOF'
[Unit]
Description=Vectorless engine (benchmark rig)
After=docker.service network-online.target
[Service]
EnvironmentFile=/etc/vectorless-engine.env
ExecStart=/usr/local/bin/vls-server --config /etc/vectorless-engine.yaml --role server
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
  until docker exec vlb-pg pg_isready -U vectorless >/dev/null 2>&1; do sleep 1; done
  systemctl daemon-reload && systemctl enable --now vls-server
  until curl -sf localhost:8080/v1/health >/dev/null; do sleep 1; done
else
  docker run -d --restart unless-stopped --name vlb-pgvector -e POSTGRES_USER=vlbench -e POSTGRES_PASSWORD=vlbench \
    -e POSTGRES_DB=vlbench -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16
  cd /opt/vectorless-bench && HOME=/root uv sync -q --extra all
  umask 077
  cat >> /etc/vlbench.env <<EOF
GOOGLE_CLOUD_PROJECT=$PROJECT
GOOGLE_CLOUD_LOCATION=global
VLBENCH_THINKING_BUDGET=$THINKING_BUDGET
VLBENCH_PG_DSN=postgresql://vlbench:vlbench@127.0.0.1:5433/vlbench
VECTORLESS_API_KEY=$(secret vlbench-engine-api-key)
EOF
  mkdir -p /results
fi

cat > /etc/systemd/system/vlb-probe.service <<'EOF'
[Unit]
Description=vectorless-bench host probe (read-only CPU and process counters, VPC only)
[Service]
ExecStart=/usr/bin/python3 /opt/vectorless-bench/src/vectorless_bench/hostprobe.py serve --port 9101
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now vlb-probe
touch /var/run/vlb-ready
