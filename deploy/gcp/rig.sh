#!/usr/bin/env bash
# The speed rig on GCP (HAL-3032): the engine and its Postgres on one VM, the
# harness and every rival system on another, in one zone, on dedicated-core
# machines. A run happens on the bench VM; results go to GCS; `down` deletes
# both VMs.
#
#   deploy/gcp/rig.sh up                     # create vlb-engine + vlb-bench, wait until ready
#   deploy/gcp/rig.sh run configs/x.yaml     # run one config on the bench VM, upload results
#   deploy/gcp/rig.sh status                 # what is running, and the engine's version
#   deploy/gcp/rig.sh down                   # delete both VMs (results stay in GCS)
#
# Parallel accuracy workers (HAL-3143). A worker runs one dataset's accuracy
# pass: quality, cost and graded answers, with no timing recorded as speed.
# It shares the engine VM and the Vertex project; it has its own harness,
# pgvector and rival indexes. Timed numbers still come only from the rig.
#
#   deploy/gcp/rig.sh data-push              # on the bench VM: datasets + embedding cache -> GCS
#   deploy/gcp/rig.sh worker-up NAME         # create a worker (WORKER_MACHINE, default e2-standard-2)
#   deploy/gcp/rig.sh worker-run NAME DATASET [CONFIG]   # pull that dataset, run its accuracy pass, upload
#   deploy/gcp/rig.sh worker-down NAME
#
# Knobs (env): PROJECT ZONE MACHINE ENGINE_REF BENCH_REF BUCKET MODEL THINKING_BUDGET
# Prereqs: gcloud logged in; the vlbench-runner service account (Vertex user,
# bucket writer, secret accessor); secrets vlbench-typesafe-api-key (the Jev
# key) and vlbench-engine-api-key (created by `up` if missing).
set -euo pipefail

PROJECT=${PROJECT:-vectorless-prod}
ZONE=${ZONE:-us-central1-a}
# Dedicated cores only: never e2 or shared-core types, whose CPU steal is the
# competition the rig exists to rule out.
MACHINE=${MACHINE:-n2-standard-8}
ENGINE_REF=${ENGINE_REF:-main}
BENCH_REF=${BENCH_REF:-main}
BUCKET=${BUCKET:-gs://vectorless-prod-bench}
MODEL=${MODEL:-gemini-2.5-flash-lite}
THINKING_BUDGET=${THINKING_BUDGET:-0}
SA=${SA:-vlbench-runner@${PROJECT}.iam.gserviceaccount.com}
ENGINE_VM=${ENGINE_VM:-vlb-engine}
BENCH_VM=${BENCH_VM:-vlb-bench}

HERE="$(cd "$(dirname "$0")" && pwd)"
g() { gcloud --project="$PROJECT" --quiet "$@"; }
on() { local vm=$1; shift; g compute ssh "$vm" --zone="$ZONE" --command="$*"; }

WORKER_MACHINE=${WORKER_MACHINE:-e2-standard-2}

rig_machine_ok() {
  # Timed VMs only: a worker's numbers are never read as speed.
  case $MACHINE in
    e2-*|*-micro|*-small|*-medium)
      echo "refusing $MACHINE for the rig: shared-core / E2 machines show CPU steal; use n2d, n2 or c3" >&2; exit 2 ;;
  esac
}

ensure_secrets() {
  g services enable secretmanager.googleapis.com >/dev/null
  if ! g secrets describe vlbench-engine-api-key >/dev/null 2>&1; then
    head -c 32 /dev/urandom | base64 | tr -d '/+=\n' | g secrets create vlbench-engine-api-key --data-file=- >/dev/null
    echo "created secret vlbench-engine-api-key"
  fi
  if ! g secrets describe vlbench-typesafe-api-key >/dev/null 2>&1; then
    echo "missing secret vlbench-typesafe-api-key (the Jev key). Create it with:" >&2
    echo "  printf %s \"\$TYPESAFE_API_KEY\" | gcloud secrets create vlbench-typesafe-api-key --project $PROJECT --data-file=-" >&2
    exit 1
  fi
  for s in vlbench-engine-api-key vlbench-typesafe-api-key; do
    g secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null
  done
}

create_vm() {
  local name=$1 role=$2 machine=${3:-$MACHINE} disk=${4:-100GB} disk_type=${5:-pd-balanced}
  g compute instances describe "$name" --zone="$ZONE" >/dev/null 2>&1 && { echo "$name exists"; return; }
  g compute instances create "$name" --zone="$ZONE" --machine-type="$machine" \
    --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
    --boot-disk-size="$disk" --boot-disk-type="$disk_type" \
    --service-account="$SA" --scopes=cloud-platform \
    --labels="vlbench=1,role=$role" \
    --metadata="engine-ref=$ENGINE_REF,bench-ref=$BENCH_REF,model=$MODEL,thinking-budget=$THINKING_BUDGET,role=$role" \
    --metadata-from-file=startup-script="$HERE/rig-startup.sh" >/dev/null
  echo "created $name ($machine, $ZONE)"
}

wait_ready() {
  local name=$1
  for _ in $(seq 1 80); do
    if on "$name" "test -f /var/run/vlb-ready" >/dev/null 2>&1; then echo "$name ready"; return; fi
    sleep 15
  done
  echo "$name did not become ready; see: gcloud compute instances get-serial-port-output $name --zone $ZONE" >&2
  exit 1
}

engine_ip() { g compute instances describe "$ENGINE_VM" --zone="$ZONE" --format='value(networkInterfaces[0].networkIP)'; }

cmd=${1:-}
case $cmd in
  up)
    rig_machine_ok
    ensure_secrets
    create_vm "$ENGINE_VM" engine
    create_vm "$BENCH_VM" bench
    wait_ready "$ENGINE_VM"
    IP=$(engine_ip)
    on "$BENCH_VM" "sudo bash -c 'touch /etc/vlbench.env && chmod 600 /etc/vlbench.env && sed -i \"/^\\(VECTORLESS_BASE_URL\\|VLBENCH_ENGINE_PROBE\\)=/d\" /etc/vlbench.env && echo VECTORLESS_BASE_URL=http://$IP:8080 >> /etc/vlbench.env && echo VLBENCH_ENGINE_PROBE=http://$IP:9101 >> /etc/vlbench.env'"
    wait_ready "$BENCH_VM"
    on "$BENCH_VM" "curl -sf http://$IP:8080/v1/version && echo && curl -sf http://$IP:9101/health"
    ;;
  run)
    CONFIG=${2:?usage: rig.sh run <config.yaml> [extra vlbench args]}
    shift 2
    RUN_NAME="rig-$(date +%Y%m%d-%H%M%S)"
    on "$BENCH_VM" "set -euo pipefail; set -a; source /etc/vlbench.env; set +a; cd /opt/vectorless-bench && \
      sudo -E HOME=/root /usr/local/bin/uv run vlbench run --config '$CONFIG' --out /results/$RUN_NAME \
        --engine-probe \"\$VLBENCH_ENGINE_PROBE\" $* && \
      sudo gcloud storage cp -r /results/$RUN_NAME '$BUCKET/runs/'"
    echo "results: $BUCKET/runs/$RUN_NAME/"
    ;;
  status)
    g compute instances list --filter="labels.vlbench=1" --format="table(name,zone.basename(),machineType.basename(),status,networkInterfaces[0].networkIP)"
    g compute instances describe "$ENGINE_VM" --zone="$ZONE" >/dev/null 2>&1 && on "$BENCH_VM" "set -a; source /etc/vlbench.env; set +a; curl -s \$VECTORLESS_BASE_URL/v1/version; echo" || true
    ;;
  down)
    for vm in "$BENCH_VM" "$ENGINE_VM"; do
      g compute instances delete "$vm" --zone="$ZONE" >/dev/null 2>&1 && echo "deleted $vm" || echo "$vm not present"
    done
    left=$(g compute instances list --filter="labels.vlbench=1 AND labels.role!=dev" --format='value(name)')
    [ -z "$left" ] && echo "no rig VMs left running" || echo "still running: $left"
    ;;
  data-push)
    # Run on the bench VM, from the repo root: every fetched dataset and the
    # per-document embedding cache, so a worker pulls instead of re-fetching.
    gcloud storage rsync -r --quiet data "$BUCKET/data"
    echo "pushed data/ -> $BUCKET/data"
    ;;
  worker-up)
    NAME=${2:?usage: rig.sh worker-up NAME}
    ensure_secrets
    # Standard disk: a worker holds one dataset (under 1 GB) and the
    # environment; it does not need SSD, and pd-standard does not count
    # against the SSD quota.
    create_vm "$NAME" worker "$WORKER_MACHINE" 40GB pd-standard
    IP=$(engine_ip)
    wait_ready "$NAME"
    on "$NAME" "sudo bash -c 'touch /etc/vlbench.env && chmod 600 /etc/vlbench.env && sed -i \"/^VECTORLESS_BASE_URL=/d\" /etc/vlbench.env && echo VECTORLESS_BASE_URL=http://$IP:8080 >> /etc/vlbench.env'"
    on "$NAME" "curl -sf http://$IP:8080/v1/version && echo"
    ;;
  worker-run)
    NAME=${2:?usage: rig.sh worker-run NAME DATASET [CONFIG]}
    DATASET=${3:?usage: rig.sh worker-run NAME DATASET [CONFIG]}
    CONFIG=${4:-configs/pilot/$DATASET.yaml}
    RUN_NAME="acc-$DATASET-$(date +%Y%m%d-%H%M%S)"
    # The run script goes over ssh's stdin (no nested quoting), then runs
    # detached so it outlives this session; its log and results go to GCS.
    g compute ssh "$NAME" --zone="$ZONE" --command="sudo tee /root/$RUN_NAME.sh >/dev/null" <<EOS
set -uo pipefail
set -a; . /etc/vlbench.env; set +a
cd /opt/vectorless-bench
git fetch -q origin && git checkout -q $BENCH_REF && git pull -q --ff-only || true
mkdir -p data/$DATASET data/cache
gcloud storage rsync -r --quiet $BUCKET/data/$DATASET data/$DATASET
gcloud storage rsync -r --quiet $BUCKET/data/cache data/cache
HOME=/root /usr/local/bin/uv sync -q --extra all
HOME=/root /usr/local/bin/uv run --no-sync vlbench run --config $CONFIG --phase accuracy --out /results/$RUN_NAME
rc=\$?
gcloud storage cp -r /results/$RUN_NAME $BUCKET/runs/
echo "exit=\$rc"
gcloud storage cp /root/$RUN_NAME.log $BUCKET/runs/$RUN_NAME.log
EOS
    on "$NAME" "sudo bash -c 'setsid nohup bash /root/$RUN_NAME.sh > /root/$RUN_NAME.log 2>&1 < /dev/null &'; echo started"
    echo "started $RUN_NAME on $NAME; results -> $BUCKET/runs/$RUN_NAME/"
    ;;
  worker-down)
    NAME=${2:?usage: rig.sh worker-down NAME}
    g compute instances delete "$NAME" --zone="$ZONE" >/dev/null 2>&1 && echo "deleted $NAME" || echo "$NAME not present"
    ;;
  *)
    sed -n 2,26p "$0"; exit 2 ;;
esac
