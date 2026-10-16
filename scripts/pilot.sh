#!/usr/bin/env bash
# The pilot (HAL-3048): every dataset in turn, then the scoreboard.
# Run on vlb-bench from the repo root, detached:
#   setsid nohup scripts/pilot.sh > ~/pilot.log 2>&1 < /dev/null &
set -uo pipefail
set -a; . /etc/vlbench.env; set +a
ORDER=${PILOT_ORDER:-"financebench manuals qasper cuad legalbench_rag mmlongbench_doc multihop_rag"}
for d in $ORDER; do
  echo "=== $(date -u +%FT%TZ) $d"
  uv run --no-sync vlbench run --config configs/pilot/$d.yaml || echo "!!! $d exited $?"
  uv run --no-sync vlbench scoreboard runs/pilot --out runs/pilot/scoreboard || true
done
echo "=== $(date -u +%FT%TZ) pilot done"
