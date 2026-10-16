#!/usr/bin/env bash
# The pilot (HAL-3048): every dataset in turn, then the scoreboard.
# Run on vlb-bench from the repo root, detached:
#   setsid nohup scripts/pilot.sh > ~/pilot.log 2>&1 < /dev/null &
# A dataset that fails does not stop the others; the script exits nonzero
# and names every failure at the end.
set -uo pipefail
set -a; . /etc/vlbench.env; set +a
ORDER=${PILOT_ORDER:-"financebench manuals qasper cuad legalbench_rag mmlongbench_doc multihop_rag"}
failed=()
for d in $ORDER; do
  echo "=== $(date -u +%FT%TZ) $d"
  if ! uv run --no-sync vlbench run --config "configs/pilot/$d.yaml"; then
    echo "!!! $d: run failed"
    failed+=("$d:run")
  fi
  if ! uv run --no-sync vlbench scoreboard runs/pilot --out runs/pilot/scoreboard; then
    echo "!!! $d: scoreboard failed"
    failed+=("$d:scoreboard")
  fi
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "=== $(date -u +%FT%TZ) pilot finished with failures: ${failed[*]}"
  exit 1
fi
echo "=== $(date -u +%FT%TZ) pilot done"
