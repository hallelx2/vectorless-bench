#!/usr/bin/env bash
# Spend guard, billing side (HAL-3033): a budget on the billing account that
# pays for the benchmark, filtered to the benchmark's project, alerting at
# 50%, 80% and 100%. The billing account also pays for MBPrep, so this is what
# tells you the benchmark, not MBPrep, is spending. Alerts only: the hard stop
# is the harness's --max-spend, which reads the usage each call reports.
#
#   AMOUNT=1500 deploy/gcp/budget.sh            # create or update
#   deploy/gcp/budget.sh show
set -euo pipefail
PROJECT=${PROJECT:-vectorless-prod}
AMOUNT=${AMOUNT:-1500}
NAME="vectorless benchmark (${PROJECT})"
ACCOUNT=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' | sed 's#billingAccounts/##')
NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud services enable billingbudgets.googleapis.com --project "$PROJECT" >/dev/null

existing=$(gcloud billing budgets list --billing-project="$PROJECT" --billing-account="$ACCOUNT" --format='value(name,displayName)' | awk -F'\t' -v n="$NAME" '$2==n{print $1}')
if [ "${1:-}" = show ]; then
  [ -n "$existing" ] && gcloud billing budgets describe --billing-project="$PROJECT" "$existing" --billing-account="$ACCOUNT" || echo "no budget named '$NAME'"
  exit 0
fi
args=(--billing-account="$ACCOUNT" --display-name="$NAME" --budget-amount="${AMOUNT}USD"
      --filter-projects="projects/$NUM" --calendar-period=month
      --threshold-rule=percent=0.5 --threshold-rule=percent=0.8 --threshold-rule=percent=1.0)
if [ -n "$existing" ]; then
  gcloud billing budgets update --billing-project="$PROJECT" "$existing" --billing-account="$ACCOUNT" --budget-amount="${AMOUNT}USD" >/dev/null
  echo "updated budget '$NAME' to \$$AMOUNT/month"
else
  gcloud billing budgets create --billing-project="$PROJECT" "${args[@]}" >/dev/null
  echo "created budget '$NAME': \$$AMOUNT/month on project $PROJECT, alerts at 50/80/100%"
fi
