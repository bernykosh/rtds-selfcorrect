#!/usr/bin/env bash
# Step 7, part 2: run the official SWE-bench harness on each patch set.
#
#   ./run_eval.sh               # all six sets, all instances in the predictions files
#   ./run_eval.sh a1 S          # only these sets
#   INSTANCE_IDS="id1 id2" ./run_eval.sh   # restrict to some instances (batching)
#
# The harness skips instances that already have a report for the same run_id,
# so you can rerun this script after an interruption.
set -euo pipefail
cd "$(dirname "$0")"

RUN_ID="${RUN_ID:-study2026}"
WORKERS="${WORKERS:-2}"
SETS=("$@")
if [ ${#SETS[@]} -eq 0 ]; then SETS=(a1 a2 a3 S T ST); fi

for s in "${SETS[@]}"; do
  echo "=== harness: $s ==="
  args=(-p "predictions/$s.jsonl" -id "$RUN_ID" --max_workers "$WORKERS" --report_dir harness_reports)
  if [ -n "${INSTANCE_IDS:-}" ]; then args+=(-i $INSTANCE_IDS); fi
  python -m swebench.harness.run_evaluation "${args[@]}"
done
echo "reports in harness_reports/, per-instance logs in logs/run_evaluation/$RUN_ID/<set>/<instance_id>/report.json"
