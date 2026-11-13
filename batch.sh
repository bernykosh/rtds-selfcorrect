#!/usr/bin/env bash
# Run the whole pipeline for a batch of instances, then remove their Docker
# images to keep disk use under the 40 GB Docker Desktop limit.
#
#   ./batch.sh 0 20     # instances 0..19 in sampled order
#   ./batch.sh 20 40
#
# Logs and harness reports accumulate across batches. analyze.py reads them all.
set -euo pipefail
cd "$(dirname "$0")"
START=$1; END=$2
IDS=$(python - "$START" "$END" << 'PY'
import json, sys
ids = json.load(open("data/instances.json"))["instance_ids"]
print(" ".join(ids[int(sys.argv[1]):int(sys.argv[2])]))
PY
)
echo "batch: $IDS"
python fetch_oracle_files.py --limit "$END"
python run_conditions.py first --only $IDS
python run_conditions.py S     --only $IDS
python run_conditions.py T     --only $IDS
python run_conditions.py ST    --only $IDS
python build_predictions.py --limit "$END"
INSTANCE_IDS="$IDS" ./run_eval.sh
if [ "${KEEP_IMAGES:-0}" != "1" ]; then
  python - "$IDS" << 'PY'
import json, subprocess, sys
from common import load_lite_dataset
ids = sys.argv[1].split()
for inst in load_lite_dataset(ids):
    subprocess.run(["docker", "rmi", "-f", inst["image"]], check=False)
PY
  docker image prune -f >/dev/null
fi
