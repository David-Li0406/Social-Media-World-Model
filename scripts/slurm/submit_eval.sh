#!/bin/bash
# Evaluate every results/*.jsonl produced by the baselines.
set -euo pipefail
WORKDIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WORKDIR"
PYTHON=${PYTHON:-python3}
mkdir -p logs

sbatch \
  --job-name=smwm_eval \
  --account=grp_huanliu \
  --partition=public --qos=public \
  --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=8G --time=00:30:00 \
  --output="logs/smwm_eval_%j.out" \
  --error="logs/smwm_eval_%j.err" \
  --wrap="$PYTHON -m smwm.cli.run_eval results/*.jsonl"
