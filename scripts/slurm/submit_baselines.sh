#!/bin/bash
# Submit one inference job per baseline YAML.
# Usage: scripts/slurm/submit_baselines.sh [config1.yaml config2.yaml ...]
# If no configs given, submits every YAML under configs/baseline/.

set -euo pipefail
WORKDIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WORKDIR"

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    mapfile -t CONFIGS < <(ls configs/baseline/*.yaml)
fi

PYTHON=${PYTHON:-python3}
TRAIN=${TRAIN:-train_test/RC_2025-12_politics_width_chain01_train.jsonl}
TEST=${TEST:-train_test/RC_2025-12_politics_width_chain01_test.jsonl}

mkdir -p logs results

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    needs_gpu=0
    grep -Eq '^(baseline:\s*(encoder|llm))' "$cfg" && needs_gpu=1

    sbatch_args=(
        --job-name="bl_${name}"
        --account=grp_huanliu
        --partition=public --qos=public
        --nodes=1 --ntasks=1 --cpus-per-task=4
        --mem=32G --time=4:00:00
        --output="logs/bl_${name}_%j.out"
        --error="logs/bl_${name}_%j.err"
    )
    if [ "$needs_gpu" -eq 1 ]; then
        sbatch_args+=(--gres=gpu:a100:1)
    fi

    cmd="$PYTHON -m smwm.cli.run_inference --config $cfg --train $TRAIN --test $TEST"
    sbatch "${sbatch_args[@]}" --wrap="$cmd"
done
