#!/bin/bash

PYTHON=/home/akutteti/.conda/envs/lf/bin/python3
WORKDIR=/scratch/akutteti/reddit_dump_analysis
SCRIPT="$WORKDIR/metrics.py"
HF_CACHE=/scratch/akutteti/.cache/huggingface
PREAMBLE="cd $WORKDIR && module load mamba/latest && source activate lf && export HF_HOME=$HF_CACHE"

mkdir -p "$WORKDIR/logs"

sbatch \
  --job-name=metrics \
  --account=grp_huanliu \
  --partition=public \
  --qos=public \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=1:00:00 \
  --gres=gpu:a100:1 \
  --output="$WORKDIR/logs/metrics_%j.out" \
  --error="$WORKDIR/logs/metrics_%j.err" \
  --wrap="$PREAMBLE && $PYTHON $SCRIPT"
