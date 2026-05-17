#!/bin/bash

PYTHON=/home/akutteti/.conda/envs/lf/bin/python3
WORKDIR=/scratch/akutteti/reddit_dump_analysis
SCRIPT="$WORKDIR/test_inference.py"
HF_CACHE=/scratch/akutteti/.cache/huggingface
PREAMBLE="cd $WORKDIR && module load mamba/latest && source activate lf && export HF_HOME=$HF_CACHE"

mkdir -p "$WORKDIR/logs"

sbatch \
  --job-name=tinf_qwen3_4b \
  --account=grp_huanliu \
  --partition=public \
  --qos=public \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=8:00:00 \
  --gres=gpu:a100:1 \
  --output="$WORKDIR/logs/tinf_qwen3_4b_%j.out" \
  --error="$WORKDIR/logs/tinf_qwen3_4b_%j.err" \
  --wrap="$PREAMBLE && $PYTHON $SCRIPT --model Qwen/Qwen3-4B"

sbatch \
  --job-name=tinf_llama3_8b \
  --account=grp_huanliu \
  --partition=public \
  --qos=public \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=8:00:00 \
  --gres=gpu:a100:1 \
  --output="$WORKDIR/logs/tinf_llama3_8b_%j.out" \
  --error="$WORKDIR/logs/tinf_llama3_8b_%j.err" \
  --wrap="$PREAMBLE && $PYTHON $SCRIPT --model meta-llama/Llama-3.1-8B-Instruct"

sbatch \
  --job-name=tinf_qwen3_32b \
  --account=grp_huanliu \
  --partition=general \
  --qos=private \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=64G --time=8:00:00 \
  --gres=gpu:h100:1 \
  --output="$WORKDIR/logs/tinf_qwen3_32b_%j.out" \
  --error="$WORKDIR/logs/tinf_qwen3_32b_%j.err" \
  --wrap="$PREAMBLE && $PYTHON $SCRIPT --model Qwen/Qwen3-32B"

sbatch \
  --job-name=tinf_llama3_70b \
  --account=grp_huanliu \
  --partition=general \
  --qos=private \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=64G --time=8:00:00 \
  --gres=gpu:h100:1 \
  --output="$WORKDIR/logs/tinf_llama3_70b_%j.out" \
  --error="$WORKDIR/logs/tinf_llama3_70b_%j.err" \
  --wrap="$PREAMBLE && $PYTHON $SCRIPT --model meta-llama/Llama-3.3-70B-Instruct"
