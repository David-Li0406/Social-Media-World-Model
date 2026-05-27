#!/usr/bin/env bash
# Wrapper for launching SFT training on Machine B with the right env.
#
# Required env (set by the workflow):
#   CONFIG     - path to a configs/baseline/*.yaml
#   RUN_NAME   - experiment name; runs/<RUN_NAME>/ collects small artifacts
#   DATA_ROOT  - dataset cache dir (defaults to $HOME/datasets/smwm)
set -euo pipefail

CONFIG="${CONFIG:?need CONFIG}"
RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"

# GPUs 0,1 (per meta-skill CLAUDE.md). DeepSpeed honours CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

OUT_DIR="runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "[gpu_run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[gpu_run] CONFIG=$CONFIG  RUN_NAME=$RUN_NAME  DATA_ROOT=$DATA_ROOT"
nvidia-smi || true

TRAIN_JSONL="$DATA_ROOT/RC_2025-12_politics_width_chain01_train.jsonl"
TEST_JSONL="$DATA_ROOT/RC_2025-12_politics_width_chain01_test.jsonl"
if [[ "$CONFIG" == *"_debug"* ]]; then
  TRAIN_JSONL="$DATA_ROOT/train_debug.jsonl"
  TEST_JSONL="$DATA_ROOT/test_debug.jsonl"
  echo "[gpu_run] debug mode -> $TRAIN_JSONL / $TEST_JSONL"
fi

uv run python train.py \
  --config "$CONFIG" \
  --output_dir "$OUT_DIR" \
  --train "$TRAIN_JSONL" \
  --test  "$TEST_JSONL" \
  --data_root "$DATA_ROOT" \
  2>&1 | tee "$OUT_DIR/train.log"
