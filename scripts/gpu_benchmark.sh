#!/usr/bin/env bash
# Run the GPU half of the baseline benchmark across all four GPUs at once.
#
# A self-hosted runner executes one workflow job at a time, so four dispatches
# would queue rather than parallelize. This job therefore fans out itself: one
# background process per GPU, each pinned with CUDA_VISIBLE_DEVICES, then waits.
#
# Shards are balanced against measured throughput (llm_sft trains at 2.26
# samples/s, llm_reghead at 8.4, generation costs ~0.6 s/record batched), so no
# lane runs much past ~3h and the whole fan-out fits the 4h budget.
set -uo pipefail

RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"
TRAIN="${BENCH_TRAIN:-$DATA_ROOT/bench_temporal_train.jsonl.gz}"
EVAL="${BENCH_EVAL:-$DATA_ROOT/bench_temporal_eval.jsonl.gz}"
GEN_BS="${GEN_BATCH:-16}"
SFT_TRAIN_LIMIT="${SFT_TRAIN_LIMIT:-12000}"

mkdir -p "$OUT_DIR"
echo "[bench] train=$TRAIN"
echo "[bench] eval =$EVAL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# lane <gpu> <tag> <config> <train_limit> <batch_size>
lane() {
  local gpu="$1" tag="$2" cfg="$3" tlim="$4" bs="$5"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[lane $tag] GPU $gpu start $(date +%T)"
    uv run python -m smwm.cli.run_inference \
      --config "$cfg" --train "$TRAIN" --test "$EVAL" \
      --train-limit "$tlim" --batch-size "$bs" \
      --results "$OUT_DIR/${tag}.jsonl" > "$OUT_DIR/${tag}.log" 2>&1
    echo "[lane $tag] GPU $gpu exit=$? $(date +%T)"
  ) &
}

# GPU 4 - cheap neural rows, sequential inside the lane
(
  export CUDA_VISIBLE_DEVICES=4
  for t in encoder frozen_mlp; do
    echo "[lane $t] GPU 4 start $(date +%T)"
    uv run python -m smwm.cli.run_inference \
      --config "configs/bench/${t}.yaml" --train "$TRAIN" --test "$EVAL" \
      --train-limit 50000 --batch-size 256 \
      --results "$OUT_DIR/${t}.jsonl" > "$OUT_DIR/${t}.log" 2>&1
    echo "[lane $t] GPU 4 exit=$? $(date +%T)"
  done
) &

lane 5 llm_reghead configs/bench/llm_reghead.yaml 50000 64
lane 6 llm_sft     configs/bench/llm_sft.yaml     "$SFT_TRAIN_LIMIT" "$GEN_BS"

# GPU 7 - the three prompting rows, no training, sequential inside the lane
(
  export CUDA_VISIBLE_DEVICES=7
  for t in llm_zeroshot llm_fewshot llm_cot; do
    echo "[lane $t] GPU 7 start $(date +%T)"
    uv run python -m smwm.cli.run_inference \
      --config "configs/bench/${t}.yaml" --train "$TRAIN" --test "$EVAL" \
      --train-limit 2000 --batch-size "$GEN_BS" \
      --results "$OUT_DIR/${t}.jsonl" > "$OUT_DIR/${t}.log" 2>&1
    echo "[lane $t] GPU 7 exit=$? $(date +%T)"
  done
) &

wait
echo "[bench] all lanes finished $(date +%T)"
ls -la "$OUT_DIR"/*.jsonl 2>/dev/null || echo "[bench] WARNING: no result files"
uv run python -m smwm.cli.run_eval "$OUT_DIR"/*.jsonl 2>&1 | tee "$OUT_DIR/metrics.txt" || true
