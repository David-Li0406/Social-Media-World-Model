#!/usr/bin/env bash
# LLM sweep for the reply-forecasting task + the two one-for-all architectures.
#
# Four lanes, one per GPU, each sequential inside itself:
#   GPU 4  Qwen3-4B            zero-shot -> LoRA SFT
#   GPU 5  Qwen3-8B            zero-shot -> LoRA SFT
#   GPU 6  Llama-3.1-8B-Instr  zero-shot -> LoRA SFT
#   GPU 7  one-for-all: all-generative, then hybrid (heads + generative)
#
# Every run scores the identical eval file, so single-task, multi-task and
# architecture comparisons are all on the same records.
set -uo pipefail
RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"; mkdir -p "$OUT_DIR"
TRAIN="${FC_TRAIN:-$DATA_ROOT/forecast_train.jsonl.gz}"
EVAL="${FC_EVAL:-$DATA_ROOT/forecast_eval.jsonl.gz}"
EVAL_LIMIT="${EVAL_LIMIT:-2000}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[sweep] train=$TRAIN eval=$EVAL"

run() { # gpu tag extra-args...
  local gpu="$1" tag="$2"; shift 2
  echo "[$tag] GPU $gpu start $(date +%T)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train_multitask.py \
    --eval "$EVAL" --eval-limit "$EVAL_LIMIT" \
    --out "$OUT_DIR/${tag}.jsonl" "$@" > "$OUT_DIR/${tag}.log" 2>&1
  echo "[$tag] GPU $gpu exit=$? $(date +%T)"
}

lane() { # gpu model short
  local gpu="$1" model="$2" short="$3"
  ( run "$gpu" "${short}_zeroshot" --model "$model" --tasks summary --zero-shot
    run "$gpu" "${short}_sft"      --model "$model" --tasks summary \
        --train "$TRAIN" --epochs 1 ) &
}

lane 4 "Qwen/Qwen3-4B"                  qwen3_4b
lane 5 "Qwen/Qwen3-8B"                  qwen3_8b
lane 6 "meta-llama/Llama-3.1-8B-Instruct" llama31_8b

# GPU 7: the two one-for-all settings, same backbone, same data, same eval
( run 7 "allgen_qwen3_4b"    --model Qwen/Qwen3-4B --arch generative \
      --tasks score,controversiality,width,summary --train "$TRAIN" --epochs 1
  run 7 "allhybrid_qwen3_4b" --model Qwen/Qwen3-4B --arch hybrid \
      --tasks score,controversiality,width,summary --train "$TRAIN" --epochs 1 ) &

wait
echo "[sweep] all lanes finished $(date +%T)"
ok=0
for f in "$OUT_DIR"/*.jsonl; do
  [ -s "$f" ] && { echo "[sweep] OK   $(basename "$f") ($(wc -l < "$f"))"; ok=$((ok+1)); } \
               || echo "[sweep] FAIL $(basename "$f")"
done
for t in qwen3_4b qwen3_8b llama31_8b allgen_qwen3_4b allhybrid_qwen3_4b; do
  for f in "$OUT_DIR/${t}"*.log; do
    [ -f "$f" ] && [ ! -s "${f%.log}.jsonl" ] && { echo "--- $(basename "$f") tail ---"; tail -4 "$f"; }
  done
done
[ "$ok" -eq 0 ] && { echo "[sweep] FATAL: no lane produced output" >&2; exit 1; }
exit 0
