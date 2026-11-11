#!/usr/bin/env bash
# Chain-of-thought inference across 10 models, all four tasks, on GPUs 4-7.
#
# One eval file (forecast_eval) carries all four targets, so every model is
# scored on identical records for score/controversiality/width AND summary.
# Note the population: these records all have width>=1, so numeric figures here
# are NOT comparable to the main benchmark table, which samples the full corpus.
#
# Lanes are balanced by model size against measured H20 throughput, so no lane
# runs far past the longest single model (Mistral-Small-24B).
set -uo pipefail
RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"; mkdir -p "$OUT_DIR"
EVAL="${FC_EVAL:-$DATA_ROOT/forecast_eval.jsonl.gz}"
EVAL_LIMIT="${EVAL_LIMIT:-2000}"
GEN_BS="${GEN_BATCH:-16}"
TASKS="score,controversiality,width,summary"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
echo "[cot] eval=$EVAL limit=$EVAL_LIMIT"

run() { # gpu tag model
  local gpu="$1" tag="$2" model="$3"
  echo "[$tag] GPU $gpu start $(date +%T)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train_multitask.py \
    --model "$model" --tasks "$TASKS" --cot --zero-shot \
    --eval "$EVAL" --eval-limit "$EVAL_LIMIT" \
    --gen-batch "$GEN_BS" --max-new-tokens 320 \
    --out "$OUT_DIR/${tag}.jsonl" > "$OUT_DIR/${tag}.log" 2>&1
  echo "[$tag] GPU $gpu exit=$? $(date +%T)"
}

# GPU 4 - the single largest model
( run 4 cot_mistral_small_24b "mistralai/Mistral-Small-24B-Instruct-2501" ) &
# GPU 5 - two ~9B
( run 5 cot_gemma2_9b   "google/gemma-2-9b-it"
  run 5 cot_llama31_8b  "meta-llama/Llama-3.1-8B-Instruct" ) &
# GPU 6 - two ~7-8B
( run 6 cot_ministral_8b "mistralai/Ministral-8B-Instruct-2410"
  run 6 cot_mistral_7b   "mistralai/Mistral-7B-Instruct-v0.3" ) &
# GPU 7 - the five small ones
( run 7 cot_qwen35_4b   "Qwen/Qwen3.5-4B"
  run 7 cot_llama32_3b  "meta-llama/Llama-3.2-3B-Instruct"
  run 7 cot_qwen35_2b   "Qwen/Qwen3.5-2B"
  run 7 cot_llama32_1b  "meta-llama/Llama-3.2-1B-Instruct"
  run 7 cot_qwen35_08b  "Qwen/Qwen3.5-0.8B" ) &

wait
echo "[cot] all lanes finished $(date +%T)"
ok=0; fail=0
for t in cot_mistral_small_24b cot_gemma2_9b cot_llama31_8b cot_ministral_8b cot_mistral_7b \
         cot_qwen35_4b cot_llama32_3b cot_qwen35_2b cot_llama32_1b cot_qwen35_08b; do
  if [ -s "$OUT_DIR/$t.jsonl" ]; then
    echo "[cot] OK   $t ($(wc -l < "$OUT_DIR/$t.jsonl"))"; ok=$((ok+1))
  else
    echo "[cot] FAIL $t"; fail=$((fail+1))
    tail -4 "$OUT_DIR/$t.log" 2>/dev/null | sed 's/^/    /'
  fi
done
echo "[cot] $ok ok, $fail failed"
[ "$ok" -eq 0 ] && { echo "[cot] FATAL: no model produced output" >&2; exit 1; }
exit 0
