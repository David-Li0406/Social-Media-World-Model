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
wait_for_gpus() { # <min_free_MiB> <max_wait_minutes>
  local need="$1" maxmin="$2" waited=0
  # Fail fast if the requested indices do not exist on this host -- the runner
  # may have moved to a machine with a different GPU count, and waiting 12h for
  # cards that are not there is the worst outcome.
  local total
  total=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${total:-0}" -lt 8 ]; then
    echo "[gpu-wait] FATAL: host has ${total:-0} GPUs; this sweep targets indices 4-7" >&2
    nvidia-smi --query-gpu=index,name,memory.total --format=csv >&2
    return 1
  fi
  echo "[gpu-wait] host has $total GPUs; targeting 4,5,6,7"
  while :; do
    local free_ok
    free_ok=$(nvidia-smi --id=4,5,6,7 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
              | awk -v n="$need" 'BEGIN{c=0} {if ($1+0 >= n) c++} END{print c}')
    [ "${free_ok:-0}" -ge 4 ] && { echo "[gpu-wait] all 4 GPUs have >= ${need}MiB free after ${waited}m"; return 0; }
    if [ "$waited" -ge "$maxmin" ]; then
      echo "[gpu-wait] FATAL: after ${maxmin}m only ${free_ok:-0}/4 GPUs have >= ${need}MiB free" >&2
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2
      return 1
    fi
    [ $((waited % 15)) -eq 0 ] && echo "[gpu-wait] ${free_ok:-0}/4 GPUs free; waited ${waited}m of ${maxmin}m"
    sleep 60; waited=$((waited+1))
  done
}

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
wait_for_gpus "${MIN_FREE_MIB:-60000}" "${GPU_WAIT_MIN:-720}" || exit 1
echo "[cot] eval=$EVAL limit=$EVAL_LIMIT"

run() { # gpu tag model [batch]
  local gpu="$1" tag="$2" model="$3" bs="${4:-$GEN_BS}"
  echo "[$tag] GPU $gpu start $(date +%T) batch=$bs"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train_multitask.py \
    --model "$model" --tasks "$TASKS" --cot --zero-shot \
    --eval "$EVAL" --eval-limit "$EVAL_LIMIT" \
    --gen-batch "$bs" --max-new-tokens 320 \
    --out "$OUT_DIR/${tag}.jsonl" > "$OUT_DIR/${tag}.log" 2>&1
  echo "[$tag] GPU $gpu exit=$? $(date +%T)"
}

# Batch scales with free memory after weights: a 24B in bf16 eats ~48GB and
# leaves room for ~24; a 0.8B leaves nearly the whole card. GPU 4 carries only
# the 24B because it sets the critical path.
( run 4 cot_mistral_small_24b "mistralai/Mistral-Small-24B-Instruct-2501" 24 ) &
( run 5 cot_gemma2_9b   "google/gemma-2-9b-it"                 32
  run 5 cot_qwen35_4b   "Qwen/Qwen3.5-4B"                      64
  run 5 cot_qwen35_08b  "Qwen/Qwen3.5-0.8B"                    96 ) &
( run 6 cot_llama31_8b  "meta-llama/Llama-3.1-8B-Instruct"     32
  run 6 cot_ministral_8b "mistralai/Ministral-8B-Instruct-2410" 32
  run 6 cot_llama32_1b  "meta-llama/Llama-3.2-1B-Instruct"     96 ) &
( run 7 cot_mistral_7b  "mistralai/Mistral-7B-Instruct-v0.3"   32
  run 7 cot_llama32_3b  "meta-llama/Llama-3.2-3B-Instruct"     64
  run 7 cot_qwen35_2b   "Qwen/Qwen3.5-2B"                      80 ) &

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
