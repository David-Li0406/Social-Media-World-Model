#!/usr/bin/env bash
# Run the GPU half of the baseline benchmark across all four GPUs at once.
#
# A self-hosted runner executes one workflow job at a time, so four dispatches
# would queue rather than parallelize. This job fans out itself: lanes are
# grouped by GPU, each group runs in its own background subshell pinned with
# CUDA_VISIBLE_DEVICES, and the job waits for all of them.
#
# LANES selects which lanes to run (default: all). Re-running a failed subset
# costs only that subset -- e.g. LANES="llm_reghead llm_sft frozen_mlp".
set -uo pipefail

RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"
TRAIN="${BENCH_TRAIN:-$DATA_ROOT/bench_temporal_train.jsonl.gz}"
EVAL="${BENCH_EVAL:-$DATA_ROOT/bench_temporal_eval.jsonl.gz}"
GEN_BS="${GEN_BATCH:-16}"
SFT_TRAIN_LIMIT="${SFT_TRAIN_LIMIT:-12000}"

# tag | gpu | config | train_limit | batch_size
LANE_SPEC="encoder|4|configs/bench/encoder.yaml|50000|256
frozen_mlp|4|configs/bench/frozen_mlp.yaml|50000|256
llm_reghead|5|configs/bench/llm_reghead.yaml|50000|64
llm_sft|6|configs/bench/llm_sft.yaml|${SFT_TRAIN_LIMIT}|${GEN_BS}
llm_zeroshot|7|configs/bench/llm_zeroshot.yaml|2000|${GEN_BS}
llm_fewshot|7|configs/bench/llm_fewshot.yaml|2000|${GEN_BS}
llm_cot|7|configs/bench/llm_cot.yaml|2000|${GEN_BS}"

LANES="${LANES:-encoder frozen_mlp llm_reghead llm_sft llm_zeroshot llm_fewshot llm_cot}"

mkdir -p "$OUT_DIR"
echo "[bench] train=$TRAIN"
echo "[bench] eval =$EVAL"
echo "[bench] lanes=$LANES"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

selected() { case " $LANES " in *" $1 "*) return 0;; *) return 1;; esac; }

# Group the selected lanes by GPU so same-GPU lanes run sequentially and
# different-GPU lanes run concurrently.
for gpu in 4 5 6 7; do
  group=""
  while IFS='|' read -r tag g cfg tlim bs; do
    [ -z "$tag" ] && continue
    [ "$g" = "$gpu" ] || continue
    selected "$tag" || continue
    group+="$tag|$cfg|$tlim|$bs"$'\n'
  done <<< "$LANE_SPEC"
  [ -z "$group" ] && continue
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    while IFS='|' read -r tag cfg tlim bs; do
      [ -z "$tag" ] && continue
      echo "[lane $tag] GPU $gpu start $(date +%T)"
      uv run python -m smwm.cli.run_inference \
        --config "$cfg" --train "$TRAIN" --test "$EVAL" \
        --train-limit "$tlim" --batch-size "$bs" \
        --results "$OUT_DIR/${tag}.jsonl" > "$OUT_DIR/${tag}.log" 2>&1
      echo "[lane $tag] GPU $gpu exit=$? $(date +%T)"
    done <<< "$group"
  ) &
done

wait
echo "[bench] all lanes finished $(date +%T)"
for t in $LANES; do
  if [ -s "$OUT_DIR/$t.jsonl" ]; then
    echo "[bench] OK   $t ($(wc -l < "$OUT_DIR/$t.jsonl") records)"
  else
    echo "[bench] FAIL $t -- no results; tail of its log:"
    tail -5 "$OUT_DIR/$t.log" 2>/dev/null | sed 's/^/    /'
  fi
done
uv run python -m smwm.cli.run_eval "$OUT_DIR"/*.jsonl 2>&1 | tee "$OUT_DIR/metrics.txt" || true
