#!/usr/bin/env bash
# CPU half of the baseline benchmark. Runs on whatever machine holds the data
# (no GPU needed), on the SAME capped train/eval files the GPU lanes use, so
# every row in the final table is scored on identical records.
set -uo pipefail
OUT_DIR="${OUT_DIR:-runs/cpu_bench}"
TRAIN="${BENCH_TRAIN:-train_test/bench_temporal_train.jsonl.gz}"
EVAL="${BENCH_EVAL:-train_test/bench_temporal_eval.jsonl.gz}"
JOBS="${JOBS:-4}"
mkdir -p "$OUT_DIR"

BASELINES=(constant_mean constant_median structural_prior feature_gbdt
           quantile_gbm glm_poisson glm_tweedie hawkes gnn
           retrieval_tfidf retrieval_sbert)

run_one() {
  local b="$1"
  local cfg="configs/baseline/${b}.yaml"
  [ -f "$cfg" ] || { echo "[cpu] no config for $b"; return 0; }
  local t0=$SECONDS
  PYTHONPATH=src OMP_NUM_THREADS=4 python3 -m smwm.cli.run_inference \
    --config "$cfg" --train "$TRAIN" --test "$EVAL" --batch-size 512 \
    --results "$OUT_DIR/${b}.jsonl" > "$OUT_DIR/${b}.log" 2>&1
  echo "[cpu] $b exit=$? in $((SECONDS-t0))s"
}
export -f run_one
export OUT_DIR TRAIN EVAL

printf '%s\n' "${BASELINES[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
echo "[cpu] all done"
ls -la "$OUT_DIR"/*.jsonl 2>/dev/null | wc -l
