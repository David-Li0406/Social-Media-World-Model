#!/usr/bin/env bash
# Reply-forecasting pipeline across GPUs 4-7, in one workflow job.
#
# Three stages, each sharded four ways so every GPU is busy throughout:
#   1. reference  - summarize the REAL replies (the evaluation reference)
#   2. predict    - forecast replies from comment + metadata ONLY (never the replies)
#   3. judge      - rate each forecast twice: against the real replies, and
#                   against the generated summary, so the two can be compared
#
# The failed frozen_mlp baseline rides along on GPU 4 during stage 1, since that
# lane is otherwise just waiting on its shard.
set -uo pipefail

RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"; mkdir -p "$OUT_DIR"
SUBSET="${SUMMARY_INPUT:-$DATA_ROOT/summary_subset.jsonl}"
GPUS=(4 5 6 7)
SUM_BS="${SUM_BATCH:-8}"; PRED_BS="${PRED_BATCH:-16}"; JUDGE_BS="${JUDGE_BATCH:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-2000}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"

echo "[sum] subset=$SUBSET"; nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

# ---------- stage 0/1: frozen_mlp (GPU 4) + reference summaries (all GPUs) ----
( export CUDA_VISIBLE_DEVICES=4
  echo "[lane frozen_mlp] start $(date +%T)"
  uv run python -m smwm.cli.run_inference \
    --config configs/bench/frozen_mlp.yaml \
    --train "$DATA_ROOT/bench_temporal_train.jsonl.gz" \
    --test  "$DATA_ROOT/bench_temporal_eval.jsonl.gz" \
    --train-limit 50000 --batch-size 256 \
    --results "$OUT_DIR/frozen_mlp.jsonl" > "$OUT_DIR/frozen_mlp.log" 2>&1
  echo "[lane frozen_mlp] exit=$? $(date +%T)" ) &
FROZEN_PID=$!

for i in "${!GPUS[@]}"; do
  ( export CUDA_VISIBLE_DEVICES="${GPUS[$i]}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "[ref $i] GPU ${GPUS[$i]} start $(date +%T)"
    uv run python scripts/summarize_subset.py \
      --input "$SUBSET" --output "$OUT_DIR/reference_$i.jsonl" \
      --batch-size "$SUM_BS" --shard "$i" --num-shards 4 \
      > "$OUT_DIR/reference_$i.log" 2>&1
    echo "[ref $i] exit=$? $(date +%T)" ) &
done
wait
cat "$OUT_DIR"/reference_*.jsonl > "$OUT_DIR/reference_summaries.jsonl" 2>/dev/null
echo "[sum] references: $(wc -l < "$OUT_DIR/reference_summaries.jsonl") records"

# ---------- stage 2: forecast replies from the comment alone -----------------
# Evaluate on the held-out side of the subset.
uv run python - "$OUT_DIR/reference_summaries.jsonl" "$OUT_DIR/eval_pool.jsonl" "$EVAL_LIMIT" <<'PYEOF'
import json, sys
src, dst, lim = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
test = [r for r in rows if r.get("split") == "test" and r.get("reply_summary")]
if len(test) < lim:
    test += [r for r in rows if r.get("split") != "test" and r.get("reply_summary")][: lim - len(test)]
with open(dst, "w", encoding="utf-8") as f:
    for r in test[:lim]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[eval pool] {min(len(test), lim)} records")
PYEOF

for i in "${!GPUS[@]}"; do
  ( export CUDA_VISIBLE_DEVICES="${GPUS[$i]}"
    uv run python - "$OUT_DIR/eval_pool.jsonl" "$OUT_DIR/pool_$i.jsonl" "$i" <<'PYEOF'
import json, sys
src, dst, sh = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l for k, l in enumerate(open(src, encoding="utf-8")) if l.strip() and k % 4 == sh]
open(dst, "w", encoding="utf-8").writelines(rows)
PYEOF
    echo "[pred $i] GPU ${GPUS[$i]} start $(date +%T)"
    uv run python scripts/predict_reply_summary.py \
      --input "$OUT_DIR/pool_$i.jsonl" --output "$OUT_DIR/pred_$i.jsonl" \
      --batch-size "$PRED_BS" > "$OUT_DIR/pred_$i.log" 2>&1
    echo "[pred $i] exit=$? $(date +%T)" ) &
done
wait
cat "$OUT_DIR"/pred_*.jsonl > "$OUT_DIR/predictions.jsonl" 2>/dev/null
echo "[sum] predictions: $(wc -l < "$OUT_DIR/predictions.jsonl") records"

# ---------- stage 3: judge, dual reference ------------------------------------
for i in "${!GPUS[@]}"; do
  ( export CUDA_VISIBLE_DEVICES="${GPUS[$i]}"
    echo "[judge $i] GPU ${GPUS[$i]} start $(date +%T)"
    uv run python scripts/judge_reply_predictions.py \
      --subset "$OUT_DIR/reference_summaries.jsonl" \
      --predictions "$OUT_DIR/predictions.jsonl" \
      --output "$OUT_DIR/ratings_$i.jsonl" \
      --judge "$JUDGE_MODEL" --batch-size "$JUDGE_BS" \
      --shard "$i" --num-shards 4 \
      --engine "${JUDGE_ENGINE:-auto}" --gpu-mem-util "${JUDGE_GPU_UTIL:-0.90}" \
      --load-in-4bit \
      > "$OUT_DIR/judge_$i.log" 2>&1
    echo "[judge $i] exit=$? $(date +%T)" ) &
done
wait
cat "$OUT_DIR"/ratings_*.jsonl > "$OUT_DIR/judge_ratings.jsonl" 2>/dev/null

uv run python - "$OUT_DIR/judge_ratings.jsonl" "$OUT_DIR/judge_summary.json" <<'PYEOF'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
out = {"n": len(rows)}
for col in ("vs_replies", "vs_summary"):
    v = [r[col] for r in rows if r.get(col)]
    out[col] = {"n": len(v), "mean": (sum(v)/len(v) if v else None),
                "dist": dict(sorted(Counter(v).items()))}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
echo "[sum] all stages finished $(date +%T)"
