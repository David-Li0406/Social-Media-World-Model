#!/usr/bin/env bash
# Judge every forecaster's summaries against the same reference, sharded 4 ways.
# Each model's predictions are rated twice (real replies, reference summary),
# so forecasters are comparable to each other and to replyforecast-004.
set -uo pipefail
RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"; mkdir -p "$OUT_DIR"
SUBSET="${FC_EVAL:-$DATA_ROOT/forecast_eval.jsonl.gz}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
PRED_DIR="${PRED_DIR:-forecast_preds}"
FAILED=0

for pf in "$PRED_DIR"/*.jsonl; do
  tag="$(basename "$pf" .jsonl)"
  echo "[judge] === $tag ($(wc -l < "$pf") summaries) $(date +%T) ==="
  for i in 0 1 2 3; do
    gpu=$((i+4))
    ( CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/judge_reply_predictions.py \
        --subset "$SUBSET" --predictions "$pf" \
        --output "$OUT_DIR/ratings_${tag}_$i.jsonl" \
        --judge "$JUDGE_MODEL" --batch-size "${JUDGE_BATCH:-4}" \
        --shard "$i" --num-shards 4 --engine "${JUDGE_ENGINE:-auto}" --load-in-4bit \
        > "$OUT_DIR/judge_${tag}_$i.log" 2>&1
      echo "[judge] $tag shard $i GPU $gpu exit=$?" ) &
  done
  wait
  cat "$OUT_DIR"/ratings_${tag}_*.jsonl > "$OUT_DIR/ratings_${tag}.jsonl" 2>/dev/null
  n=$(wc -l < "$OUT_DIR/ratings_${tag}.jsonl" 2>/dev/null || echo 0)
  echo "[judge] $tag -> $n ratings"
  if [ "$n" -eq 0 ]; then
    echo "[judge] $tag produced nothing; shard 0 log tail:" >&2
    tail -6 "$OUT_DIR/judge_${tag}_0.log" >&2 2>/dev/null
    FAILED=$((FAILED+1))
  fi
done

uv run python - "$OUT_DIR" <<'PYEOF'
import glob, json, os, sys
from collections import Counter
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "ratings_*.jsonl"))):
    tag = os.path.basename(f)[len("ratings_"):-6]
    if tag[-2:-1] == "_" and tag[-1].isdigit():
        continue                                  # skip per-shard files
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    e = {"n": len(rows)}
    for col in ("vs_replies", "vs_summary"):
        v = [r[col] for r in rows if r.get(col)]
        e[col] = {"n": len(v), "mean": (sum(v)/len(v) if v else None),
                  "dist": dict(sorted(Counter(v).items()))}
    out[tag] = e
json.dump(out, open(os.path.join(sys.argv[1], "judge_summary.json"), "w"), indent=2)
for k, v in sorted(out.items(), key=lambda kv: -(kv[1]["vs_replies"]["mean"] or 0)):
    print(f"{k:26s} n={v['n']:>5}  vs_replies={v['vs_replies']['mean']:.3f}  "
          f"vs_summary={v['vs_summary']['mean']:.3f}")
PYEOF
if [ "$FAILED" -gt 0 ]; then
  echo "[judge] FATAL: $FAILED model(s) produced no ratings" >&2
  exit 1
fi
echo "[judge] done $(date +%T)"
