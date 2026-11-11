#!/usr/bin/env bash
# CoT sweep followed immediately by LLM-as-a-judge, in one job.
#
# Chaining avoids an artifact round-trip: the sweep's predictions are already on
# this machine's disk, so the judge reads them directly instead of requiring a
# commit-and-redispatch cycle between the two halves.
set -uo pipefail
RUN_NAME="${RUN_NAME:?need RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
OUT_DIR="runs/$RUN_NAME"; mkdir -p "$OUT_DIR"
EVAL="${FC_EVAL:-$DATA_ROOT/forecast_eval.jsonl.gz}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
JUDGE_SAMPLE="${JUDGE_SAMPLE:-500}"   # per model; 500 gives a mean to about +-0.08

echo "===== PHASE 1: CoT inference ====="
RUN_NAME="$RUN_NAME" DATA_ROOT="$DATA_ROOT" ./scripts/gpu_cot_sweep.sh || {
  echo "[cotjudge] CoT phase failed; not judging" >&2; exit 1; }

echo "===== PHASE 2: extract judge-ready summaries ====="
PRED="$OUT_DIR/preds"; mkdir -p "$PRED"
uv run python scripts/collect_predictions.py --run-dir "$OUT_DIR" --out "$PRED" --prefix cot_ --clear
ls "$PRED"/*.jsonl >/dev/null 2>&1 || { echo "[cotjudge] FATAL: no summaries to judge" >&2; exit 1; }

echo "===== PHASE 3: LLM-as-a-judge (dual reference) ====="
FAILED=0
for pf in "$PRED"/*.jsonl; do
  tag="$(basename "$pf" .jsonl)"
  echo "[judge] === $tag ($(wc -l < "$pf") summaries, sampling $JUDGE_SAMPLE) $(date +%T) ==="
  for i in 0 1 2 3; do
    ( CUDA_VISIBLE_DEVICES=$((i+4)) uv run python scripts/judge_reply_predictions.py \
        --subset "$EVAL" --predictions "$pf" \
        --output "$OUT_DIR/ratings_${tag}_$i.jsonl" \
        --judge "$JUDGE_MODEL" --batch-size "${JUDGE_BATCH:-4}" \
        --shard "$i" --num-shards 4 --limit $((JUDGE_SAMPLE/4)) \
        --engine "${JUDGE_ENGINE:-auto}" --load-in-4bit \
        > "$OUT_DIR/judge_${tag}_$i.log" 2>&1 ) &
  done
  wait
  cat "$OUT_DIR"/ratings_${tag}_*.jsonl > "$OUT_DIR/ratings_${tag}.jsonl" 2>/dev/null
  n=$(wc -l < "$OUT_DIR/ratings_${tag}.jsonl" 2>/dev/null || echo 0)
  echo "[judge] $tag -> $n ratings"
  if [ "$n" -eq 0 ]; then
    echo "[judge] $tag produced nothing; shard 0 tail:" >&2
    tail -6 "$OUT_DIR/judge_${tag}_0.log" >&2 2>/dev/null; FAILED=$((FAILED+1))
  fi
done

uv run python - "$OUT_DIR" <<'PYEOF'
import glob, json, os, re, sys
from collections import Counter
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "ratings_*.jsonl"))):
    tag = os.path.basename(f)[len("ratings_"):-6]
    if re.search(r"_\d$", tag):
        continue
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    e = {"n": len(rows)}
    for col in ("vs_replies", "vs_summary"):
        v = [r[col] for r in rows if r.get(col)]
        e[col] = {"n": len(v), "mean": (sum(v)/len(v) if v else None),
                  "dist": dict(sorted(Counter(v).items()))}
    out[tag] = e
json.dump(out, open(os.path.join(sys.argv[1], "judge_summary.json"), "w"), indent=2)
print(f"\n{'model':26s}{'n':>6}{'vs_replies':>12}{'vs_summary':>12}")
print("-"*56)
for k, v in sorted(out.items(), key=lambda kv: -(kv[1]["vs_replies"]["mean"] or 0)):
    r, s = v["vs_replies"]["mean"], v["vs_summary"]["mean"]
    print(f"{k:26s}{v['n']:>6}{(f'{r:.3f}' if r else '--'):>12}{(f'{s:.3f}' if s else '--'):>12}")
PYEOF
[ "$FAILED" -gt 0 ] && { echo "[cotjudge] FATAL: $FAILED model(s) unjudged" >&2; exit 1; }
echo "[cotjudge] done $(date +%T)"
