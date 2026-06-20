"""Run LLM-as-a-judge over predicted reply_summaries with Qwen3-32B.

Judges multiple models on a COMMON sample of records (same record_ids judged
for every model -> fair comparison), then aggregates mean rating + distribution.

Usage (on the GPU runner):
  python scripts/judge_summaries.py \
      --results results/Qwen_Qwen3-4B_judge.jsonl results/Qwen_Qwen3-32B_judge.jsonl \
                results/llm_sft_qwen3_4b_judge.jsonl \
      --judge Qwen/Qwen3-32B --sample 200 --out runs/judge
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from smwm.eval.judge import SummaryJudge  # noqa: E402


def usable(rec):
    p = rec.get("predicted") or {}
    s = (p.get("reply_summary") or "").strip()
    g = ((rec.get("ground_truth") or {}).get("reply_summary") or "").strip()
    return len(s) >= 10 and len(g) >= 10 and g not in ("No replies.", "Failed")


def load_indexed(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r.get("record_id")] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen3-32B")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", default="runs/judge")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    indexed = {Path(p).stem: load_indexed(p) for p in args.results}
    # common record_ids usable in every model
    common = None
    for name, idx in indexed.items():
        ids = {rid for rid, r in idx.items() if usable(r)}
        common = ids if common is None else (common & ids)
    common = sorted(common)
    print(f"[judge] {len(common)} records usable across all {len(indexed)} models", flush=True)
    rng = random.Random(args.seed)
    if len(common) > args.sample:
        common = rng.sample(common, args.sample)
    print(f"[judge] judging {len(common)} common records with {args.judge}", flush=True)

    judge = SummaryJudge(model_id=args.judge)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / "judge_ratings.jsonl"
    summary = {}

    with open(ratings_path, "w") as rf:
        for name, idx in indexed.items():
            scores = []
            for i, rid in enumerate(common):
                rec = idx[rid]
                stim = (rec.get("stimulus") or {}).get("body", "")
                gt = (rec["ground_truth"]["reply_summary"] or "")
                pred = (rec["predicted"]["reply_summary"] or "")
                rating, raw = judge.rate(stim, gt, pred)
                rf.write(json.dumps({"model": name, "record_id": rid,
                                     "rating": rating, "raw": raw}, ensure_ascii=False) + "\n")
                rf.flush()
                if rating is not None:
                    scores.append(rating)
                if (i + 1) % 25 == 0:
                    print(f"[judge] {name}: {i+1}/{len(common)}", flush=True)
            n = len(scores)
            mean = sum(scores) / n if n else 0.0
            dist = dict(sorted(Counter(scores).items()))
            summary[name] = {"n": n, "mean_rating": round(mean, 3), "distribution": dist}
            print(f"[judge] {name}: n={n} mean={mean:.3f} dist={dist}", flush=True)

    with open(out_dir / "judge_summary.json", "w") as f:
        json.dump({"judge": args.judge, "sample": len(common), "models": summary}, f, indent=2)
    print(f"[judge] wrote {out_dir/'judge_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
