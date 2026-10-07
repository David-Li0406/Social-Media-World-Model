"""Cap the multi-year corpus to a benchmark that finishes inside a wall-clock budget.

Measured costs on this project set the caps:
  * generation-based LLM inference runs at 4.7 s/record unbatched, so a 60k
    test set is 78 h per LLM row. At 5,000 records with batched generation it
    is ~50 min, and the Spearman CI is still +-0.011 -- tighter than the
    +-0.018 the entire published benchmark was built on.
  * llm_sft trains at 2.26 samples/s, so its train cap is set separately and
    deliberately lower than the shared cap.

Records are written lean (no prompt/completion): those fields are 62% of the
bytes and every baseline regenerates the prompt from context+stimulus anyway.

Usage:
    PYTHONPATH=src python3 scripts/build_benchmark_subset.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

D = "/scratch/daweili5/smwm/data_multiyear_v2"
KEEP = ("record_id", "context", "stimulus", "ground_truth", "meta")


def sample(src: Path, n: int, seed: int):
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({k: r[k] for k in KEEP if k in r})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    ap.add_argument("--out", default="train_test")
    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--eval-n", type=int, default=5000)
    ap.add_argument("--regime", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.train_n), ("test", args.eval_n)):
        rows = sample(d / f"{args.regime}_{split}.jsonl", n, args.seed)
        name = f"bench_{args.regime}_{'train' if split=='train' else 'eval'}.jsonl.gz"
        with gzip.open(out / name, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        posts = len({r["meta"]["link_id"] for r in rows})
        pos = sum(r["ground_truth"]["controversiality"] for r in rows)
        w0 = sum(1 for r in rows if r["ground_truth"]["width"] == 0)
        mb = (out / name).stat().st_size / 1048576
        print(f"{name}: {len(rows):,} records, {posts:,} posts, "
              f"contr+ {pos:,} ({100*pos/len(rows):.2f}%), width=0 {100*w0/len(rows):.0f}%, "
              f"{mb:.1f} MB gz")

    # post-disjointness must survive the subsampling
    import gzip as gz
    def posts_of(p):
        with gz.open(p, "rt", encoding="utf-8") as f:
            return {json.loads(l)["meta"]["link_id"] for l in f if l.strip()}
    a = posts_of(out / f"bench_{args.regime}_train.jsonl.gz")
    b = posts_of(out / f"bench_{args.regime}_eval.jsonl.gz")
    print(f"\npost overlap train/eval: {len(a & b)} (must be 0)")


if __name__ == "__main__":
    main()
