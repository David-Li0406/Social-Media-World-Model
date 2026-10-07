"""Build the multi-year, multi-subreddit corpus from /scratch/daweili5/parquet_v2.

Twelve months spread over 2012-2025 x five subreddits, sampled so that the
resulting benchmark can actually support the claims the project wants to make:

  * width >= 0 (the old data was 100% width>=1 by construction)
  * post-disjoint train/test (the old split shared 90% of its posts)
  * one record per comment (the old chain01 had 11% duplicate stimuli)
  * >= 20k test records per slice, the size the measured power curve says is
    needed to resolve the ~0.01 rho effects this project argues about
  * per-record `meta` (depth, root_score, created_utc, subreddit) so analyses
    can stratify or reweight after the fact instead of rebuilding

The archive scan dominates cost (~30GB/month), so each month is streamed ONCE
for all five subreddits and cached to per-(month, subreddit) parquet under
$SMWM_SCRATCH/corpus_cache. Re-sampling afterwards is nearly free.

Usage:
    PYTHONPATH=src python3 scripts/build_multiyear.py --months 2024-10 --dry-run
    PYTHONPATH=src python3 scripts/build_multiyear.py            # full build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("SMWM_SCRATCH", "/scratch/daweili5/smwm")

from smwm.data.build_corpus import (  # noqa: E402
    build_records,
    extract_subreddits,
    post_disjoint_split,
    to_examples,
    write_split,
)

ARCHIVE = "/scratch/daweili5/parquet_v2"

# Stratified across eras and US election cycles, all within the period where
# these five subreddits carry real volume. r/politics exists from 2007 but the
# 2006-2011 files are far too thin for a 5-subreddit panel.
MONTHS = [
    "2012-10",  # Obama/Romney
    "2014-05",
    "2015-11",
    "2016-09",  # Trump/Clinton
    "2017-10",
    "2018-10",  # midterms
    "2020-08",  # Trump/Biden + COVID
    "2021-06",
    "2022-11",  # midterms
    "2023-09",
    "2024-10",  # Trump/Harris
    "2025-09",  # most recent in archive
]

SUBREDDITS = ["politics", "news", "worldnews", "Conservative", "technology"]

# 12 months x 5 subs x 4000 = 240k records ceiling; real yield will be lower
# for thin (month, subreddit) cells, which is why per-cell counts are logged.
PER_CELL_RECORDS = 4000
PER_CELL_POSTS = 1500
PER_POST_CAP = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--months", nargs="*", default=MONTHS)
    ap.add_argument("--subreddits", nargs="*", default=SUBREDDITS)
    ap.add_argument("--out", default="data_multiyear")
    ap.add_argument("--cache", default=None, help="parquet cache dir (default $SMWM_SCRATCH/corpus_cache)")
    ap.add_argument("--per-cell", type=int, default=PER_CELL_RECORDS)
    ap.add_argument("--per-cell-posts", type=int, default=PER_CELL_POSTS)
    ap.add_argument("--per-post-cap", type=int, default=PER_POST_CAP)
    ap.add_argument("--tail-cutoff-days", type=float, default=2.0)
    ap.add_argument("--min-root-score", type=int, default=0)
    ap.add_argument("--post-sampling", default="uniform", choices=["uniform", "pps"],
                    help="uniform = thread-diverse; pps = comment-representative")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="extract + report yields, write nothing")
    args = ap.parse_args()

    cache = Path(args.cache or (Path(os.environ["SMWM_SCRATCH"]) / "corpus_cache"))
    out_dir = Path(args.out)
    manifest: list[dict] = []
    next_id = 0

    for month in args.months:
        pq_path = Path(args.archive) / f"RC_{month}.parquet"
        if not pq_path.exists():
            print(f"[skip] {pq_path} not found")
            continue
        t0 = time.time()
        paths = extract_subreddits(pq_path, args.subreddits, cache, month)
        print(f"[extract] {month} took {time.time()-t0:.0f}s")

        for sub in args.subreddits:
            import pandas as pd

            df = pd.read_parquet(paths[sub])
            if df.empty:
                print(f"[build] {month}/{sub}: EMPTY, skipping")
                manifest.append({"month": month, "subreddit": sub, "comments": 0, "records": 0})
                continue
            recs = build_records(
                df,
                max_posts=args.per_cell_posts,
                max_records=args.per_cell,
                per_post_cap=args.per_post_cap,
                min_root_score=args.min_root_score,
                tail_cutoff_days=args.tail_cutoff_days,
                post_sampling=args.post_sampling,
                seed=args.seed,
            )
            zero = sum(1 for r in recs if r["ground_truth"]["width"] == 0)
            pos = sum(r["ground_truth"]["controversiality"] for r in recs)
            row = {
                "month": month, "subreddit": sub,
                "comments": int(len(df)), "posts": int(df["link_id"].nunique()),
                "records": len(recs),
                "width0_frac": round(zero / max(len(recs), 1), 4),
                "contr_rate": round(pos / max(len(recs), 1), 4),
            }
            manifest.append(row)
            print(f"[build] {month}/{sub}: {len(recs)} records "
                  f"(width=0 {100*row['width0_frac']:.1f}%, contr {100*row['contr_rate']:.2f}%)")

            if not args.dry_run and recs:
                ex = to_examples(recs, start_id=next_id)
                next_id += len(ex)
                train, test = post_disjoint_split(ex, seed=args.seed)
                write_split(out_dir, f"{sub}_{month}", train, test)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tot = sum(m["records"] for m in manifest)
    print(f"\ntotal records: {tot:,} across {len(manifest)} (month, subreddit) cells")
    print(f"manifest -> {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
