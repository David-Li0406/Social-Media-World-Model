"""Pool the per-cell multi-year splits into the two evaluation regimes.

Per-cell test slices are only ~800 records, far below the ~20k the measured
power curve requires to resolve 0.01-scale rho effects, so analyses must pool.
Two regimes are produced, and they answer different questions:

  random   -- posts split within every (month, subreddit) cell, then pooled.
              "How well does it predict engagement in a world it has seen?"
  temporal -- whole months held out at the end of the timeline.
              "How well does it predict the FUTURE?" This is the world-model
              question, and it is the one the old random-split benchmark could
              not ask.

Both are post-disjoint by construction; this script re-verifies that and also
checks for stimulus-id collisions across cells before writing anything.

Usage:
    PYTHONPATH=src python3 scripts/pool_multiyear.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TEMPORAL_TEST_MONTHS = ["2023-09", "2024-10", "2025-09"]


def read(p: Path):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_multiyear")
    ap.add_argument("--test-months", nargs="*", default=TEMPORAL_TEST_MONTHS)
    args = ap.parse_args()
    d = Path(args.dir)

    cells = sorted({p.name.rsplit("_", 1)[0] for p in d.glob("*_train.jsonl")})
    print(f"pooling {len(cells)} cells from {d}")

    # A post can appear in MORE THAN ONE cell: long-lived threads keep receiving
    # comments, so one link_id can show up in two sampled months. The per-cell
    # split cannot see that, so both regimes are re-derived globally here.
    all_rows, by_month = [], {}
    for cell in cells:
        month = cell.rsplit("_", 1)[-1]
        rows = list(read(d / f"{cell}_train.jsonl")) + list(read(d / f"{cell}_test.jsonl"))
        all_rows += rows
        by_month.setdefault(month, []).extend(rows)

    # --- random regime: assign each POST globally, once ---
    import random as _random
    posts = sorted({r["meta"]["link_id"] for r in all_rows})
    rng = _random.Random(42)
    rng.shuffle(posts)
    test_posts = set(posts[: int(len(posts) * 0.2)])
    rand_train = [r for r in all_rows if r["meta"]["link_id"] not in test_posts]
    rand_test = [r for r in all_rows if r["meta"]["link_id"] in test_posts]

    # --- temporal regime: months decide sides, then drop any TRAIN record whose
    # post also appears in test. Dropping from train (not test) keeps the
    # evaluation set intact and unbiased; the cost is a handful of records.
    temp_test = [r for m in args.test_months for r in by_month.get(m, [])]
    test_post_ids = {r["meta"]["link_id"] for r in temp_test}
    temp_train_raw = [r for m, rows in by_month.items() if m not in args.test_months
                      for r in rows]
    temp_train = [r for r in temp_train_raw if r["meta"]["link_id"] not in test_post_ids]
    dropped = len(temp_train_raw) - len(temp_train)
    if dropped:
        print(f"  temporal: dropped {dropped} train records whose post recurs in a test month")

    def verify(name, train, test):
        ptr = {r["meta"]["link_id"] for r in train}
        pte = {r["meta"]["link_id"] for r in test}
        str_ = {r["stimulus"]["id"] for r in train}
        ste = {r["stimulus"]["id"] for r in test}
        dup_tr = len(train) - len(str_)
        print(f"  {name:9s} train={len(train):>7,} test={len(test):>7,} "
              f"| post overlap={len(ptr & pte)} stimulus overlap={len(str_ & ste)} "
              f"| dup stimuli in train={dup_tr}")
        return len(ptr & pte) == 0 and len(str_ & ste) == 0 and dup_tr == 0

    ok_r = verify("random", rand_train, rand_test)
    ok_t = verify("temporal", temp_train, temp_test)
    if not (ok_r and ok_t):
        raise SystemExit("INTEGRITY CHECK FAILED - refusing to write")

    for name, rows in [("random_train", rand_train), ("random_test", rand_test),
                       ("temporal_train", temp_train), ("temporal_test", temp_test)]:
        out = d / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out} ({len(rows):,})")

    all_months = {c.rsplit("_", 1)[-1] for c in cells}
    print("\ntemporal test months:", sorted(args.test_months))
    print("temporal train months:", sorted(all_months - set(args.test_months)))


if __name__ == "__main__":
    main()
