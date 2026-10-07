"""Profile every month in the archive BEFORE sampling from it.

The previous build picked months by intuition and landed 2023-09 -- whose
`controversiality` field is unset for the entire dump (6 positives in 20,000
records) -- directly into the temporal TEST set. A third of that test set was
therefore dead weight for one of the three targets.

This reads a few row groups per month instead of the whole file, so all 121
months can be screened in minutes rather than hours. For each month it reports
the fraction of rows with controversiality=1 and the volume of the target
subreddits, which is enough to reject dumps where the field is missing.

Usage:
    python3 scripts/profile_months.py --out data_analysis/month_profile.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ARCHIVE = "/scratch/daweili5/parquet_v2"
SUBS = ["politics", "news", "worldnews", "Conservative", "technology"]


def profile(path: Path, subs, n_groups: int, rows_cap: int) -> dict:
    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    # spread the probes through the file: a dump can be ordered, so the head
    # alone is not representative
    picks = sorted({int(i * (n_rg - 1) / max(n_groups - 1, 1)) for i in range(n_groups)})
    seen = 0
    contr = 0
    sub_counts = {s: 0 for s in subs}
    sub_contr = {s: 0 for s in subs}
    for rg in picks:
        t = pf.read_row_group(rg, columns=["subreddit", "controversiality", "score"])
        c = t.column("controversiality")
        seen += t.num_rows
        contr += pc.sum(pc.cast(c, "int64")).as_py() or 0
        sv = t.column("subreddit")
        for s in subs:
            m = pc.equal(sv, s)
            k = pc.sum(pc.cast(m, "int64")).as_py() or 0
            if k:
                sub_counts[s] += k
                sub_contr[s] += pc.sum(pc.cast(pc.filter(c, m), "int64")).as_py() or 0
        if seen >= rows_cap:
            break
    return {
        "rows_total": pf.metadata.num_rows,
        "rows_sampled": seen,
        "contr_rate": (contr / seen) if seen else 0.0,
        "sub_sampled": sub_counts,
        "sub_contr_rate": {s: (sub_contr[s] / sub_counts[s]) if sub_counts[s] else None
                           for s in subs},
        "sample_frac": (seen / pf.metadata.num_rows) if pf.metadata.num_rows else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--subreddits", nargs="*", default=SUBS)
    ap.add_argument("--groups", type=int, default=6, help="row groups probed per month")
    ap.add_argument("--rows-cap", type=int, default=3_000_000)
    ap.add_argument("--out", default="data_analysis/month_profile.json")
    args = ap.parse_args()

    files = sorted(Path(args.archive).glob("RC_*.parquet"))
    print(f"profiling {len(files)} months from {args.archive}")
    out = {}
    t0 = time.time()
    for i, f in enumerate(files, 1):
        month = f.stem.replace("RC_", "")
        try:
            out[month] = profile(f, args.subreddits, args.groups, args.rows_cap)
        except Exception as e:  # a corrupt dump must not kill the sweep
            out[month] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{i}/{len(files)}] {month}: ERROR {e}")
            continue
        r = out[month]
        print(f"  [{i}/{len(files)}] {month}: contr={100*r['contr_rate']:.3f}% "
              f"sampled={r['rows_sampled']:,} politics={r['sub_sampled'].get('politics',0):,} "
              f"({time.time()-t0:.0f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
