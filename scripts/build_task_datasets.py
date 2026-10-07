"""Materialize the four world-modeling task datasets, balancing where needed.

Measured positive rates on the multi-year corpus:

    controversiality   3.8 - 4.2%   ~23:1   <- severely imbalanced, balanced here
    width > 0         26 - 31%      ~2.5:1  <- mild; left at natural prevalence
    score              regression           <- no class label to balance
    reply_summary      generation           <- no class label to balance

Only controversiality is rebalanced. It is undersampled on the NEGATIVE class
in TRAIN, keeping every positive, which costs nothing a classifier wanted and
removes the collapse-to-majority failure mode seen earlier.

For TEST both versions are written and they answer different questions:
  * natural  -- real prevalence, so precision/recall mean what they mean in
                deployment. This is the one to report.
  * balanced -- equal classes, so F1 and accuracy are readable against a 0.5
                chance line. Useful for comparing models, misleading as an
                absolute capability number.
Balancing only ever DROPS records, so the post-disjointness of the underlying
splits is preserved.

Tasks 1 and 2 read different targets off the same records, so they share the
pooled split files rather than duplicating ~1.5GB of JSONL on disk.

Usage:
    PYTHONPATH=src python3 scripts/build_task_datasets.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

D = "/scratch/daweili5/smwm/data_multiyear_v2"
SUBSET = f"{D}/summary_subset.jsonl"


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def balance_binary(src: Path, dst: Path, key, ratio: float, seed: int):
    """Keep every positive; reservoir-sample `ratio` x that many negatives."""
    rng = random.Random(seed)
    pos, neg_res, n_neg = [], [], 0
    for line in iter_jsonl(src):
        r = json.loads(line)
        if key(r):
            pos.append(line)
        else:
            n_neg += 1
            cap = int(len(pos) * ratio) + 1024      # grows as positives are found
            if len(neg_res) < cap:
                neg_res.append(line)
            else:
                j = rng.randrange(n_neg)
                if j < cap:
                    neg_res[j] = line
    keep_neg = min(int(len(pos) * ratio), len(neg_res))
    rng.shuffle(neg_res)
    rows = pos + neg_res[:keep_neg]
    rng.shuffle(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")
    return len(pos), keep_neg, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    ap.add_argument("--out", default=f"{D}/tasks")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="negatives per positive in the balanced sets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d, out = Path(args.dir), Path(args.out)
    contr = lambda r: int(r["ground_truth"]["controversiality"]) == 1  # noqa: E731

    stats = {}
    for reg in ("random", "temporal"):
        for sp in ("train", "test"):
            src = d / f"{reg}_{sp}.jsonl"
            dst = out / "controversiality" / f"{reg}_{sp}_balanced.jsonl"
            p, n, tot_neg = balance_binary(src, dst, contr, args.ratio, args.seed)
            stats[f"contr_{reg}_{sp}"] = {
                "natural_n": p + tot_neg, "natural_pos": p,
                "balanced_n": p + n, "balanced_pos": p,
                "path": str(dst),
            }
            print(f"[contr] {reg}_{sp}: natural {p+tot_neg:,} ({p:,} pos) "
                  f"-> balanced {p+n:,} ({p:,} pos / {n:,} neg)")

    # ---- task 4: summary subset, split to match each regime's record ids ----
    if Path(SUBSET).exists():
        sub = [json.loads(l) for l in iter_jsonl(SUBSET)]
        usable = [r for r in sub if r.get("summarizable")]
        for reg in ("random", "temporal"):
            ids = {json.loads(l)["record_id"] for l in iter_jsonl(d / f"{reg}_train.jsonl")}
            tr = [r for r in usable if r["record_id"] in ids]
            te = [r for r in usable if r["record_id"] not in ids]
            for sp, rows in (("train", tr), ("test", te)):
                dst = out / "summary" / f"{reg}_{sp}.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[f"summary_{reg}"] = {"train": len(tr), "test": len(te),
                                       "total_usable": len(usable), "total": len(sub)}
            print(f"[summary] {reg}: train {len(tr):,} / test {len(te):,} "
                  f"(of {len(usable):,} summarizable)")

    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out}/manifest.json")


if __name__ == "__main__":
    main()
