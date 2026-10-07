"""How fast does the engagement world model rot? Measured over ~11 years.

The published temporal analysis trained on politics 2025-12 and looked back at
most 16 months, concluding decay is "<=0.03 rho". That conclusion rests on four
test sets inside a single presidential term. The resampled multi-year corpus
spans 2014-2025 with a validated controversiality field, so the same question
can be asked at horizons up to a decade -- and asked FORWARD, which is the
direction a world model is actually used in.

Protocol: fit once on every record from the training window, then evaluate on
each later month separately. Horizon is the gap between the end of the window
and the test month. The noise cell travels along because the in-domain work
showed rho rises ~0.03 from feature-block width alone.

Usage:
    PYTHONPATH=src python3 scripts/run_temporal_decay.py --dir <corpus> --train-until 2017-10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("SMWM_SCRATCH", "/scratch/daweili5/smwm")
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")

from smwm.baselines.ablation import (  # noqa: E402
    FeatureAblation, SentenceEmbedder, semantic_texts,
)
from smwm.metrics.pointwise import f1_binary_macro  # noqa: E402
from smwm.metrics.ranking import spearman  # noqa: E402

CELLS = [
    ("S", ("structural",), {}),
    ("S+L", ("structural", "lexical"), {}),
    ("S+N768", ("structural", "noise"), {"noise_dim": 768}),
    ("S+E", ("structural", "semantic"), {}),
]


def months_between(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)


def load_by_month(d: Path, cap: int, seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    by_month: dict[str, list] = defaultdict(list)
    for p in sorted(d.glob("*_train.jsonl")) + sorted(d.glob("*_test.jsonl")):
        stem = p.name.replace("_train.jsonl", "").replace("_test.jsonl", "")
        month = stem.rsplit("_", 1)[-1]
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    # drop the prompt/completion duplicates (62% of bytes)
                    by_month[month].append(
                        {k: r[k] for k in
                         ("record_id", "context", "stimulus", "ground_truth", "meta")
                         if k in r})
    for m in by_month:
        rng.shuffle(by_month[m])
        if cap:
            by_month[m] = by_month[m][:cap]
    return dict(by_month)


def targets(recs):
    s = np.array([r["ground_truth"]["score"] for r in recs], float)
    w = np.array([r["ground_truth"]["width"] for r in recs], float)
    c = np.array([r["ground_truth"]["controversiality"] for r in recs], int)
    return s, w, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/daweili5/smwm/data_multiyear_v2")
    ap.add_argument("--train-until", default="2017-10")
    ap.add_argument("--per-month-cap", type=int, default=3000)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="data_analysis/temporal_decay.json")
    args = ap.parse_args()

    by_month = load_by_month(Path(args.dir), args.per_month_cap, seed=42)
    months = sorted(by_month)
    train_months = [m for m in months if m <= args.train_until]
    test_months = [m for m in months if m > args.train_until]
    train = [r for m in train_months for r in by_month[m]]
    print(f"train window {train_months[0]}..{train_months[-1]}  ({len(train):,} records)")
    print(f"test months: {len(test_months)}  horizons "
          f"{months_between(args.train_until, test_months[0])}..{months_between(args.train_until, test_months[-1])} months")

    embedder = SentenceEmbedder(args.embed_model)
    t0 = time.time()
    embedder.warm([t for m in months for r in by_month[m] for t in semantic_texts(r)])
    embedder.save()
    print(f"[embed] {time.time()-t0:.0f}s")

    results: dict[str, dict] = {}
    for name, blocks, kw in CELLS:
        model = FeatureAblation(blocks=blocks, embedder=embedder, **kw)
        t0 = time.time()
        model.fit(train)
        results[name] = {}
        for m in test_months:
            recs = by_month[m]
            out = model.predict_batch(recs)
            pw = np.array([o["width"] for o in out], float)
            ps = np.array([o["score"] for o in out], float)
            pc = np.array([o["controversiality"] for o in out], float)
            ts, tw, tc = targets(recs)
            results[name][m] = {
                "n": len(recs),
                "horizon_months": months_between(args.train_until, m),
                "score_rho": spearman(ps, ts)["spearman_rho"],
                "width_rho": spearman(pw, tw)["spearman_rho"],
                "width_any_f1": f1_binary_macro(((pw > 0).astype(int)).tolist(),
                                                ((tw > 0).astype(int)).tolist()),
                "contr_f1": f1_binary_macro(pc.astype(int).tolist(), tc.astype(int).tolist()),
            }
        print(f"[{name}] fit+eval {time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": {"dir": args.dir, "train_until": args.train_until,
                    "train_months": train_months, "n_train": len(train),
                    "per_month_cap": args.per_month_cap}, "cells": results}, indent=2))

    print("\n" + "=" * 100)
    print(f"SCORE rho vs FORWARD HORIZON  (fit on {train_months[0]}..{train_months[-1]})")
    print("=" * 100)
    print(f"{'test month':12s}{'horizon':>9s}" + "".join(f"{c[0]:>11s}" for c in CELLS))
    for m in test_months:
        h = results[CELLS[0][0]][m]["horizon_months"]
        print(f"{m:12s}{h:>6d}mo" + "".join(f"{results[c[0]][m]['score_rho']:>11.3f}" for c in CELLS))
    print()
    print("Trend: rho change per YEAR of horizon (OLS slope x 12)")
    for c, _, _ in CELLS:
        h = np.array([results[c][m]["horizon_months"] for m in test_months], float)
        y = np.array([results[c][m]["score_rho"] for m in test_months], float)
        ok = ~np.isnan(y)
        slope = np.polyfit(h[ok], y[ok], 1)[0] * 12 if ok.sum() > 2 else float("nan")
        print(f"  {c:8s} {slope:+.4f} rho/year   (first={y[ok][0]:.3f} last={y[ok][-1]:.3f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
