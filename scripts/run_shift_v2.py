"""Cross-domain and cross-time generalization on the multi-year corpus.

Repeats the project's earlier generalization study on the rebuilt data, where
the earlier version was limited to one month of r/politics with a leaky split
and width>=1 by construction. Here:

  cross-domain : fit on r/politics, test on each other subreddit (time pooled)
  cross-time   : fit on the earliest months, test on each later month, so the
                 horizon runs forward -- the direction a world model is used in

A noise cell rides along in both: the in-domain ablation showed rho rises ~0.03
from feature-block width alone at small n, so a "generalization" gap smaller
than that is not evidence of anything.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from smwm.baselines import registry  # noqa: E402
from smwm.config import load_yaml  # noqa: E402
from smwm.metrics.pointwise import f1_binary_macro  # noqa: E402
from smwm.metrics.ranking import spearman  # noqa: E402

CORPUS = "/scratch/daweili5/smwm/data_multiyear_v2"
KEEP = ("record_id", "context", "stimulus", "ground_truth", "meta")
MODELS = ["structural_prior", "feature_gbdt", "quantile_gbm", "gnn", "glm_poisson", "hawkes"]


def load_cells(corpus: Path, per_cell: int, seed: int):
    rng = random.Random(seed)
    by = defaultdict(list)
    for f in sorted(corpus.glob("*_train.jsonl")) + sorted(corpus.glob("*_test.jsonl")):
        if f.name.startswith(("random_", "temporal_")):
            continue
        sub, month = f.name.replace("_train.jsonl", "").replace("_test.jsonl", "").rsplit("_", 1)
        rows = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append({k: r[k] for k in KEEP if k in r})
        by[(sub, month)].extend(rows)
    for k in by:
        rng.shuffle(by[k])
        by[k] = by[k][:per_cell]
    return by


def ev(model, recs):
    out = model.predict_batch(recs) if hasattr(model, "predict_batch") else [model.predict(r) for r in recs]
    ps = np.array([o["score"] for o in out], float)
    pw = np.array([o["width"] for o in out], float)
    pc = np.array([o["controversiality"] for o in out], int)
    ts = np.array([r["ground_truth"]["score"] for r in recs], float)
    tw = np.array([r["ground_truth"]["width"] for r in recs], float)
    tc = np.array([r["ground_truth"]["controversiality"] for r in recs], int)
    return {"n": len(recs),
            "score_rho": spearman(ps, ts)["spearman_rho"],
            "width_rho": spearman(pw, tw)["spearman_rho"],
            "any_f1": f1_binary_macro((pw > 0).astype(int).tolist(), (tw > 0).astype(int).tolist()),
            "contr_f1": f1_binary_macro(pc.tolist(), tc.tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--per-cell", type=int, default=1200)
    ap.add_argument("--train-cap", type=int, default=40000)
    ap.add_argument("--home", default="politics")
    ap.add_argument("--train-until", default="2018-10")
    ap.add_argument("--out", default="data_analysis/shift_v2.json")
    args = ap.parse_args()

    by = load_cells(Path(args.corpus), args.per_cell, 42)
    subs = sorted({s for s, _ in by}); months = sorted({m for _, m in by})
    print(f"cells={len(by)} subs={subs} months={len(months)}")
    res = {"cross_domain": {}, "cross_time": {}}

    def fit(name, train):
        cfg = load_yaml(f"configs/baseline/{name}.yaml")
        m = registry.get(cfg["baseline"])(**(cfg.get("params") or {}))
        m.fit(train); return m

    # ---------------- cross-domain: fit on home subreddit, test on others -----
    home = [r for (s, _), rows in by.items() if s == args.home for r in rows][: args.train_cap]
    print(f"\n[cross-domain] fit on r/{args.home}: {len(home):,} records")
    for name in MODELS:
        t0 = time.time(); m = fit(name, home); res["cross_domain"][name] = {}
        for s in subs:
            test = [r for (ss, _), rows in by.items() if ss == s for r in rows]
            test = test[:4000]
            res["cross_domain"][name][s] = ev(m, test)
        line = "  ".join(f"{s[:11]}={res['cross_domain'][name][s]['score_rho']:.3f}" for s in subs)
        print(f"  {name:17s} {line}  ({time.time()-t0:.0f}s)")

    # ---------------- cross-time: fit on early months, test forward -----------
    tr_months = [m for m in months if m <= args.train_until]
    te_months = [m for m in months if m > args.train_until]
    early = [r for (_, mo), rows in by.items() if mo in tr_months for r in rows][: args.train_cap]
    print(f"\n[cross-time] fit on {tr_months[0]}..{tr_months[-1]} ({len(early):,} records), "
          f"test {len(te_months)} later months")
    for name in MODELS:
        t0 = time.time(); m = fit(name, early); res["cross_time"][name] = {}
        for mo in te_months:
            test = [r for (_, mm), rows in by.items() if mm == mo for r in rows]
            res["cross_time"][name][mo] = ev(m, test)
        y = [res["cross_time"][name][mo]["score_rho"] for mo in te_months]
        h = [(int(mo[:4]) - int(args.train_until[:4])) * 12 + int(mo[5:7]) - int(args.train_until[5:7])
             for mo in te_months]
        ok = ~np.isnan(np.array(y, float))
        slope = np.polyfit(np.array(h)[ok], np.array(y)[ok], 1)[0] * 12 if ok.sum() > 2 else float("nan")
        print(f"  {name:17s} first={y[0]:.3f} last={y[-1]:.3f}  slope={slope:+.4f} rho/yr  ({time.time()-t0:.0f}s)")
        res["cross_time"][name]["_slope_per_year"] = float(slope)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "subs": subs, "months": months,
         "train_months": tr_months, "test_months": te_months, **res}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
