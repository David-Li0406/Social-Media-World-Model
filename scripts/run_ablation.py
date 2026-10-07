"""Controlled text-vs-structure ablation with paired bootstrap CIs.

Fits one identical LightGBM across six feature sets so the only thing that
changes is *what the model is allowed to see*:

    S      structural only          (graph + time)
    L      lexical only             (shallow text statistics)
    E      semantic only            (frozen sentence embeddings)
    S+L    structural + lexical     (~ the published feature_gbdt / quantile_gbm)
    S+E    structural + semantic    (the cell the benchmark never ran)
    S+L+E  everything

Differences between cells are tested with a PAIRED bootstrap: one resample of
the test indices is scored for both models, so the CI is on the *difference*,
which is what "does semantics actually add?" requires. An unpaired comparison
of two overlapping CIs would be far too conservative here.

Usage:
    PYTHONPATH=src python3 scripts/run_ablation.py [--embed-model ID] [--pca N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("SMWM_SCRATCH", "/scratch/daweili5/smwm")
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")

from smwm.baselines.ablation import (  # noqa: E402
    FeatureAblation,
    SentenceEmbedder,
    semantic_texts,
)
from smwm.data.io import read_jsonl  # noqa: E402

# `prompt` is 62% of a record's bytes and merely re-serializes context+stimulus
# for the LLM baselines; no feature cell reads it. Dropping it at parse time is
# the difference between fitting inside the 64GB SLURM cgroup and being
# OOM-killed at 66GB on the 383k-record multi-year train split.
_KEEP = ("record_id", "context", "stimulus", "ground_truth", "meta")


def read_lean(path, limit: int = 0, seed: int = 42):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r[k] for k in _KEEP if k in r})
    if limit and len(rows) > limit:
        import random as _r
        _r.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows
from smwm.metrics.pointwise import f1_binary_macro, mae  # noqa: E402
from smwm.metrics.ranking import pairwise_accuracy, spearman  # noqa: E402

TRAIN = "train_test/RC_2025-12_politics_width_chain01_train.jsonl"
TEST = "train_test/RC_2025-12_politics_width_chain01_test.jsonl"

# On the multi-year corpus width is ~70% zeros, so a plain Spearman on width is
# dominated by ties in the TRUTH (54% of random pairs). The engagement question
# splits cleanly in two, so score it that way as well:
#   hurdle : "will anyone reply at all?"  -- binary, well balanced at 70/30
#   rho|>0 : "given a reply, how many?"   -- comparable to the old width>=1 data
def hurdle_metrics(pred_w, true_w):
    import numpy as np
    from smwm.metrics.pointwise import f1_binary_macro
    pb = (np.asarray(pred_w) > 0).astype(int)
    tb = (np.asarray(true_w) > 0).astype(int)
    out = {"width_any_f1": f1_binary_macro(pb.tolist(), tb.tolist()),
           "width_any_rate_true": float(tb.mean()),
           "width_any_rate_pred": float(pb.mean())}
    m = np.asarray(true_w) > 0
    out["width_rho_pos"] = (spearman(np.asarray(pred_w)[m], np.asarray(true_w)[m])["spearman_rho"]
                            if m.sum() > 2 else float("nan"))
    out["n_width_pos"] = int(m.sum())
    return out

SUITES = {
    # The main question: what does each kind of "text" add over structure?
    "main": [
        ("S", ("structural",), {}),
        ("L", ("lexical",), {}),
        ("E", ("semantic",), {}),
        ("S+L", ("structural", "lexical"), {}),
        ("S+E", ("structural", "semantic"), {}),
        ("S+L+E", ("structural", "lexical", "semantic"), {}),
    ],
    # Falsification + decomposition of any semantic gain.
    #   S+N  : 768 columns of pure noise. If S+E only wins because LightGBM
    #          likes wide inputs, S+N wins too. It must not.
    #   S+Es : is the signal in what the stimulus said...
    #   S+Ep : ...or in what it was replying to?
    "control": [
        ("S", ("structural",), {}),
        ("S+N", ("structural", "noise"), {}),
        ("S+Es", ("structural", "semantic_stim"), {}),
        ("S+Ep", ("structural", "semantic_parent"), {}),
        ("S+E", ("structural", "semantic"), {}),
    ],
    # Every cell gets its own hyperparameter search, and each semantic cell is
    # paired with a noise block of IDENTICAL width. Only then is "semantics vs
    # more columns" a controlled comparison.
    "tuned": [
        ("S", ("structural",), {}),
        ("S+L", ("structural", "lexical"), {}),
        ("S+N384", ("structural", "noise"), {"noise_dim": 384}),
        ("S+Es", ("structural", "semantic_stim"), {}),
        ("S+N768", ("structural", "noise"), {"noise_dim": 768}),
        ("S+E", ("structural", "semantic"), {}),
        ("S+L+E", ("structural", "lexical", "semantic"), {}),
    ],
}

# Same 10 configs for every cell, so no cell is advantaged by a bigger search.
# Spans heavy regularization (which a 7-feature model may need) through wide,
# lightly-regularized trees (which a 775-feature model may need).
GRID = [
    dict(num_leaves=31, feature_fraction=1.0, min_child_samples=20),   # published default
    dict(num_leaves=15, feature_fraction=1.0, min_child_samples=20),
    dict(num_leaves=7, feature_fraction=1.0, min_child_samples=50),
    dict(num_leaves=31, feature_fraction=1.0, min_child_samples=100),
    dict(num_leaves=15, feature_fraction=1.0, min_child_samples=200),
    dict(num_leaves=31, feature_fraction=0.7, min_child_samples=20),
    dict(num_leaves=31, feature_fraction=0.3, min_child_samples=50),
    dict(num_leaves=63, feature_fraction=0.7, min_child_samples=50),
    dict(num_leaves=63, feature_fraction=0.3, min_child_samples=100),
    dict(num_leaves=127, feature_fraction=0.3, min_child_samples=200),
]

DELTA_PAIRS = {
    "main": [("S", "S+E"), ("S", "S+L"), ("S+L", "S+L+E"), ("S", "L"), ("S", "E"),
             ("S+L", "S+E"), ("S+E", "S+L+E")],
    "control": [("S", "S+N"), ("S", "S+Es"), ("S", "S+Ep"), ("S", "S+E"),
                ("S+N", "S+E"), ("S+Es", "S+E")],
    # The comparisons that matter are semantic-vs-noise at MATCHED width.
    "tuned": [("S", "S+L"), ("S", "S+N384"), ("S", "S+N768"), ("S", "S+Es"), ("S", "S+E"),
              ("S+N384", "S+Es"), ("S+N768", "S+E"), ("S+L", "S+L+E")],
}


def truths(records):
    s, w, c = [], [], []
    for r in records:
        g = r.get("ground_truth") or json.loads(r["completion"])
        s.append(float(g["score"]))
        w.append(float(g["width"]))
        c.append(int(g["controversiality"]))
    return np.asarray(s), np.asarray(w), np.asarray(c)


def point_metrics(pred, true):
    ps, pw, pc = pred
    ts, tw, tc = true
    return {
        "score_rho": spearman(ps, ts)["spearman_rho"],
        "score_pair_acc": pairwise_accuracy(ps, ts),
        "score_mae": mae(ps, ts),
        "width_rho": spearman(pw, tw)["spearman_rho"],
        "width_pair_acc": pairwise_accuracy(pw, tw),
        "width_mae": mae(pw, tw),
        "contr_f1": f1_binary_macro(pc.astype(int).tolist(), tc.astype(int).tolist()),
        "contr_pos_rate_pred": float(pc.mean()),
        **hurdle_metrics(pw, tw),
    }


def bootstrap_indices(n, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def boot_stat(pred, true, idx_matrix):
    """Bootstrap distributions for score rho, width rho, controversiality F1."""
    ps, pw, pc = pred
    ts, tw, tc = true
    out = {"score_rho": [], "width_rho": [], "contr_f1": []}
    for idx in idx_matrix:
        out["score_rho"].append(spearman(ps[idx], ts[idx])["spearman_rho"])
        out["width_rho"].append(spearman(pw[idx], tw[idx])["spearman_rho"])
        out["contr_f1"].append(
            f1_binary_macro(pc[idx].astype(int).tolist(), tc[idx].astype(int).tolist())
        )
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def ci(vals, lo=2.5, hi=97.5):
    v = vals[~np.isnan(vals)]
    if not len(v):
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="data_analysis/ablation_results.json")
    ap.add_argument("--tag", default=None, help="label for this run in the output json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N train / N//4 test")
    ap.add_argument("--suite", default="main", choices=sorted(SUITES))
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--train-limit", type=int, default=0,
                    help="subsample train (test is left whole; it drives comparison precision)")
    ap.add_argument("--test", default=TEST)
    ap.add_argument("--tune", action="store_true",
                    help="per-cell grid search on a held-out 20%% of train")
    args = ap.parse_args()

    cells = SUITES[args.suite]

    tag = args.tag or (args.suite + "_" + args.embed_model.split("/")[-1]
                       + (f"_pca{args.pca}" if args.pca else ""))

    train = read_lean(args.train, limit=args.train_limit, seed=42)
    test = read_lean(args.test)
    if args.limit:
        train, test = train[: args.limit], test[: max(2, args.limit // 4)]
    print(f"train={len(train)}  test={len(test)}  embed_model={args.embed_model}  pca={args.pca}")
    true = truths(test)

    # One shared embedder => the GPU pass is paid once across all six cells.
    embedder = SentenceEmbedder(args.embed_model)
    t0 = time.time()
    embedder.warm([t for r in train + test for t in semantic_texts(r)])
    embedder.save()
    print(f"[embed] warm done in {time.time()-t0:.1f}s  dim={embedder.dim}")

    # Held-out slice of train for hyperparameter selection. build_splits already
    # shuffled with seed 42, so a tail slice is a random split.
    n_fit = int(len(train) * 0.8)
    tr_inner, valid = train[:n_fit], train[n_fit:]
    valid_true = truths(valid)

    def select_params(blocks, kw):
        """Pick the grid config maximizing mean(score rho, width rho) on valid."""
        best, best_cfg = -np.inf, GRID[0]
        for cfg in GRID:
            m = FeatureAblation(blocks=blocks, pca_dim=args.pca, embedder=embedder, **kw, **cfg)
            m.fit(tr_inner)
            out = m.predict_batch(valid)
            ps = np.asarray([o["score"] for o in out], dtype=float)
            pw = np.asarray([o["width"] for o in out], dtype=float)
            r_s = spearman(ps, valid_true[0])["spearman_rho"]
            r_w = spearman(pw, valid_true[1])["spearman_rho"]
            obj = np.nanmean([r_s, r_w])
            if obj > best:
                best, best_cfg = obj, cfg
        return best_cfg, float(best)

    preds: dict[str, tuple] = {}
    results: dict[str, dict] = {}
    for name, blocks, kw in cells:
        t0 = time.time()
        cfg, valid_obj = ({}, float("nan"))
        if args.tune:
            cfg, valid_obj = select_params(blocks, kw)
            print(f"  {name:7s} tuned -> {cfg}  (valid mean rho={valid_obj:.4f})")
        model = FeatureAblation(blocks=blocks, pca_dim=args.pca, embedder=embedder, **kw, **cfg)
        model.fit(train)
        out = model.predict_batch(test)
        ps = np.asarray([o["score"] for o in out], dtype=float)
        pw = np.asarray([o["width"] for o in out], dtype=float)
        pc = np.asarray([o["controversiality"] for o in out], dtype=float)
        preds[name] = (ps, pw, pc)
        m = point_metrics(preds[name], true)
        m["n_features"] = int(model._project(model._matrix(test[:2]), fit=False).shape[1])  # noqa: E501
        m["fit_predict_sec"] = round(time.time() - t0, 1)
        m["params"] = cfg
        m["valid_obj"] = valid_obj
        results[name] = m
        del model
        import gc as _gc; _gc.collect()
        print(f"  {name:7s} score_rho={m['score_rho']:.4f}  width_rho={m['width_rho']:.4f}  "
              f"anyF1={m['width_any_f1']:.4f}  rho|>0={m['width_rho_pos']:.4f}  "
              f"contrF1={m['contr_f1']:.4f}  d={m['n_features']}  ({m['fit_predict_sec']}s)")

    # ---- bootstrap: shared resample indices across cells => paired deltas ----
    print(f"\n[bootstrap] {args.n_boot} resamples (paired across cells)...")
    idx = bootstrap_indices(len(test), args.n_boot, seed=0)
    boots = {name: boot_stat(preds[name], true, idx) for name, _, _ in cells}
    for name in results:
        for metric in ("score_rho", "width_rho", "contr_f1"):
            lo, hi = ci(boots[name][metric])
            results[name][f"{metric}_ci95"] = [lo, hi]

    deltas: dict[str, dict] = {}
    for ref, cmp_ in DELTA_PAIRS[args.suite]:
        d = {}
        for metric in ("score_rho", "width_rho", "contr_f1"):
            diff = boots[cmp_][metric] - boots[ref][metric]
            lo, hi = ci(diff)
            point = results[cmp_][metric] - results[ref][metric]
            d[metric] = {
                "delta": float(point),
                "ci95": [lo, hi],
                # two-sided bootstrap p: how often the sign flips
                "p_two_sided": float(2 * min(
                    (diff <= 0).mean(), (diff >= 0).mean())),
            }
        deltas[f"{cmp_} - {ref}"] = d

    payload = {
        "config": {
            "suite": args.suite,
            "embed_model": args.embed_model,
            "pca": args.pca,
            "n_boot": args.n_boot,
            "tuned": bool(args.tune),
            "train": args.train, "test": args.test,
            "n_train": len(train), "n_test": len(test),
        },
        "cells": results,
        "paired_deltas": deltas,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[tag] = payload
    out_path.write_text(json.dumps(existing, indent=2))

    # ------------------------------------------------------------- reporting
    print("\n" + "=" * 92)
    print(f"ABLATION  ({tag})   n_test={len(test)}   [95% bootstrap CI]")
    print("=" * 92)
    print(f"{'cell':7s}{'dim':>6}{'score rho':>22}{'width rho':>22}{'contr F1':>20}")
    for name, _, _ in cells:
        m = results[name]
        print(f"{name:7s}{m['n_features']:>6}"
              f"{m['score_rho']:>10.4f} [{m['score_rho_ci95'][0]:.3f},{m['score_rho_ci95'][1]:.3f}]"
              f"{m['width_rho']:>10.4f} [{m['width_rho_ci95'][0]:.3f},{m['width_rho_ci95'][1]:.3f}]"
              f"{m['contr_f1']:>8.4f} [{m['contr_f1_ci95'][0]:.3f},{m['contr_f1_ci95'][1]:.3f}]")

    print("\n" + "=" * 92)
    print("PAIRED DELTAS (does the added block actually help?)")
    print("=" * 92)
    print(f"{'comparison':20s}{'score rho delta':>34}{'width rho delta':>34}")
    for k, d in deltas.items():
        s, w = d["score_rho"], d["width_rho"]
        print(f"{k:20s}"
              f"{s['delta']:>+10.4f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] p={s['p_two_sided']:.3f}"
              f"{w['delta']:>+10.4f} [{w['ci95'][0]:+.3f},{w['ci95'][1]:+.3f}] p={w['p_two_sided']:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
