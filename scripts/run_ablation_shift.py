"""Does any semantic gain survive domain / temporal shift?

The in-domain ablation asks whether embedding the comment text helps at all.
This asks the world-model question: does that help *transfer*? Semantic
features are the obvious candidate to be domain-specific -- politics lexicon
learned on r/politics need not mean anything on r/technology -- whereas the
structural signal should be domain-general.

Each cell is fitted ONCE on the politics train split (reusing the
hyperparameters chosen by run_ablation.py --tune) and then evaluated on every
shifted test set. The noise cells travel along as the dimensionality control,
because the "wider input helps LightGBM" effect found in-domain would
otherwise masquerade as transferable semantic signal.

Usage:
    PYTHONPATH=src python3 scripts/run_ablation_shift.py
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
from smwm.metrics.pointwise import f1_binary_macro  # noqa: E402
from smwm.metrics.ranking import spearman  # noqa: E402
from run_ablation import (  # noqa: E402
    bootstrap_indices,
    boot_stat,
    ci,
    truths,
)

TRAIN = "train_test/RC_2025-12_politics_width_chain01_train.jsonl"
DOM = "data_analysis/domains"

TEST_SETS = [
    ("politics_2025-12 (in-domain)", "train_test/RC_2025-12_politics_width_chain01_test.jsonl"),
    ("politics_2025-09", f"{DOM}/politics_2025-09_test.jsonl"),
    ("politics_2025-02", f"{DOM}/politics_2025-02_test.jsonl"),
    ("politics_2024-08", f"{DOM}/politics_2024-08_test.jsonl"),
    ("news_2025-09", f"{DOM}/news_2025-09_test.jsonl"),
    ("worldnews_2025-09", f"{DOM}/worldnews_2025-09_test.jsonl"),
    ("Conservative_2025-09", f"{DOM}/Conservative_2025-09_test.jsonl"),
    ("technology_2025-09", f"{DOM}/technology_2025-09_test.jsonl"),
]

CELLS = [
    ("S", ("structural",), {}),
    ("S+L", ("structural", "lexical"), {}),
    ("S+N768", ("structural", "noise"), {"noise_dim": 768}),
    ("S+Es", ("structural", "semantic_stim"), {}),
    ("S+E", ("structural", "semantic"), {}),
]

# The contrasts that isolate semantics from mere width.
CONTRASTS = [("S+N768", "S+E"), ("S+N768", "S+Es"), ("S", "S+E")]


def load_tuned_params(path: str, tag: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[shift] {path} missing; using default LightGBM params")
        return {}
    blob = json.loads(p.read_text())
    if tag not in blob:
        print(f"[shift] tag '{tag}' not in {path}; using default params")
        return {}
    out = {k: v.get("params") or {} for k, v in blob[tag]["cells"].items()}
    print(f"[shift] loaded tuned params for {sorted(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tuned-json", default="data_analysis/ablation_results.json")
    ap.add_argument("--tuned-tag", default="tuned_all-MiniLM-L6-v2")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="data_analysis/ablation_shift.json")
    args = ap.parse_args()

    tuned = load_tuned_params(args.tuned_json, args.tuned_tag)
    train = read_jsonl(TRAIN)
    tests = {name: read_jsonl(path) for name, path in TEST_SETS}
    print(f"train={len(train)}  test sets={ {k: len(v) for k, v in tests.items()} }")

    embedder = SentenceEmbedder(args.embed_model)
    t0 = time.time()
    all_recs = train + [r for recs in tests.values() for r in recs]
    embedder.warm([t for r in all_recs for t in semantic_texts(r)])
    embedder.save()
    print(f"[embed] warm done in {time.time()-t0:.1f}s")

    results: dict[str, dict] = {}
    boots: dict[str, dict] = {}
    for cname, blocks, kw in CELLS:
        cfg = tuned.get(cname, {})
        model = FeatureAblation(blocks=blocks, embedder=embedder, **kw, **cfg)
        t0 = time.time()
        model.fit(train)
        print(f"[fit] {cname} params={cfg} ({time.time()-t0:.1f}s)")
        results[cname] = {}
        boots[cname] = {}
        for tname, recs in tests.items():
            out = model.predict_batch(recs)
            ps = np.asarray([o["score"] for o in out], dtype=float)
            pw = np.asarray([o["width"] for o in out], dtype=float)
            pc = np.asarray([o["controversiality"] for o in out], dtype=float)
            true = truths(recs)
            results[cname][tname] = {
                "n": len(recs),
                "score_rho": spearman(ps, true[0])["spearman_rho"],
                "width_rho": spearman(pw, true[1])["spearman_rho"],
                "contr_f1": f1_binary_macro(pc.astype(int).tolist(), true[2].astype(int).tolist()),
            }
            idx = bootstrap_indices(len(recs), args.n_boot, seed=0)
            boots[cname][tname] = boot_stat((ps, pw, pc), true, idx)
        print("   " + "  ".join(
            f"{t.split(' ')[0][:18]}={results[cname][t]['score_rho']:.3f}" for t in tests))

    deltas: dict[str, dict] = {}
    for ref, cmp_ in CONTRASTS:
        deltas[f"{cmp_} - {ref}"] = {}
        for tname in tests:
            d = {}
            for metric in ("score_rho", "width_rho"):
                diff = boots[cmp_][tname][metric] - boots[ref][tname][metric]
                lo, hi = ci(diff)
                d[metric] = {
                    "delta": float(results[cmp_][tname][metric] - results[ref][tname][metric]),
                    "ci95": [lo, hi],
                    "p_two_sided": float(2 * min((diff <= 0).mean(), (diff >= 0).mean())),
                }
            deltas[f"{cmp_} - {ref}"][tname] = d

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": {"embed_model": args.embed_model, "tuned_tag": args.tuned_tag,
                    "n_boot": args.n_boot},
         "cells": results, "contrasts": deltas}, indent=2))

    # ------------------------------------------------------------- reporting
    names = [n for n, _ in TEST_SETS]
    print("\n" + "=" * 118)
    print("SCORE rho BY TEST SET (fit once on politics 2025-12)")
    print("=" * 118)
    print(f"{'cell':8s}" + "".join(f"{n[:17]:>19s}" for n in names))
    for cname, _, _ in CELLS:
        print(f"{cname:8s}" + "".join(
            f"{results[cname][n]['score_rho']:>19.3f}" for n in names))

    print("\n" + "=" * 118)
    print("SEMANTICS MINUS DIMENSION-MATCHED NOISE (positive => real semantic transfer)")
    print("=" * 118)
    for key, per_test in deltas.items():
        print(f"\n{key}   [score rho delta, 95% CI, p]")
        for n in names:
            d = per_test[n]["score_rho"]
            flag = "  *" if d["p_two_sided"] < 0.05 else ""
            print(f"   {n:28s}{d['delta']:>+8.4f}  [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
                  f"  p={d['p_two_sided']:.3f}{flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
