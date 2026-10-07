"""Why do 768 columns of pure noise raise Spearman rho?

Hypothesis: it is not regularization, it is TIE-BREAKING. Every tabular
baseline in this repo rounds its prediction to an int before scoring
(`int(round(max(s, 0)))`). A 7-feature model over coarse, largely integer
structural columns emits the same few values over and over, and Spearman's
tie correction drags rho down. Bolt on 768 noise columns and the trees can
split arbitrarily finely, so near-duplicate records get distinct predictions
-- ties dissolve and rho rises without a single bit of new information.

If that is the mechanism, then on the SAME fitted models:
  * rho computed on the raw (unrounded) predictions should be much closer
    between S and S+N than the rounded rho is, and
  * the number of distinct predicted values should be far lower for S.

Usage:
    PYTHONPATH=src python3 scripts/diagnose_ties.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("SMWM_SCRATCH", "/scratch/daweili5/smwm")
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")

from smwm.baselines.ablation import FeatureAblation, SentenceEmbedder, semantic_texts  # noqa: E402
from smwm.data.io import read_jsonl  # noqa: E402
from smwm.metrics.ranking import spearman  # noqa: E402

TRAIN = "train_test/RC_2025-12_politics_width_chain01_train.jsonl"
TEST = "train_test/RC_2025-12_politics_width_chain01_test.jsonl"

CELLS = [
    ("S", ("structural",), {}),
    ("S+L", ("structural", "lexical"), {}),
    ("S+N768", ("structural", "noise"), {"noise_dim": 768}),
    ("S+E", ("structural", "semantic"), {}),
]


def raw_predictions(model, records):
    """The continuous predictions, before the int() rounding the repo applies."""
    X = model._project(model._matrix(records), fit=False)
    return np.expm1(model.score_reg.predict(X)), np.expm1(model.width_reg.predict(X))


def main():
    train, test = read_jsonl(TRAIN), read_jsonl(TEST)
    true_s, true_w = [], []
    for r in test:
        g = r.get("ground_truth") or json.loads(r["completion"])
        true_s.append(float(g["score"]))
        true_w.append(float(g["width"]))
    true_s, true_w = np.asarray(true_s), np.asarray(true_w)

    embedder = SentenceEmbedder()
    embedder.warm([t for r in train + test for t in semantic_texts(r)])

    print(f"{'cell':9s}{'uniq_s':>8}{'uniq_w':>8}"
          f"{'rho_s(int)':>12}{'rho_s(raw)':>12}{'rho_w(int)':>12}{'rho_w(raw)':>12}")
    print("-" * 73)
    rows = {}
    for name, blocks, kw in CELLS:
        m = FeatureAblation(blocks=blocks, embedder=embedder, **kw)
        m.fit(train)
        out = m.predict_batch(test)
        pi_s = np.asarray([o["score"] for o in out], dtype=float)
        pi_w = np.asarray([o["width"] for o in out], dtype=float)
        pr_s, pr_w = raw_predictions(m, test)

        row = {
            "unique_rounded_score": int(len(np.unique(pi_s))),
            "unique_rounded_width": int(len(np.unique(pi_w))),
            "unique_raw_score": int(len(np.unique(pr_s))),
            "rho_score_rounded": spearman(pi_s, true_s)["spearman_rho"],
            "rho_score_raw": spearman(pr_s, true_s)["spearman_rho"],
            "rho_width_rounded": spearman(pi_w, true_w)["spearman_rho"],
            "rho_width_raw": spearman(pr_w, true_w)["spearman_rho"],
        }
        rows[name] = row
        print(f"{name:9s}{row['unique_rounded_score']:>8}{row['unique_rounded_width']:>8}"
              f"{row['rho_score_rounded']:>12.4f}{row['rho_score_raw']:>12.4f}"
              f"{row['rho_width_rounded']:>12.4f}{row['rho_width_raw']:>12.4f}")

    print(f"\nn_test={len(test)}   (uniq_* = distinct predicted values after int rounding)")
    print("\nGap S+N768 - S:")
    for tgt in ("score", "width"):
        r_int = rows["S+N768"][f"rho_{tgt}_rounded"] - rows["S"][f"rho_{tgt}_rounded"]
        r_raw = rows["S+N768"][f"rho_{tgt}_raw"] - rows["S"][f"rho_{tgt}_raw"]
        print(f"  {tgt:6s} rounded {r_int:+.4f}   raw {r_raw:+.4f}   "
              f"=> {100*(1 - abs(r_raw)/abs(r_int)) if r_int else float('nan'):.0f}% of the gap "
              f"is rounding artifact")

    Path("data_analysis").mkdir(exist_ok=True)
    Path("data_analysis/tie_diagnosis.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote data_analysis/tie_diagnosis.json")


if __name__ == "__main__":
    main()
