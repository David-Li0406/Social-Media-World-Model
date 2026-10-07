"""Ranking-based metrics that ignore absolute error and only assess order.

The user's example: if truth is (10, 20) and prediction is (100, 200), the
pairwise order is preserved so the model gets full credit here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def spearman(preds: Sequence[float], truths: Sequence[float]) -> dict:
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truths, dtype=float)
    if len(p) < 2 or np.all(p == p[0]) or np.all(t == t[0]):
        return {"spearman_rho": float("nan"), "spearman_p": float("nan")}
    rho, pval = spearmanr(p, t)
    return {"spearman_rho": float(rho), "spearman_p": float(pval)}


def pairwise_accuracy(preds: Sequence[float], truths: Sequence[float],
                      max_n: int = 5000, seed: int = 0) -> float:
    """Fraction of pairs (i,j) where sign(pred_i-pred_j) == sign(true_i-true_j).

    Tied truths are excluded from the denominator since their "correct order" is
    undefined; tied predictions on non-tied truths count as wrong.

    This is O(n^2) in MEMORY: it materializes two n x n sign matrices plus the
    upper-triangle index arrays. That is 55MB at the n=1900 this was written
    for, but 80GB at n=60000, which OOM-kills the process. Above `max_n` the
    estimate is therefore taken on a deterministic random subsample -- 5000
    records is still 12.5M pairs, so the sampling error on the fraction is
    ~1e-4, far below any effect this project cares about. Behaviour for
    n <= max_n is unchanged, so previously reported numbers still reproduce.
    """
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truths, dtype=float)
    n = len(p)
    if n < 2:
        return float("nan")
    if max_n and n > max_n:
        idx = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
        p, t = p[idx], t[idx]
        n = max_n
    # Vectorize over upper triangle.
    dp = np.sign(p[:, None] - p[None, :])
    dt = np.sign(t[:, None] - t[None, :])
    iu = np.triu_indices(n, k=1)
    dp_u, dt_u = dp[iu], dt[iu]
    mask = dt_u != 0
    if mask.sum() == 0:
        return float("nan")
    correct = (dp_u[mask] == dt_u[mask]).sum()
    return float(correct / mask.sum())
