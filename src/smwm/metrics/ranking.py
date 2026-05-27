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


def pairwise_accuracy(preds: Sequence[float], truths: Sequence[float]) -> float:
    """Fraction of pairs (i,j) where sign(pred_i-pred_j) == sign(true_i-true_j).

    Tied truths are excluded from the denominator since their "correct order" is
    undefined; tied predictions on non-tied truths count as wrong.
    """
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truths, dtype=float)
    n = len(p)
    if n < 2:
        return float("nan")
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
