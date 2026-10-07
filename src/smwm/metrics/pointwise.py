"""Point-wise metrics: MSE, MAE, macro-F1.

Ported from metrics.py.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mse(preds: Sequence[float], truths: Sequence[float]) -> float:
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truths, dtype=float)
    return float(np.mean((p - t) ** 2))


def mae(preds: Sequence[float], truths: Sequence[float]) -> float:
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truths, dtype=float)
    return float(np.mean(np.abs(p - t)))


def f1_binary_macro(preds: Sequence[int], truths: Sequence[int]) -> float:
    # Vectorized: the bootstrap calls this thousands of times on 60k-element
    # vectors, where the original Python loop cost minutes. Counts (and hence
    # the returned value) are identical.
    p = np.asarray(preds)
    t = np.asarray(truths)
    pe1, te1 = (p == 1), (t == 1)
    tp = int(np.count_nonzero(pe1 & te1))
    fp = int(np.count_nonzero(pe1 & ~te1))
    fn = int(np.count_nonzero(~pe1 & te1))
    tn = int(len(p) - tp - fp - fn)

    def _f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
