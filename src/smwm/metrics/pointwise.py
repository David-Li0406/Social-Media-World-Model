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
    tp = fp = fn = tn = 0
    for p, t in zip(preds, truths):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1

    def _f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
