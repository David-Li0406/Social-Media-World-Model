"""Hawkes / self-exciting branching baseline for reply cascades.

A reply tree is a branching process: each comment spawns offspring (direct
replies) at a rate driven by its own "infectivity" (popularity) and decaying
with generation/depth. The expected offspring count (= width) of a node is
modelled as

    E[width] = exp( mu + a * log1p(parent_score) + b * depth + c * log1p(root_score) )

which is the Poisson MLE of a depth-decayed self-exciting branching kernel.
Fit with sklearn's PoissonRegressor on those self-exciting drivers only (no
lexical features — that's the point of the cascade model). score and
controversiality are auxiliary here; we predict them with small structural
models so the baseline still emits the full 4-field dict.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from ._features import safe_float, safe_int
from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


def _cascade_drivers(record: dict) -> list[float]:
    ctx = get_context(record)
    stim = get_stimulus(record)
    parent = ctx[-1] if ctx else {}
    root = ctx[0] if ctx else {}
    depth = len(ctx)
    return [
        np.log1p(max(safe_float(parent.get("score", 0.0)), 0.0)),  # infectivity
        float(depth),                                               # generation decay
        np.log1p(max(safe_float(root.get("score", 0.0)), 0.0)),    # cascade size proxy
        float(len(stim.get("body", "") or "")),                    # mild content proxy
    ]


@register("hawkes")
class HawkesCascade(Baseline):
    def __init__(self, **kwargs):
        self.width_reg = PoissonRegressor(max_iter=2000)
        self.score_reg = PoissonRegressor(max_iter=2000)
        self.contr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._has_two_classes = True
        self._majority_contr = 0

    def fit(self, train_records: list[dict]) -> None:
        X, yw, ys, yc = [], [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            X.append(_cascade_drivers(r))
            yw.append(max(float(g.get("width", 0)), 0.0))
            ys.append(max(float(g.get("score", 0)), 0.0))
            yc.append(int(g.get("controversiality", 0)))
        X = np.asarray(X, dtype=float)
        self.width_reg.fit(X, np.asarray(yw))
        self.score_reg.fit(X, np.asarray(ys))
        yc = np.asarray(yc, dtype=int)
        if len(set(yc.tolist())) >= 2:
            self.contr_clf.fit(X, yc)
            self._has_two_classes = True
        else:
            self._has_two_classes = False
            self._majority_contr = int(yc[0]) if len(yc) else 0

    def predict(self, record: dict) -> dict:
        x = np.asarray([_cascade_drivers(record)], dtype=float)
        w = float(self.width_reg.predict(x)[0])
        s = float(self.score_reg.predict(x)[0])
        c = int(self.contr_clf.predict(x)[0]) if self._has_two_classes else self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
