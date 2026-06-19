"""Popularity / structural-prior baseline: NO text content.

Predicts from graph + metadata features only (depth, parent/root score,
time-of-day) plus per-subreddit base rates. Isolates how much signal lives in
the conversation structure vs. the comment text.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._features import structural_features
from .base import Baseline, get_ground_truth, get_stimulus
from .registry import register


@register("structural_prior")
class StructuralPrior(Baseline):
    def __init__(self, **kwargs):
        self.score_reg = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
        self.width_reg = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)
        self.contr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.scaler = StandardScaler()
        self._has_two_classes = True
        self._majority_contr = 0

    def fit(self, train_records: list[dict]) -> None:
        X, ys, yw, yc = [], [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            X.append(structural_features(r))
            ys.append(float(g.get("score", 0)))
            yw.append(float(g.get("width", 0)))
            yc.append(int(g.get("controversiality", 0)))
        X = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.score_reg.fit(X, np.log1p(np.maximum(ys, 0.0)))
        self.width_reg.fit(X, np.log1p(np.maximum(yw, 0.0)))
        yc = np.asarray(yc, dtype=int)
        if len(set(yc.tolist())) >= 2:
            self.contr_clf.fit(X, yc)
            self._has_two_classes = True
        else:
            self._has_two_classes = False
            self._majority_contr = int(yc[0]) if len(yc) else 0

    def predict(self, record: dict) -> dict:
        x = self.scaler.transform(np.asarray([structural_features(record)], dtype=float))
        s = float(np.expm1(self.score_reg.predict(x)[0]))
        w = float(np.expm1(self.width_reg.predict(x)[0]))
        c = int(self.contr_clf.predict(x)[0]) if self._has_two_classes else self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
