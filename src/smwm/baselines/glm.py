"""Count-GLM baseline: Poisson / Tweedie regression for score & width.

A proper count model for heavy-tailed reply/score counts, giving a principled
mean predictor. Poisson is the classic choice; Tweedie (power in (1,2)) models
over-dispersion the way a Negative-Binomial does, without an extra dependency.
controversiality uses LogisticRegression.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor, TweedieRegressor
from sklearn.preprocessing import StandardScaler

from ._features import text_features
from .base import Baseline, get_ground_truth
from .registry import register


class _GLMBase(Baseline):
    family = "poisson"

    def __init__(self, family: str | None = None, tweedie_power: float = 1.5, **kwargs):
        self.family = family or self.family
        self.tweedie_power = float(tweedie_power)
        self.scaler = StandardScaler()
        self.contr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._has_two_classes = True
        self._majority_contr = 0
        self.score_reg = self._make()
        self.width_reg = self._make()

    def _make(self):
        if self.family == "tweedie":
            return TweedieRegressor(power=self.tweedie_power, link="log", max_iter=2000)
        return PoissonRegressor(max_iter=2000)

    def fit(self, train_records: list[dict]) -> None:
        X, ys, yw, yc = [], [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            X.append(text_features(r))
            ys.append(max(float(g.get("score", 0)), 0.0))
            yw.append(max(float(g.get("width", 0)), 0.0))
            yc.append(int(g.get("controversiality", 0)))
        X = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.score_reg.fit(X, np.asarray(ys))
        self.width_reg.fit(X, np.asarray(yw))
        yc = np.asarray(yc, dtype=int)
        if len(set(yc.tolist())) >= 2:
            self.contr_clf.fit(X, yc)
            self._has_two_classes = True
        else:
            self._has_two_classes = False
            self._majority_contr = int(yc[0]) if len(yc) else 0

    def predict(self, record: dict) -> dict:
        x = self.scaler.transform(np.asarray([text_features(record)], dtype=float))
        s = float(self.score_reg.predict(x)[0])
        w = float(self.width_reg.predict(x)[0])
        c = int(self.contr_clf.predict(x)[0]) if self._has_two_classes else self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }


@register("glm_poisson")
class GLMPoisson(_GLMBase):
    family = "poisson"


@register("glm_tweedie")
class GLMTweedie(_GLMBase):
    family = "tweedie"
