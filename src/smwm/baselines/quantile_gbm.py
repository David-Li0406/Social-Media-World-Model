"""Quantile gradient boosting (LightGBM) baseline.

LightGBM with quantile (pinball) loss at the median for score & width on
log1p targets, plus a LightGBM classifier for controversiality. Stronger
tabular model than sklearn's GradientBoosting and naturally produces
prediction intervals (extra quantiles trained but not required by the metrics).
"""
from __future__ import annotations

import numpy as np

from ._features import text_features
from .base import Baseline, get_ground_truth
from .registry import register


@register("quantile_gbm")
class QuantileGBM(Baseline):
    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 31, alpha: float = 0.5, **kwargs):
        import lightgbm as lgb  # noqa: F401 (ensure available at construct time)

        self.lgb = lgb
        self.params = dict(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            random_state=42,
            verbose=-1,
        )
        self.alpha = float(alpha)
        self.score_reg = None
        self.width_reg = None
        self.contr_clf = None
        self._has_two_classes = True
        self._majority_contr = 0

    def fit(self, train_records: list[dict]) -> None:
        X, ys, yw, yc = [], [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            X.append(text_features(r))
            ys.append(float(g.get("score", 0)))
            yw.append(float(g.get("width", 0)))
            yc.append(int(g.get("controversiality", 0)))
        X = np.asarray(X, dtype=float)
        ys = np.log1p(np.maximum(np.asarray(ys), 0.0))
        yw = np.log1p(np.maximum(np.asarray(yw), 0.0))
        yc = np.asarray(yc, dtype=int)

        self.score_reg = self.lgb.LGBMRegressor(objective="quantile", alpha=self.alpha, **self.params)
        self.width_reg = self.lgb.LGBMRegressor(objective="quantile", alpha=self.alpha, **self.params)
        self.score_reg.fit(X, ys)
        self.width_reg.fit(X, yw)
        if len(set(yc.tolist())) >= 2:
            self.contr_clf = self.lgb.LGBMClassifier(class_weight="balanced", **self.params)
            self.contr_clf.fit(X, yc)
            self._has_two_classes = True
        else:
            self._has_two_classes = False
            self._majority_contr = int(yc[0]) if len(yc) else 0

    def predict(self, record: dict) -> dict:
        x = np.asarray([text_features(record)], dtype=float)
        s = float(np.expm1(self.score_reg.predict(x)[0]))
        w = float(np.expm1(self.width_reg.predict(x)[0]))
        c = int(self.contr_clf.predict(x)[0]) if self._has_two_classes else self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
