"""Feature-based regressor: GBDT on metadata + simple text features.

Targets: score (log1p) and width (log1p) via GBDT; controversiality via
LogisticRegression. reply_summary is left empty (this baseline doesn't
produce text).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


_URL_TOKENS = ("http://", "https://", "www.")


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_features(record: dict) -> list[float]:
    stim = get_stimulus(record)
    ctx = get_context(record)
    body = stim.get("body", "") or ""
    parent = ctx[-1] if ctx else {}
    root = ctx[0] if ctx else {}

    n_chars = len(body)
    n_words = len(body.split())
    n_upper = sum(1 for c in body if c.isupper())
    n_punct = sum(1 for c in body if c in "!?.,;:")
    has_url = float(any(t in body for t in _URL_TOKENS))
    upper_ratio = n_upper / n_chars if n_chars else 0.0

    ts = _safe_int(stim.get("created_utc", 0))
    if ts > 0:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        dow = dt.weekday()
    else:
        hour, dow = 0, 0

    depth = len(ctx)
    parent_score = _safe_float(parent.get("score", 0.0))
    parent_contr = _safe_float(parent.get("controversiality", 0.0))
    parent_len = len((parent.get("body") or ""))
    root_score = _safe_float(root.get("score", 0.0))
    root_len = len((root.get("body") or ""))

    return [
        n_chars,
        n_words,
        n_punct,
        upper_ratio,
        has_url,
        float(hour),
        float(dow),
        float(depth),
        np.log1p(max(parent_score, 0.0)),
        parent_contr,
        float(parent_len),
        np.log1p(max(root_score, 0.0)),
        float(root_len),
    ]


@register("feature_gbdt")
class FeatureGBDT(Baseline):
    def __init__(self, **kwargs):
        self.score_reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=42
        )
        self.width_reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=42
        )
        self.contr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.scaler = StandardScaler()
        self._has_pos_class = True
        self._majority_contr = 0

    def fit(self, train_records: list[dict]) -> None:
        X, y_score, y_width, y_contr = [], [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            X.append(_extract_features(r))
            y_score.append(float(g.get("score", 0)))
            y_width.append(float(g.get("width", 0)))
            y_contr.append(int(g.get("controversiality", 0)))
        X = np.asarray(X, dtype=float)
        y_score = np.log1p(np.maximum(np.asarray(y_score), 0.0))
        y_width = np.log1p(np.maximum(np.asarray(y_width), 0.0))
        y_contr = np.asarray(y_contr, dtype=int)

        Xs = self.scaler.fit_transform(X)
        self.score_reg.fit(Xs, y_score)
        self.width_reg.fit(Xs, y_width)
        if len(set(y_contr.tolist())) >= 2:
            self.contr_clf.fit(Xs, y_contr)
            self._has_pos_class = True
        else:
            self._has_pos_class = False
            self._majority_contr = int(y_contr[0]) if len(y_contr) else 0

    def predict(self, record: dict) -> dict:
        x = np.asarray([_extract_features(record)], dtype=float)
        xs = self.scaler.transform(x)
        s = float(np.expm1(self.score_reg.predict(xs)[0]))
        w = float(np.expm1(self.width_reg.predict(xs)[0]))
        if self._has_pos_class:
            c = int(self.contr_clf.predict(xs)[0])
        else:
            c = self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
