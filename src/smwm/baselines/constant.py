"""Constant predictors: predict the training-set mean/median/mode."""
from __future__ import annotations

from statistics import mean, median

from .base import Baseline, get_ground_truth
from .registry import register


@register("constant_mean")
class ConstantMean(Baseline):
    def __init__(self, **kwargs):
        self._score = 0.0
        self._width = 0.0
        self._contr = 0
        self._reply = ""

    def fit(self, train_records: list[dict]) -> None:
        scores, widths, contrs = [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            scores.append(float(g.get("score", 0)))
            widths.append(float(g.get("width", 0)))
            contrs.append(int(g.get("controversiality", 0)))
        self._score = mean(scores) if scores else 0.0
        self._width = mean(widths) if widths else 0.0
        # mode of binary -> majority
        self._contr = 1 if contrs and (sum(contrs) > len(contrs) / 2) else 0
        self._reply = ""

    def predict(self, record: dict) -> dict:
        return {
            "score": int(round(self._score)),
            "width": int(round(self._width)),
            "controversiality": self._contr,
            "reply_summary": self._reply,
        }


@register("constant_median")
class ConstantMedian(ConstantMean):
    def fit(self, train_records: list[dict]) -> None:
        scores, widths, contrs = [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            if not g:
                continue
            scores.append(float(g.get("score", 0)))
            widths.append(float(g.get("width", 0)))
            contrs.append(int(g.get("controversiality", 0)))
        self._score = median(scores) if scores else 0.0
        self._width = median(widths) if widths else 0.0
        self._contr = 1 if contrs and (sum(contrs) > len(contrs) / 2) else 0
        self._reply = ""
