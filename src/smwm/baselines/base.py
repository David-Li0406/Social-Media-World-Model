"""Baseline ABC + helpers to access fields off a record."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class Baseline(ABC):
    """A baseline takes train_records, learns (or noops), then predicts."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, train_records: list[dict]) -> None: ...

    @abstractmethod
    def predict(self, record: dict) -> dict:
        """Return a dict with keys score, controversiality, width, reply_summary."""

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Predict a chunk at once. Default loops; override where it pays.

        It pays a lot for two families: generation LLMs, where batching is the
        difference between 4.7 s and ~0.6 s per record, and the sklearn/LightGBM
        baselines, whose per-call overhead dominates their actual work."""
        return [self.predict(r) for r in records]


def get_context(record: dict) -> list[dict]:
    """Recover the conversation context, preferring structured field over prompt."""
    if "context" in record and isinstance(record["context"], list):
        return record["context"]
    return []


def get_stimulus(record: dict) -> dict:
    if "stimulus" in record and isinstance(record["stimulus"], dict):
        return record["stimulus"]
    return {}


def get_ground_truth(record: dict) -> dict:
    if "ground_truth" in record and isinstance(record["ground_truth"], dict):
        return record["ground_truth"]
    if "completion" in record:
        try:
            return json.loads(record["completion"])
        except (TypeError, ValueError):
            pass
    return {}
