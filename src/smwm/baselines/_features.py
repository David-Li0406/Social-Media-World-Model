"""Shared feature extraction for tabular / structural baselines.

Two feature sets:
  - structural_features: graph + metadata only (NO text content). Used by the
    structural_prior baseline to isolate "how much does text add?".
  - text_features: structural + cheap lexical signals (length, punctuation,
    caps, url). Used by feature_gbdt / glm / quantile_gbm.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .base import get_context, get_stimulus

_URL_TOKENS = ("http://", "https://", "www.")


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _time_feats(ts: int) -> tuple[float, float]:
    if ts > 0:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return float(dt.hour), float(dt.weekday())
    return 0.0, 0.0


def structural_features(record: dict) -> list[float]:
    """Graph + metadata only, no text content."""
    stim = get_stimulus(record)
    ctx = get_context(record)
    parent = ctx[-1] if ctx else {}
    root = ctx[0] if ctx else {}

    hour, dow = _time_feats(safe_int(stim.get("created_utc", 0)))
    depth = len(ctx)
    return [
        float(depth),
        hour,
        dow,
        np.log1p(max(safe_float(parent.get("score", 0.0)), 0.0)),
        safe_float(parent.get("controversiality", 0.0)),
        np.log1p(max(safe_float(root.get("score", 0.0)), 0.0)),
        float(len(ctx)),
    ]


def text_features(record: dict) -> list[float]:
    """Structural features + cheap lexical signals from the stimulus body."""
    stim = get_stimulus(record)
    ctx = get_context(record)
    parent = ctx[-1] if ctx else {}
    root = ctx[0] if ctx else {}
    body = stim.get("body", "") or ""

    n_chars = len(body)
    n_words = len(body.split())
    n_upper = sum(1 for c in body if c.isupper())
    n_punct = sum(1 for c in body if c in "!?.,;:")
    has_url = float(any(t in body for t in _URL_TOKENS))
    upper_ratio = n_upper / n_chars if n_chars else 0.0
    hour, dow = _time_feats(safe_int(stim.get("created_utc", 0)))

    return [
        float(n_chars),
        float(n_words),
        float(n_punct),
        upper_ratio,
        has_url,
        hour,
        dow,
        float(len(ctx)),
        np.log1p(max(safe_float(parent.get("score", 0.0)), 0.0)),
        safe_float(parent.get("controversiality", 0.0)),
        float(len(parent.get("body") or "")),
        np.log1p(max(safe_float(root.get("score", 0.0)), 0.0)),
        float(len(root.get("body") or "")),
    ]
