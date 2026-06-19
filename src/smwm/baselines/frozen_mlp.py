"""Frozen-embedding + MLP head baseline.

Encodes (context + stimulus) with a FROZEN sentence-transformer, then trains a
small sklearn MLP head on top. Isolates "is a good frozen representation
enough?" from end-to-end fine-tuning (the encoder baseline).
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


def _record_text(record: dict, max_chars: int = 1200) -> str:
    ctx = get_context(record)
    stim = get_stimulus(record)
    ctx_text = " \n ".join((c.get("body") or "") for c in ctx)
    text = ctx_text + " [SEP] " + (stim.get("body") or "")
    # Cap length: stimulus (most informative) sits at the end, so keep the tail.
    return text[-max_chars:]


@register("frozen_mlp")
class FrozenEmbeddingMLP(Baseline):
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 64,
                 hidden: tuple[int, ...] = (256, 64), max_iter: int = 300, **kwargs):
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self.hidden = tuple(hidden)
        self.max_iter = int(max_iter)
        self._encoder = None
        self.score_reg = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=42)
        self.width_reg = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=42)
        self.contr_clf = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=self.max_iter, random_state=42)
        self._has_two_classes = True
        self._majority_contr = 0

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_id, device=self.device)
            # Short sequences keep CPU encoding fast (stimulus is short text).
            self._encoder.max_seq_length = 128

    def _embed(self, records: list[dict]) -> np.ndarray:
        self._load()
        texts = [_record_text(r) for r in records]
        return self._encoder.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )

    def fit(self, train_records: list[dict]) -> None:
        usable = [r for r in train_records if get_ground_truth(r)]
        X = self._embed(usable)
        ys, yw, yc = [], [], []
        for r in usable:
            g = get_ground_truth(r)
            ys.append(float(g.get("score", 0)))
            yw.append(float(g.get("width", 0)))
            yc.append(int(g.get("controversiality", 0)))
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
        x = self._embed([record])
        s = float(np.expm1(self.score_reg.predict(x)[0]))
        w = float(np.expm1(self.width_reg.predict(x)[0]))
        c = int(self.contr_clf.predict(x)[0]) if self._has_two_classes else self._majority_contr
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
