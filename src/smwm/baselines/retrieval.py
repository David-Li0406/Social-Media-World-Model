"""Retrieval-based baselines: TF-IDF and sentence-embedding kNN.

For each test record, embed (context joined + stimulus.body), retrieve top-k
neighbors from train, and predict by similarity-weighted aggregation.
"""
from __future__ import annotations

import numpy as np

from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


def _record_text(record: dict) -> str:
    ctx = get_context(record)
    stim = get_stimulus(record)
    ctx_text = " \n ".join((c.get("body") or "") for c in ctx)
    stim_text = stim.get("body") or ""
    return ctx_text + " [SEP] " + stim_text


class _RetrievalCore(Baseline):
    """Shared aggregation logic for retrieval baselines."""

    def __init__(self, k: int = 5, **kwargs):
        self.k = int(k)
        self._train_scores: np.ndarray | None = None
        self._train_widths: np.ndarray | None = None
        self._train_contrs: np.ndarray | None = None
        self._train_replies: list[str] = []

    def _store_train_targets(self, train_records: list[dict]) -> None:
        s, w, c, r = [], [], [], []
        for rec in train_records:
            g = get_ground_truth(rec)
            s.append(float(g.get("score", 0)))
            w.append(float(g.get("width", 0)))
            c.append(int(g.get("controversiality", 0)))
            r.append(str(g.get("reply_summary", "")))
        self._train_scores = np.asarray(s, dtype=float)
        self._train_widths = np.asarray(w, dtype=float)
        self._train_contrs = np.asarray(c, dtype=int)
        self._train_replies = r

    def _aggregate(self, top_idx: np.ndarray, sims: np.ndarray) -> dict:
        weights = np.maximum(sims, 1e-8)
        w_sum = weights.sum()
        s = float((self._train_scores[top_idx] * weights).sum() / w_sum)
        w = float((self._train_widths[top_idx] * weights).sum() / w_sum)
        c_vals = self._train_contrs[top_idx]
        # Weighted majority for binary
        pos = float((weights * (c_vals == 1)).sum())
        neg = float((weights * (c_vals == 0)).sum())
        c = 1 if pos >= neg else 0
        reply = self._train_replies[int(top_idx[int(np.argmax(weights))])]
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": reply,
        }


@register("retrieval_tfidf")
class RetrievalTFIDF(_RetrievalCore):
    def __init__(self, k: int = 5, max_features: int = 30000, **kwargs):
        super().__init__(k=k)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(max_features=int(max_features), ngram_range=(1, 2))
        self._train_X = None

    def fit(self, train_records: list[dict]) -> None:
        texts = [_record_text(r) for r in train_records]
        self._train_X = self.vec.fit_transform(texts)
        self._store_train_targets(train_records)

    def predict(self, record: dict) -> dict:
        from sklearn.metrics.pairwise import cosine_similarity

        q = self.vec.transform([_record_text(record)])
        sims = cosine_similarity(q, self._train_X).ravel()
        k = min(self.k, sims.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        return self._aggregate(top_idx, sims[top_idx])


@register("retrieval_sbert")
class RetrievalSBERT(_RetrievalCore):
    def __init__(
        self,
        k: int = 5,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(k=k)
        self.model_id = model_id
        self.device = device
        self.batch_size = int(batch_size)
        self._encoder = None
        self._train_emb: np.ndarray | None = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_id, device=self.device)

    def fit(self, train_records: list[dict]) -> None:
        self._load()
        texts = [_record_text(r) for r in train_records]
        emb = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._train_emb = emb
        self._store_train_targets(train_records)

    def predict(self, record: dict) -> dict:
        self._load()
        q = self._encoder.encode(
            [_record_text(record)],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sims = (self._train_emb @ q.T).ravel()
        k = min(self.k, sims.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        return self._aggregate(top_idx, sims[top_idx])
