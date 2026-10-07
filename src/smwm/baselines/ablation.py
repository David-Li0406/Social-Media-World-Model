"""Feature-block ablation: one model class, swappable feature sets.

The benchmark's headline claim is that conversation *structure* carries most of
the signal (`structural_prior` rho=0.750 vs the best model's 0.760). That
comparison is confounded. `feature_gbdt`'s "text" features are only shallow
lexical statistics (length, punctuation, caps, url) -- no semantics at all --
while the only semantic models (`encoder`, `retrieval_sbert`) carry no
structural features. No model in the benchmark has ever combined structure with
real semantics, so "text adds nothing over structure" has never been tested.

This baseline changes exactly one thing at a time: identical LightGBM model,
identical log1p targets, only the feature blocks differ.

  structural : graph/time features (depth, parent & root score, hour, weekday)
  lexical    : shallow stimulus-text statistics (length, punctuation, caps, url)
  semantic   : frozen sentence-transformer embeddings of the stimulus body and
               of the parent body -- what was actually *said*

Embeddings are memoized in-process and cached on disk (keyed by model id +
text hash), so the six ablation cells pay the GPU cost once.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ._features import structural_features
from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register

_URL_TOKENS = ("http://", "https://", "www.")

BLOCKS = (
    "structural",
    "lexical",
    "semantic",
    # Diagnostic blocks, used to falsify / decompose a semantic gain:
    "semantic_stim",    # stimulus embedding only
    "semantic_parent",  # parent embedding only
    "noise",            # same width as `semantic`, pure per-record noise
)

# Width of the noise control. Must match the `semantic` block (2 x embed dim)
# so that "more columns" is held constant and only *informativeness* varies.
NOISE_DIM = 768


def noise_features(record: dict, dim: int = NOISE_DIM) -> np.ndarray:  # noqa: D401
    """Deterministic per-record noise: a dimensionality control.

    If a 768-column semantic block beats a 7-column structural one merely
    because LightGBM enjoys more candidate splits, then 768 columns of noise
    should help too. It must not.
    """
    stim = get_stimulus(record)
    key = str(stim.get("id") or id(record))
    seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def lexical_features(record: dict) -> list[float]:
    """Shallow text statistics only -- deliberately no semantics.

    This is the "text" that feature_gbdt actually sees, isolated from the
    structural columns it is normally bundled with.
    """
    stim = get_stimulus(record)
    ctx = get_context(record)
    parent = ctx[-1] if ctx else {}
    root = ctx[0] if ctx else {}
    body = stim.get("body", "") or ""

    n_chars = len(body)
    n_upper = sum(1 for c in body if c.isupper())
    return [
        float(n_chars),
        float(len(body.split())),
        float(sum(1 for c in body if c in "!?.,;:")),
        (n_upper / n_chars) if n_chars else 0.0,
        float(any(t in body for t in _URL_TOKENS)),
        float(len(parent.get("body") or "")),
        float(len(root.get("body") or "")),
    ]


def semantic_texts(record: dict) -> tuple[str, str]:
    """The two bodies whose meaning could plausibly drive engagement."""
    stim = get_stimulus(record)
    ctx = get_context(record)
    parent = ctx[-1] if ctx else {}
    return (stim.get("body", "") or "", parent.get("body", "") or "")


class SentenceEmbedder:
    """Frozen sentence-transformer with in-process memo + on-disk cache.

    The ablation fits six models over the same records; embedding is by far the
    most expensive step, so it must be paid exactly once.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        batch_size: int = 256,
        max_chars: int = 1000,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self.max_chars = int(max_chars)
        self.device = device
        self.cache_dir = Path(
            cache_dir
            or os.getenv("SMWM_EMB_CACHE")
            or (Path(os.getenv("SMWM_SCRATCH", os.path.expanduser("~/smwm"))) / "emb_cache")
        )
        self._model: Any = None
        self._memo: dict[str, np.ndarray] = {}
        self._dim: int | None = None
        self._loaded_disk = False

    # ------------------------------------------------------------ internals
    def _cache_path(self) -> Path:
        tag = self.model_id.replace("/", "__")
        return self.cache_dir / f"{tag}.npz"

    def _key(self, text: str) -> str:
        return hashlib.sha1(text[: self.max_chars].encode("utf-8")).hexdigest()

    def _load_disk(self) -> None:
        if self._loaded_disk:
            return
        self._loaded_disk = True
        path = self._cache_path()
        if not path.exists():
            return
        with np.load(path, allow_pickle=False) as z:
            keys, vecs = z["keys"], z["vecs"]
        self._memo.update({str(k): v for k, v in zip(keys, vecs)})
        if len(vecs):
            self._dim = int(vecs.shape[1])
        print(f"[embed] loaded {len(keys)} cached vectors from {path}")

    def save(self) -> None:
        if not self._memo:
            return
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.asarray(list(self._memo), dtype=object).astype("U40")
        vecs = np.stack([self._memo[k] for k in self._memo]).astype(np.float32)
        np.savez(path, keys=keys, vecs=vecs)
        print(f"[embed] saved {len(keys)} vectors to {path}")

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embed] loading {self.model_id} on {self.device}")
        self._model = SentenceTransformer(self.model_id, device=self.device)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load_model()
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._dim

    # ---------------------------------------------------------------- public
    def warm(self, texts: Iterable[str]) -> None:
        """Batch-embed every text not already known. One GPU pass."""
        self._load_disk()
        todo: dict[str, str] = {}
        for t in texts:
            k = self._key(t)
            if k not in self._memo and k not in todo:
                todo[k] = t[: self.max_chars]
        if not todo:
            return
        self._load_model()
        keys = list(todo)
        vecs = self._model.encode(
            [todo[k] for k in keys],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for k, v in zip(keys, vecs):
            self._memo[k] = v.astype(np.float32)
        self._dim = int(vecs.shape[1])

    def encode_one(self, text: str) -> np.ndarray:
        self._load_disk()
        k = self._key(text)
        if k not in self._memo:
            self.warm([text])
        return self._memo[k]


@register("feature_ablation")
class FeatureAblation(Baseline):
    """LightGBM over a configurable subset of {structural, lexical, semantic}."""

    def __init__(
        self,
        blocks: Sequence[str] = ("structural", "lexical"),
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_cache: str | None = None,
        pca_dim: int = 0,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        feature_fraction: float = 1.0,
        min_child_samples: int = 20,
        noise_dim: int = NOISE_DIM,
        alpha: float = 0.5,
        seed: int = 42,
        embedder: SentenceEmbedder | None = None,
        **kwargs,
    ):
        import lightgbm as lgb

        self.lgb = lgb
        self.blocks = tuple(blocks)
        unknown = set(self.blocks) - set(BLOCKS)
        if unknown:
            raise ValueError(f"unknown feature blocks {sorted(unknown)}; valid: {BLOCKS}")
        self.pca_dim = int(pca_dim)
        self.noise_dim = int(noise_dim)
        self.params = dict(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            colsample_bytree=float(feature_fraction),
            min_child_samples=int(min_child_samples),
            random_state=int(seed),
            verbose=-1,
        )
        self.alpha = float(alpha)
        self._embedder = embedder
        self._embed_model = embed_model
        self._embed_cache = embed_cache
        self._pca = None
        self.score_reg = None
        self.width_reg = None
        self.contr_clf = None
        self._has_two_classes = True
        self._majority_contr = 0

    # ------------------------------------------------------------- features
    @property
    def embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder(self._embed_model, cache_dir=self._embed_cache)
        return self._embedder

    @property
    def _needs_embeddings(self) -> bool:
        return any(b.startswith("semantic") for b in self.blocks)

    def warm(self, records: Sequence[dict]) -> None:
        """Pre-embed a whole split in one GPU pass (no-op without semantics)."""
        if not self._needs_embeddings:
            return
        texts: list[str] = []
        for r in records:
            texts.extend(semantic_texts(r))
        self.embedder.warm(texts)

    def _raw_row(self, record: dict) -> np.ndarray:
        parts: list[np.ndarray] = []
        if "structural" in self.blocks:
            parts.append(np.asarray(structural_features(record), dtype=np.float32))
        if "lexical" in self.blocks:
            parts.append(np.asarray(lexical_features(record), dtype=np.float32))
        if self._needs_embeddings:
            stim_text, parent_text = semantic_texts(record)
            if "semantic" in self.blocks or "semantic_stim" in self.blocks:
                parts.append(self.embedder.encode_one(stim_text))
            if "semantic" in self.blocks or "semantic_parent" in self.blocks:
                parts.append(self.embedder.encode_one(parent_text))
        if "noise" in self.blocks:
            parts.append(noise_features(record, self.noise_dim))
        return np.concatenate(parts)

    def _matrix(self, records: Sequence[dict]) -> np.ndarray:
        self.warm(records)
        return np.stack([self._raw_row(r) for r in records])

    def _project(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Optional PCA on the semantic block only; structural/lexical stay raw."""
        if self.pca_dim <= 0 or not self._needs_embeddings:
            return X
        n_dense = 0
        if "structural" in self.blocks:
            n_dense += 7
        if "lexical" in self.blocks:
            n_dense += 7
        dense, emb = X[:, :n_dense], X[:, n_dense:]
        if fit:
            from sklearn.decomposition import PCA

            self._pca = PCA(n_components=min(self.pca_dim, emb.shape[1], emb.shape[0]),
                            random_state=42)
            emb = self._pca.fit_transform(emb)
        else:
            emb = self._pca.transform(emb)
        return np.hstack([dense, emb])

    # ------------------------------------------------------------------ fit
    def fit(self, train_records: list[dict]) -> None:
        usable = [r for r in train_records if get_ground_truth(r)]
        X = self._project(self._matrix(usable), fit=True)

        ys, yw, yc = [], [], []
        for r in usable:
            g = get_ground_truth(r)
            ys.append(float(g.get("score", 0)))
            yw.append(float(g.get("width", 0)))
            yc.append(int(g.get("controversiality", 0)))
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

    # -------------------------------------------------------------- predict
    def predict_batch(self, records: Sequence[dict]) -> list[dict]:
        """Vectorized prediction -- the ablation scores 1900 records at a time."""
        X = self._project(self._matrix(records), fit=False)
        s = np.expm1(self.score_reg.predict(X))
        w = np.expm1(self.width_reg.predict(X))
        if self._has_two_classes:
            c = self.contr_clf.predict(X).astype(int)
        else:
            c = np.full(len(records), self._majority_contr, dtype=int)
        return [
            {
                "score": int(round(max(float(si), 0))),
                "width": int(round(max(float(wi), 0))),
                "controversiality": int(ci),
                "reply_summary": "",
            }
            for si, wi, ci in zip(s, w, c)
        ]

    def predict(self, record: dict) -> dict:
        return self.predict_batch([record])[0]
