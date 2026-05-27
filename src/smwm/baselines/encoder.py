"""Encoder-based regressor: small BERT/DeBERTa with multi-head outputs.

Heads:
  - score regression (target = log1p(score))
  - width regression (target = log1p(width))
  - controversiality binary classification

reply_summary is left empty (text generation is out of scope for this baseline).

This module imports torch/transformers lazily so the rest of the package stays
importable on CPU-only / minimal-dep environments.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


def _record_text(record: dict) -> str:
    ctx = get_context(record)
    stim = get_stimulus(record)
    ctx_text = " ".join((c.get("body") or "") for c in ctx)
    stim_text = stim.get("body") or ""
    return (ctx_text + " [SEP] " + stim_text).strip()


@register("encoder")
class EncoderBaseline(Baseline):
    def __init__(
        self,
        model_id: str = "distilbert-base-uncased",
        max_length: int = 256,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        device: str | None = None,
        score_weight: float = 1.0,
        width_weight: float = 1.0,
        contr_weight: float = 1.0,
        **kwargs,
    ):
        self.model_id = model_id
        self.max_length = int(max_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device
        self.score_weight = float(score_weight)
        self.width_weight = float(width_weight)
        self.contr_weight = float(contr_weight)
        self._tokenizer = None
        self._model: Any = None

    def _build(self):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        class MultiHeadEncoder(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(base)
                h = self.backbone.config.hidden_size
                self.score_head = nn.Linear(h, 1)
                self.width_head = nn.Linear(h, 1)
                self.contr_head = nn.Linear(h, 2)

            def forward(self, input_ids, attention_mask):
                out = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # mean-pool over tokens (mask-aware)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                return (
                    self.score_head(pooled).squeeze(-1),
                    self.width_head(pooled).squeeze(-1),
                    self.contr_head(pooled),
                )

        self._model = MultiHeadEncoder(self.model_id).to(self.device)

    def _batch_iter(self, texts, batch_size):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def fit(self, train_records: list[dict]) -> None:
        import torch
        from torch.optim import AdamW

        self._build()
        texts = [_record_text(r) for r in train_records]
        y_score, y_width, y_contr = [], [], []
        for r in train_records:
            g = get_ground_truth(r)
            y_score.append(float(g.get("score", 0)))
            y_width.append(float(g.get("width", 0)))
            y_contr.append(int(g.get("controversiality", 0)))
        y_score = np.log1p(np.maximum(np.asarray(y_score), 0.0))
        y_width = np.log1p(np.maximum(np.asarray(y_width), 0.0))
        y_contr = np.asarray(y_contr, dtype=np.int64)

        opt = AdamW(self._model.parameters(), lr=self.lr)
        huber = torch.nn.SmoothL1Loss()
        ce = torch.nn.CrossEntropyLoss()
        self._model.train()
        n = len(texts)
        for epoch in range(self.epochs):
            order = np.random.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_texts = [texts[i] for i in idx]
                enc = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                ys = torch.tensor(y_score[idx], dtype=torch.float32, device=self.device)
                yw = torch.tensor(y_width[idx], dtype=torch.float32, device=self.device)
                yc = torch.tensor(y_contr[idx], dtype=torch.long, device=self.device)

                opt.zero_grad()
                ps, pw, pc = self._model(enc.input_ids, enc.attention_mask)
                loss = (
                    self.score_weight * huber(ps, ys)
                    + self.width_weight * huber(pw, yw)
                    + self.contr_weight * ce(pc, yc)
                )
                loss.backward()
                opt.step()
                total += float(loss.item())
            print(f"[encoder] epoch={epoch+1}/{self.epochs} loss={total / max(1, n // self.batch_size):.4f}")

    def predict(self, record: dict) -> dict:
        import torch

        self._model.eval()
        enc = self._tokenizer(
            [_record_text(record)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            ps, pw, pc = self._model(enc.input_ids, enc.attention_mask)
        s = float(np.expm1(ps.item()))
        w = float(np.expm1(pw.item()))
        c = int(pc.argmax(dim=-1).item())
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
