"""Regression-head LLM baseline: Qwen3-4B backbone + numeric heads.

Instead of generating JSON text (which the SFT model botched 81% of the time),
attach regression/classification heads to a Qwen3-4B backbone and predict the
numbers directly — no decoding, no parsing. The backbone is adapted with LoRA
(or frozen via freeze_backbone=true) and mean-pooled over the prompt tokens.

Targets: log1p(score), log1p(width) via Huber; controversiality via CE.
Needs a GPU (runner). Heavy deps imported lazily.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register


def _record_prompt(record: dict) -> str:
    prompt = record.get("prompt")
    if prompt is not None:
        return prompt
    from ..data.build_splits import make_prompt

    return make_prompt(get_context(record), get_stimulus(record))


@register("llm_reghead")
class LLMRegressionHead(Baseline):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        max_length: int = 1024,
        epochs: int = 2,
        batch_size: int = 4,
        grad_accum: int = 4,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "q_proj k_proj v_proj o_proj",
        device: str | None = None,
        gpus: str = "4,5,6,7",
        bf16: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        self.max_length = int(max_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.lr = float(lr)
        self.freeze_backbone = bool(freeze_backbone)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = target_modules
        self.device = device
        self.gpus = gpus
        self.bf16 = bool(bf16)
        self._tok = None
        self._model: Any = None

    def _build(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.gpus.split(",")[0].strip() or "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tok_env = os.getenv("HUGGING_FACE")
        if tok_env:
            from huggingface_hub import login

            login(tok_env)

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        dtype = torch.bfloat16 if self.bf16 else torch.float32
        backbone = AutoModel.from_pretrained(self.model_id, dtype=dtype, attn_implementation="eager")
        if self.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        else:
            from peft import LoraConfig, get_peft_model

            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=self.lora_rank, lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout, bias="none",
                    target_modules=self.target_modules.split(),
                ),
            )
        hidden = (backbone.config.hidden_size if hasattr(backbone, "config")
                  else backbone.base_model.config.hidden_size)

        class RegLLM(nn.Module):
            def __init__(self, bb, h):
                super().__init__()
                self.backbone = bb
                self.score_head = nn.Linear(h, 1)
                self.width_head = nn.Linear(h, 1)
                self.contr_head = nn.Linear(h, 2)

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                hs = out.last_hidden_state
                m = attention_mask.unsqueeze(-1).to(hs.dtype)
                pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
                pooled = pooled.float()
                return (self.score_head(pooled).squeeze(-1),
                        self.width_head(pooled).squeeze(-1),
                        self.contr_head(pooled))

        self._torch = torch
        self._model = RegLLM(backbone, hidden).to(self.device)
        # heads in fp32
        for head in (self._model.score_head, self._model.width_head, self._model.contr_head):
            head.float()

    def _encode(self, texts):
        return self._tok(texts, padding=True, truncation=True,
                         max_length=self.max_length, return_tensors="pt").to(self.device)

    def fit(self, train_records: list[dict]) -> None:
        self._build()
        torch = self._torch
        from torch.optim import AdamW

        usable = [r for r in train_records if get_ground_truth(r)]
        texts = [_record_prompt(r) for r in usable]
        ys, yw, yc = [], [], []
        for r in usable:
            g = get_ground_truth(r)
            ys.append(np.log1p(max(float(g.get("score", 0)), 0.0)))
            yw.append(np.log1p(max(float(g.get("width", 0)), 0.0)))
            yc.append(int(g.get("controversiality", 0)))
        ys = np.asarray(ys, np.float32); yw = np.asarray(yw, np.float32); yc = np.asarray(yc, np.int64)

        params = [p for p in self._model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=self.lr)
        huber = torch.nn.SmoothL1Loss(); ce = torch.nn.CrossEntropyLoss()
        n = len(usable); self._model.train()
        step = 0
        for ep in range(self.epochs):
            order = np.random.permutation(n); total = 0.0
            opt.zero_grad()
            for s in range(0, n, self.batch_size):
                bi = order[s : s + self.batch_size]
                enc = self._encode([texts[j] for j in bi])
                ps, pw, pc = self._model(enc.input_ids, enc.attention_mask)
                loss = (huber(ps, torch.tensor(ys[bi], device=self.device))
                        + huber(pw, torch.tensor(yw[bi], device=self.device))
                        + ce(pc, torch.tensor(yc[bi], device=self.device))) / self.grad_accum
                loss.backward(); total += float(loss.item()) * self.grad_accum
                step += 1
                if step % self.grad_accum == 0:
                    opt.step(); opt.zero_grad()
            opt.step(); opt.zero_grad()
            print(f"[llm_reghead] epoch {ep+1}/{self.epochs} loss={total/max(1,n//self.batch_size):.4f}")

    def predict(self, record: dict) -> dict:
        torch = self._torch
        self._model.eval()
        enc = self._encode([_record_prompt(record)])
        with torch.no_grad():
            ps, pw, pc = self._model(enc.input_ids, enc.attention_mask)
        return {
            "score": int(round(max(float(np.expm1(ps.item())), 0))),
            "width": int(round(max(float(np.expm1(pw.item())), 0))),
            "controversiality": int(pc.argmax(dim=-1).item()),
            "reply_summary": "",
        }
