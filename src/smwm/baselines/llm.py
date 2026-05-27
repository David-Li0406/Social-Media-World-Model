"""LLM zero-shot prediction baseline.

Wraps the prompt-based inference loop from inference.py (& test_inference.py)
behind the Baseline interface. fit() is a no-op.
"""
from __future__ import annotations

import json
import os
from json.decoder import JSONDecodeError
from typing import Any

from .base import Baseline
from .registry import register


@register("llm")
class LLMBaseline(Baseline):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        max_new_tokens: int = 500,
        retries: int = 3,
        load_in_4bit: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ):
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.retries = int(retries)
        self.load_in_4bit = bool(load_in_4bit)
        self.enable_thinking = bool(enable_thinking)
        self._tokenizer = None
        self._model: Any = None

    def fit(self, train_records: list[dict]) -> None:
        # zero-shot
        return None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from huggingface_hub import login
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        tok = os.getenv("HUGGING_FACE")
        if tok:
            login(tok)
        assert torch.cuda.is_available(), "No CUDA device found"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict = {"device_map": "cuda:0"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def _generate_json(self, prompt: str) -> tuple[dict | None, str]:
        import torch

        self._load()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        raw = ""
        for attempt in range(self.retries):
            gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            out_ids = gen[0][len(inputs.input_ids[0]):].tolist()
            raw = self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            try:
                return json.loads(raw), raw
            except JSONDecodeError:
                continue
        return None, raw

    def predict(self, record: dict) -> dict:
        prompt = record.get("prompt")
        if prompt is None:
            # Build prompt on the fly from structured fields.
            from ..data.build_splits import make_prompt
            from .base import get_context, get_stimulus

            prompt = make_prompt(get_context(record), get_stimulus(record))
        parsed, raw = self._generate_json(prompt)
        if parsed is None:
            return {
                "score": 0,
                "width": 0,
                "controversiality": 0,
                "reply_summary": "",
                "_raw": raw,
            }
        return {
            "score": int(parsed.get("score", 0)),
            "width": int(parsed.get("width", 0)),
            "controversiality": int(parsed.get("controversiality", 0)),
            "reply_summary": str(parsed.get("reply_summary", "")),
            "_raw": raw,
        }
