"""LLM prompting baselines: zero-shot, few-shot ICL, chain-of-thought.

All share one generation path and the robust parser in _llm_parse (strict JSON
-> regex recovery), so malformed-JSON outputs never collapse to a zero
prediction. Subclasses override `build_messages()` to change the prompt
strategy. fit() is a no-op for zero-shot/CoT; few-shot uses it to index the
train set for exemplar retrieval.
"""
from __future__ import annotations

import os
from typing import Any

from ._llm_parse import parse_prediction
from .base import Baseline, get_context, get_stimulus
from .registry import register


def _record_prompt(record: dict) -> str:
    prompt = record.get("prompt")
    if prompt is not None:
        return prompt
    from ..data.build_splits import make_prompt

    return make_prompt(get_context(record), get_stimulus(record))


@register("llm")
class LLMBaseline(Baseline):
    """Zero-shot JSON prediction."""

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
        return None

    # ---- prompt strategy (overridden by subclasses) ----
    def build_messages(self, record: dict) -> list[dict]:
        return [{"role": "user", "content": _record_prompt(record)}]

    def _load(self):
        if self._model is not None:
            return
        import torch
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    def _generate(self, messages: list[dict]) -> str:
        import torch

        self._load()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out_ids = gen[0][len(inputs.input_ids[0]):].tolist()
        return self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    def predict(self, record: dict) -> dict:
        raw = self._generate(self.build_messages(record))
        pred = parse_prediction(raw)
        pred["_raw"] = raw
        return pred


@register("llm_cot")
class LLMChainOfThought(LLMBaseline):
    """Ask the model to reason about the audience first, then emit JSON.

    Reasoning text precedes the JSON; parse_prediction strips fences / recovers
    the final object by regex, so the chatter doesn't break parsing.
    """

    def build_messages(self, record: dict) -> list[dict]:
        base = _record_prompt(record)
        cot = (
            base
            + "\n\nFirst, briefly reason step by step about who is in this "
            "conversation, the likely audience reaction, and how divisive the "
            "stimulus is. Then, on a final line, output ONLY the JSON object "
            "with keys score, controversiality, width, reply_summary."
        )
        return [{"role": "user", "content": cot}]


@register("llm_fewshot")
class LLMFewShot(LLMBaseline):
    """Retrieval-augmented in-context learning: prepend k similar train examples."""

    def __init__(self, k: int = 4, max_features: int = 20000, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.max_features = int(max_features)
        self._vec = None
        self._train_X = None
        self._exemplars: list[dict] = []

    def fit(self, train_records: list[dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._exemplars = [r for r in train_records if r.get("completion") or r.get("ground_truth")]
        texts = [_record_prompt(r) for r in self._exemplars]
        self._vec = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        self._train_X = self._vec.fit_transform(texts)

    def _exemplar_block(self, record: dict) -> str:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vec.transform([_record_prompt(record)])
        sims = cosine_similarity(q, self._train_X).ravel()
        k = min(self.k, len(self._exemplars))
        top = np.argsort(-sims)[:k]
        blocks = []
        for j in top:
            ex = self._exemplars[int(j)]
            gt = ex.get("completion")
            if gt is None:
                import json
                gt = json.dumps(ex.get("ground_truth", {}), ensure_ascii=False)
            blocks.append(f"Example input:\n{_record_prompt(ex)}\nExample answer: {gt}")
        return "\n\n".join(blocks)

    def build_messages(self, record: dict) -> list[dict]:
        block = self._exemplar_block(record)
        content = (
            "Here are similar solved examples:\n\n" + block
            + "\n\nNow solve this one. Output ONLY the JSON object.\n\n"
            + _record_prompt(record)
        )
        return [{"role": "user", "content": content}]
