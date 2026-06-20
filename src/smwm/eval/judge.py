"""LLM-as-a-judge for the predicted `reply_summary` field.

A judge LLM (default Qwen3-32B) rates how well a model's predicted summary of a
comment's replies captures the key points and sentiment of the REFERENCE
summary (the ground-truth Qwen3-4B summary of the actual replies), on a 1-5
scale. Reference-based because the raw reply bodies are not retained in the
result files.

Heavy deps imported lazily; needs a GPU (runner). Reports per-record ratings +
aggregate mean/distribution.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_RE_RATING = re.compile(r'"?rating"?\s*[:=]\s*([1-5])')
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def build_judge_prompt(stimulus_body: str, reference: str, candidate: str) -> str:
    return (
        "You are a strict evaluator of social-media reply summaries.\n\n"
        "A model was asked to summarize the replies to the COMMENT below. You "
        "are given the gold REFERENCE summary (which faithfully summarizes the "
        "real replies) and the model's PREDICTED summary. Rate how well the "
        "PREDICTED summary captures the key points, topics, and overall "
        "sentiment of the REFERENCE summary.\n\n"
        f"COMMENT:\n{_truncate(stimulus_body, 600)}\n\n"
        f"REFERENCE summary:\n{_truncate(reference, 800)}\n\n"
        f"PREDICTED summary:\n{_truncate(candidate, 800)}\n\n"
        "Scale:\n"
        "  5 = captures essentially all main points and the sentiment\n"
        "  4 = captures most points and the sentiment\n"
        "  3 = partially captures; misses some points or nuance\n"
        "  2 = largely misses the points or sentiment\n"
        "  1 = unrelated, generic, or contradictory\n\n"
        'Return ONLY a JSON object: {"rating": <1-5 integer>, "reason": "<one short sentence>"}'
    )


def parse_rating(raw: str) -> int | None:
    if not raw:
        return None
    m = _FENCE.search(raw)
    text = m.group(1) if m else raw
    try:
        obj = json.loads(text.strip())
        r = int(obj.get("rating"))
        if 1 <= r <= 5:
            return r
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _RE_RATING.search(raw)
    return int(m.group(1)) if m else None


class SummaryJudge:
    def __init__(self, model_id: str = "Qwen/Qwen3-32B", load_in_4bit: bool = True,
                 max_new_tokens: int = 128, enable_thinking: bool = False):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tok = None
        self._model: Any = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = os.getenv("HUGGING_FACE")
        if tok:
            from huggingface_hub import login

            login(tok)
        assert torch.cuda.is_available(), "No CUDA device found"
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict = {"device_map": "cuda:0"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def rate(self, stimulus_body: str, reference: str, candidate: str) -> tuple[int | None, str]:
        import torch

        self._load()
        prompt = build_judge_prompt(stimulus_body, reference, candidate)
        text = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.enable_thinking)
        inputs = self._tok([text], return_tensors="pt").to(self._model.device)
        gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out = gen[0][len(inputs.input_ids[0]):].tolist()
        raw = self._tok.decode(out, skip_special_tokens=True).strip()
        return parse_rating(raw), raw
