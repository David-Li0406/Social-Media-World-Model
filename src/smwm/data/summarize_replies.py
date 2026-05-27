"""LLM-summarize reply bodies and collapse width(list) -> width(count).

Ported from findSamples.py. The LLM is loaded lazily so this module is
importable on a CPU-only machine.
"""
from __future__ import annotations

import json
import os
import random
from json.decoder import JSONDecodeError
from pathlib import Path

from .io import iter_jsonl  # noqa: F401  (kept for symmetry)
from .io import read_json, write_jsonl  # noqa: F401


def _make_message(replies: list[dict]) -> list[dict]:
    bodies = "\n\n".join(f"- {reply['body']}" for reply in replies)
    return [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. "
                "Given a list of replies to a social media comment, summarize "
                "the key points and overall sentiment expressed across the body "
                "section of all replies. "
                "Return a JSON object with exactly one field:\n"
                '  - "summary": a string containing the summary.\n\n'
                "Do NOT include anything outside the JSON object. "
                "Do not add extra keys, text, or formatting."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are the replies to summarize:\n\n"
                f"{bodies}\n\n"
                "Return only a JSON object with a single 'summary' field."
            ),
        },
    ]


def _load_qwen(model_id: str, load_in_4bit: bool = True):
    import torch
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = os.getenv("HUGGING_FACE")
    if tok:
        login(tok)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict = {"device_map": "cuda:0"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tokenizer, model


def _generate(messages, tokenizer, model, max_new_tokens=500, retries=3):
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for attempt in range(retries):
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = gen_ids[0][len(inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        try:
            return json.loads(content)
        except JSONDecodeError:
            if attempt == retries - 1:
                return None
    return None


def run(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
    max_new_tokens: int = 500,
    retries: int = 3,
    seed: int = 42,
    limit: int | None = None,
) -> None:
    random.seed(seed)
    chains = read_json(input_path)

    tokenizer, model = _load_qwen(model_id, load_in_4bit=load_in_4bit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chain in enumerate(chains):
            if limit is not None and i >= limit:
                break
            width_list = chain["ground_truth"]["width"]
            if len(width_list) == 0:
                reply_summary = "No replies."
            else:
                resp = _generate(
                    _make_message(width_list),
                    tokenizer,
                    model,
                    max_new_tokens=max_new_tokens,
                    retries=retries,
                )
                reply_summary = resp["summary"] if resp else "Failed"
            chain["ground_truth"]["width"] = len(width_list)
            chain["ground_truth"]["reply_summary"] = reply_summary
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[summarize_replies] {i+1}/{len(chains)} written")
