"""Sim-output JSONL -> shuffled train/test JSONL with prompt+completion.

Ported from buildTrainTest.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .io import read_jsonl, write_jsonl


def make_prompt(context: list[dict], stimulus: dict) -> str:
    return (
        "You are analyzing a Reddit political discussion. "
        "Given the conversation context and a stimulus comment, predict:\n"
        "  1. The score (upvotes) the stimulus will receive\n"
        "  2. Whether the stimulus is controversial (0 or 1)\n"
        "  3. The number of direct reply comments (width) the stimulus will generate\n\n"
        "  4. A summary of the body fields of the direct reply comments (width) the stimulus will generate"
        "Return a JSON object with exactly these fields:\n"
        '  - "score": an integer with the predicted score\n'
        '  - "controversiality": 0 or 1\n'
        '  - "width": an integer with the predicted number of direct reply comments'
        '  - "reply_summary": a string'
        "Do NOT include anything outside the JSON object.\n\n"
        "conversation context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "stimulus comment:\n"
        f"{json.dumps(stimulus, ensure_ascii=False, indent=2)}\n\n"
        "Now provide score, controversiality, width, and reply_summary as a JSON object."
    )


def make_target(ground_truth: dict) -> str:
    return json.dumps(ground_truth, ensure_ascii=False)


def run(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    records = read_jsonl(input_path)

    examples: list[dict] = []
    for idx, record in enumerate(records):
        context = record.get("context", [])
        stimulus = record.get("stimulus")
        ground_truth = record.get("ground_truth")
        if stimulus is None or ground_truth is None:
            continue
        if ground_truth.get("score") is None or ground_truth.get("reply_summary") is None:
            continue
        examples.append(
            {
                "record_id": idx,
                "context": context,
                "stimulus": stimulus,
                "ground_truth": ground_truth,
                "prompt": make_prompt(context, stimulus),
                "completion": make_target(ground_truth),
            }
        )

    rng.shuffle(examples)
    split = int(len(examples) * (1 - test_fraction))
    train, test = examples[:split], examples[split:]
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    print(f"[build_splits] total={len(examples)} train={len(train)} test={len(test)}")
