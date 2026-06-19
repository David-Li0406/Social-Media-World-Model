"""Robust parsing of LLM JSON-ish predictions into the 4-field dict.

LLMs (zero-shot and SFT alike) frequently emit *almost*-JSON: markdown code
fences, a trailing stray quote, truncated strings, etc. Strict json.loads
fails on these even though the numeric fields are present. This module:
  1. strips code fences,
  2. tries strict json.loads,
  3. falls back to regex extraction of score / controversiality / width.

Used by every LLM-family baseline (llm, llm_fewshot, llm_cot, llm_sft) so the
reported numbers reflect the model's real predictions, not a zero fallback.
"""
from __future__ import annotations

import json
import re

_RE_SCORE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+)', re.IGNORECASE)
_RE_CONTR = re.compile(r'"?controvers\w*"?\s*[:=]\s*(\d+)', re.IGNORECASE)
_RE_WIDTH = re.compile(r'"?width"?\s*[:=]\s*(-?\d+)', re.IGNORECASE)
_RE_SUMMARY = re.compile(r'"reply_summary"\s*:\s*"(.*?)"\s*[,}]', re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fence(raw: str) -> str:
    m = _FENCE.search(raw)
    return m.group(1) if m else raw


def regex_extract(raw: str) -> dict:
    out: dict = {}
    if not raw:
        return out
    for key, rx in (("score", _RE_SCORE), ("controversiality", _RE_CONTR), ("width", _RE_WIDTH)):
        m = rx.search(raw)
        if m:
            out[key] = int(m.group(1))
    sm = _RE_SUMMARY.search(raw)
    if sm:
        out["reply_summary"] = sm.group(1)
    return out


def parse_prediction(raw: str) -> dict:
    """Return a normalized {score, controversiality, width, reply_summary} dict.

    Tries strict JSON first (after stripping code fences); on failure recovers
    the numeric fields by regex. Always returns ints for the numeric fields.
    """
    parsed: dict = {}
    if raw:
        candidate = _strip_fence(raw).strip()
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                parsed = {}
        except (json.JSONDecodeError, ValueError):
            parsed = {}
    if not parsed:
        parsed = regex_extract(raw)

    def _i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    return {
        "score": _i(parsed.get("score", 0)),
        "controversiality": _i(parsed.get("controversiality", 0)),
        "width": _i(parsed.get("width", 0)),
        "reply_summary": str(parsed.get("reply_summary", "") or ""),
    }
