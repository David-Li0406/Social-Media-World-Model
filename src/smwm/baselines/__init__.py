"""Registers all baselines on import. Heavy deps (torch, transformers,
sentence-transformers) are loaded lazily inside the baselines themselves."""
from . import constant, encoder, feature, llm, retrieval  # noqa: F401
