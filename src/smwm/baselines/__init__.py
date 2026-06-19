"""Registers all baselines on import. Heavy deps (torch, transformers,
sentence-transformers, lightgbm) are loaded lazily inside the baselines."""
from . import (  # noqa: F401
    constant,
    encoder,
    feature,
    frozen_mlp,
    glm,
    gnn,
    hawkes,
    llm,
    llm_reghead,
    llm_sft,
    quantile_gbm,
    retrieval,
    structural,
)
