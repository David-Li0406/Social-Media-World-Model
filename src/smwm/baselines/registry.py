"""Decorator-driven baseline registry.

Usage:
    @register("constant_mean")
    class ConstantMean(Baseline): ...

    BaselineClass = get("constant_mean")
"""
from __future__ import annotations

from typing import Callable, Type

from .base import Baseline

_REGISTRY: dict[str, Type[Baseline]] = {}


def register(name: str) -> Callable[[Type[Baseline]], Type[Baseline]]:
    def deco(cls: Type[Baseline]) -> Type[Baseline]:
        if name in _REGISTRY:
            raise ValueError(f"baseline '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> Type[Baseline]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown baseline '{name}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_baselines() -> list[str]:
    return sorted(_REGISTRY)
