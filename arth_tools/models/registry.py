# -*- coding: utf-8 -*-
"""Architecture registry: add a backbone without editing train.py / evaluate.py."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

Builder = Callable[[Any], nn.Module]
FreezeFn = Callable[[nn.Module, bool], None]

_BUILDERS: dict[str, Builder] = {}
_ALIASES: dict[str, str] = {}
_FREEZE: dict[str, FreezeFn] = {}


def normalize_architecture_id(name: str) -> str:
    key = (name or "").strip().lower()
    return _ALIASES.get(key, key)


def register(
    name: str,
    builder: Builder,
    *,
    aliases: tuple[str, ...] = (),
    freeze_fn: FreezeFn | None = None,
) -> None:
    key = name.strip().lower()
    _BUILDERS[key] = builder
    if freeze_fn is not None:
        _FREEZE[key] = freeze_fn
    for alias in aliases:
        _ALIASES[alias.strip().lower()] = key


def registered_architectures() -> list[str]:
    return sorted(_BUILDERS)


def get_builder(architecture_id: str) -> Builder:
    key = normalize_architecture_id(architecture_id)
    if key not in _BUILDERS:
        known = ", ".join(registered_architectures()) or "(none registered)"
        raise ValueError(f"Unknown architecture_id: {architecture_id!r}. Known: {known}")
    return _BUILDERS[key]


def freeze_fn_for(architecture_id: str) -> FreezeFn | None:
    key = normalize_architecture_id(architecture_id)
    return _FREEZE.get(key)
