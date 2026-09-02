# -*- coding: utf-8 -*-
"""Minimal Darknet-style .cfg reader (Gu-inspired; written here, MIT)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_cfg(path: str | Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                blocks.append(current)
            current = {"type": line[1:-1].strip()}
            continue
        if current is None or "=" not in line:
            continue
        key, _, val = line.partition("=")
        current[key.strip()] = val.strip()
    if current is not None:
        blocks.append(current)
    return blocks


def cfg_num_classes(blocks: list[dict[str, Any]], default: int = 1) -> int:
    for block in blocks:
        if block.get("type") == "yolo" and "classes" in block:
            return int(block["classes"])
    return default
