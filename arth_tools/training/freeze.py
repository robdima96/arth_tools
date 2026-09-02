"""Criteria-gated freeze of the best checkpoint.

A copy is written under the run report AND (when possible) under E:\\ArthAgent\\frozen.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


def decide_freeze(
    *,
    best_epoch: int,
    best_metric: float,
    primary_metric: str,
    min_value: float,
    min_epoch: int,
    mode: str = "max",
) -> dict[str, Any]:
    metric_ok = (
        best_metric >= min_value
        if mode == "max"
        else best_metric <= min_value
    )
    epoch_ok = best_epoch >= min_epoch
    frozen = bool(metric_ok and epoch_ok)
    return {
        "frozen": frozen,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "primary_metric": primary_metric,
        "min_value": min_value,
        "min_epoch": min_epoch,
        "mode": mode,
        "metric_ok": metric_ok,
        "epoch_ok": epoch_ok,
    }


def write_freeze_decision(run_dir: Path, decision: dict[str, Any]) -> Path:
    dest = Path(run_dir) / "freeze_decision.json"
    dest.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return dest


def freeze_if_criteria_met(
    run_dir: Path,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    decision: dict[str, Any],
    extra_frozen_dir: Path | None = None,
) -> Path | None:
    write_freeze_decision(run_dir, decision)
    if not decision["frozen"]:
        return None
    frozen_dir = Path(run_dir) / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    dest_pt = frozen_dir / "model.pt"
    shutil.copy2(checkpoint_path, dest_pt)
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            torch.save(blob["model_state_dict"], frozen_dir / "state_dict.pt")
    except Exception:
        pass
    (frozen_dir / "metadata.json").write_text(
        json.dumps({**metadata, "freeze": decision}, indent=2, default=str),
        encoding="utf-8",
    )
    if extra_frozen_dir is not None:
        try:
            extra_frozen_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest_pt, extra_frozen_dir / "model.pt")
            shutil.copy2(frozen_dir / "metadata.json", extra_frozen_dir / "metadata.json")
        except OSError:
            pass
    return dest_pt
