# -*- coding: utf-8 -*-
"""Per-modality header-YOLO recipes (not TrainingConfig overlays)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from arth_tools.paths import CONFIGS_DIR, PACKAGE_DIR, REPO_ROOT

MODALITIES = ("xray", "ultrasound", "mri")
CLASS_NAME = "image_region"
WEIGHTS_ROOT = PACKAGE_DIR / "data" / "roi_weights"
CFG_PATH = PACKAGE_DIR / "roi" / "cfgs" / "yolov3-tiny-1cls.cfg"


@dataclass
class RoiConfig:
    modality: str = "xray"
    class_name: str = CLASS_NAME
    img_size: int = 416
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    rotation_deg: float = 15.0
    hflip: bool = False
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    data_root: Path = REPO_ROOT / "data" / "roi"
    weights_dir: Path | None = None
    cfg_path: Path = CFG_PATH
    num_classes: int = 1
    anchors: tuple[float, ...] = (
        10.0, 14.0, 23.0, 27.0, 37.0, 58.0,
        81.0, 82.0, 135.0, 169.0, 344.0, 319.0,
    )

    def __post_init__(self) -> None:
        self.modality = str(self.modality or "xray").strip().lower()
        if self.modality not in MODALITIES:
            raise ValueError(f"ROI modality must be one of {MODALITIES}, got {self.modality!r}")
        self.data_root = Path(self.data_root)
        self.cfg_path = Path(self.cfg_path)
        if self.weights_dir is None:
            self.weights_dir = WEIGHTS_ROOT / self.modality
        else:
            self.weights_dir = Path(self.weights_dir)

    def image_dir(self) -> Path:
        return self.data_root / self.modality / "images"

    def label_dir(self) -> Path:
        return self.data_root / self.modality / "labels"

    def split_dir(self) -> Path:
        return self.data_root / self.modality / "splits"

    def best_weights(self) -> Path:
        assert self.weights_dir is not None
        return self.weights_dir / "best.pt"

    def last_weights(self) -> Path:
        assert self.weights_dir is not None
        return self.weights_dir / "last.pt"


def roi_config_path(modality: str) -> Path:
    return CONFIGS_DIR / f"roi_{modality}.yaml"


def load_roi_config(modality: str | Path, *, overrides: dict[str, Any] | None = None) -> RoiConfig:
    import yaml

    if isinstance(modality, Path) or str(modality).endswith((".yaml", ".yml")):
        path = Path(modality)
        if not path.is_file():
            path = CONFIGS_DIR / path.name
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        mid = str(modality).strip().lower()
        path = roi_config_path(mid)
        raw = {}
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.setdefault("modality", mid)
    if not isinstance(raw, dict):
        raw = {}
    if overrides:
        raw.update(overrides)
    known = {f.name for f in fields(RoiConfig)}
    data = {k: v for k, v in raw.items() if k in known}
    for key in ("data_root", "weights_dir", "cfg_path"):
        val = data.get(key)
        if val in (None, ""):
            continue
        p = Path(val)
        data[key] = p if p.is_absolute() else (REPO_ROOT / p)
    return RoiConfig(**data)
