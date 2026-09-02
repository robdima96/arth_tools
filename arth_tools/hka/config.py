# -*- coding: utf-8 -*-
"""HKA tool settings (not the training control board)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from arth_tools.paths import REPO_ROOT
from arth_tools.training.config import DATA_ROOT, DICOM_ROOT, resolve_config_path


@dataclass
class HKAConfig:
    task_id: str = "hka"
    task_name: str = "Hip-knee-ankle angle (standing AP x-ray)"
    modality: str = "xray"
    anatomy: str = "lower_limb"
    dicom_root: Path = DICOM_ROOT
    output_dir: Path = DATA_ROOT / "hka"
    allowed_modalities: list[str] = field(default_factory=lambda: ["CR", "DX", "RF"])
    overlay: bool = True
    max_hips: int = 2
    bone_percentile: float = 82.0
    preprocess_consume: str = "decode_roi"

    def __post_init__(self) -> None:
        self.dicom_root = Path(self.dicom_root)
        self.output_dir = Path(self.output_dir)
        self.allowed_modalities = [str(m).strip().upper() for m in self.allowed_modalities if str(m).strip()]


def load_hka_config(path: Path | str | None) -> HKAConfig:
    if path is None:
        return HKAConfig()
    import yaml

    dest = resolve_config_path(path)
    raw = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}
    known = {f.name for f in fields(HKAConfig)}
    data: dict[str, Any] = {k: v for k, v in raw.items() if k in known}
    if "output_dir" not in data and raw.get("data_root"):
        data["output_dir"] = Path(raw["data_root"]) / "annotated"
    for key in ("dicom_root", "output_dir"):
        val = data.get(key)
        if val in (None, ""):
            continue
        p = Path(val)
        data[key] = p if p.is_absolute() else (REPO_ROOT / p)
    mods = data.get("allowed_modalities")
    if isinstance(mods, str):
        data["allowed_modalities"] = [mods]
    return HKAConfig(**data)
