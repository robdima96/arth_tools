# -*- coding: utf-8 -*-
"""Sample decoded vs recipe-preprocessed images for UI sanity checks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from arth_tools.data.labels import patient_id_from_dataset
from arth_tools.data.preprocess import (
    RASTER_SUFFIXES,
    PreprocessRecipe,
    decode_dicom,
    load_image_array,
    ndarray_to_u8,
)

PREVIEW_N = 10


@dataclass
class PreviewRow:
    path: Path
    patient_id: str
    loaded_u8: np.ndarray
    processed_u8: np.ndarray | None = None


def float_hwc_to_display_u8(arr: np.ndarray) -> np.ndarray:
    """Map a 0–1 (or already scaled) HWC float tensor to uint8 for display."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {a.shape}")
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax <= 1.0 + 1e-3 and amin >= -1e-6:
        u8 = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    elif amax <= amin + 1e-9:
        u8 = np.zeros(a.shape, dtype=np.uint8)
    else:
        u8 = ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)
    if u8.shape[-1] == 1:
        return u8[..., 0]
    return u8


def _raster_loaded_u8(path: Path) -> np.ndarray:
    pil = Image.open(path)
    arr = np.asarray(pil)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.stack(
            [ndarray_to_u8(arr[..., 0]), ndarray_to_u8(arr[..., 1]), ndarray_to_u8(arr[..., 2])],
            axis=-1,
        )
    return ndarray_to_u8(arr)


def loaded_display(path: Path) -> tuple[np.ndarray, str]:
    """Decode a file for display: photometric invert only, no ROI / window / resize."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return _raster_loaded_u8(path), path.parent.name or path.stem
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"No pixel data: {path}")
    arr, _meta = decode_dicom(ds, apply_voi=False)
    pid = patient_id_from_dataset(ds) or path.parent.name or path.stem
    return ndarray_to_u8(arr), str(pid)


def processed_display(path: Path, recipe: PreprocessRecipe) -> np.ndarray:
    """Recipe tensor (no train-set z-score) scaled to uint8."""
    rec = PreprocessRecipe.from_dict(recipe.to_dict())
    rec.zscore_mean = None
    rec.zscore_std = None
    arr = load_image_array(path, rec)
    return float_hwc_to_display_u8(arr)


def preview_row(
    path: Path,
    *,
    recipe: PreprocessRecipe | None = None,
    patient_id: str | None = None,
) -> PreviewRow:
    path = Path(path)
    loaded, pid = loaded_display(path)
    processed = processed_display(path, recipe) if recipe is not None else None
    return PreviewRow(
        path=path,
        patient_id=str(patient_id or pid),
        loaded_u8=loaded,
        processed_u8=processed,
    )


def _iter_path_items(
    paths: Iterable[Any],
) -> list[tuple[Path, str | None]]:
    items: list[tuple[Path, str | None]] = []
    for raw in paths:
        if isinstance(raw, dict):
            path = Path(raw["path"])
            pid = raw.get("patient_id")
            items.append((path, str(pid) if pid else None))
        else:
            items.append((Path(raw), None))
    return items


def sample_preview_rows(
    paths: Sequence[Any],
    *,
    recipe: PreprocessRecipe | None = None,
    n: int = PREVIEW_N,
    seed: int = 0,
    shuffle: bool = True,
) -> list[PreviewRow]:
    """Up to n readable files. Skip decode failures. Paired loaded/processed when recipe is set."""
    items = _iter_path_items(paths)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(items)
    rows: list[PreviewRow] = []
    for path, pid in items:
        if len(rows) >= int(n):
            break
        try:
            rows.append(preview_row(path, recipe=recipe, patient_id=pid))
        except Exception:
            continue
    return rows
