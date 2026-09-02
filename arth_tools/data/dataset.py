# -*- coding: utf-8 -*-
"""
Torch dataset from a CSV manifest written by export/prepare (filepath, label, patient_id).

Image-only — no segmentation channel. Grayscale ultrasound is replicated to
3 channels when the control board says so (ResNet / 3-ch CNN). Inference can
load unlabeled PNG/DICOM paths with the train-time PreprocessRecipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from arth_tools.data.preprocess import PreprocessRecipe, load_image_array, pil_to_float_hwc
from arth_tools.training.config import (
    FILEPATH_COLUMN,
    LABEL_COLUMN,
    PATIENT_ID_COLUMN,
    REPLICATE_GRAYSCALE_TO_RGB,
    RESIZE_DIMS,
)


def resolve_path(manifest_csv: Path, raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (manifest_csv.parent / p).resolve()


class ManifestClassificationDataset(Dataset):
    """One row of a manifest CSV → one (C, H, W) tensor + label."""

    def __init__(
        self,
        manifest_csv: Path,
        *,
        filepath_col: str = FILEPATH_COLUMN,
        label_col: str = LABEL_COLUMN,
        patient_col: str = PATIENT_ID_COLUMN,
        replicate_gray: bool | None = None,
        image_size: tuple[int, int] = RESIZE_DIMS,
        jitter: bool = False,
        regression: bool = False,
        output_type: str = "softmax",
        loss_name: str = "sparse_categorical_crossentropy",
        zscore_mean: float | None = None,
        zscore_std: float | None = None,
        require_label: bool = True,
        recipe: PreprocessRecipe | None = None,
    ) -> None:
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.is_file():
            raise FileNotFoundError(manifest_csv)
        df = pd.read_csv(manifest_csv)
        if df.empty:
            raise ValueError(f"Manifest empty: {manifest_csv}")
        if filepath_col not in df.columns:
            raise KeyError(f"{manifest_csv.name} missing {filepath_col!r}; have {list(df.columns)}")
        if require_label and label_col not in df.columns:
            raise KeyError(f"{manifest_csv.name} missing {label_col!r}; have {list(df.columns)}")
        if label_col not in df.columns:
            df[label_col] = 0

        self.manifest_csv = manifest_csv
        self.rows = df
        self.filepath_col = filepath_col
        self.label_col = label_col
        self.patient_col = patient_col if patient_col in df.columns else None
        self.replicate_gray = (
            REPLICATE_GRAYSCALE_TO_RGB if replicate_gray is None else replicate_gray
        )
        self.image_size = image_size
        self.jitter = jitter
        self.regression = regression
        self.output_type = output_type
        self.loss_name = loss_name
        self.zscore_mean = zscore_mean
        self.zscore_std = zscore_std
        self.require_label = require_label
        self.recipe = recipe

        missing = []
        for rel in df[filepath_col]:
            path = resolve_path(manifest_csv, str(rel))
            if not path.is_file():
                missing.append(str(path))
        if missing:
            preview = missing[:8]
            print(f"WARNING: missing files ({len(missing)} total); first samples: {preview}")

    def set_zscore(self, mean: float, std: float) -> None:
        self.zscore_mean = float(mean)
        self.zscore_std = float(std) + 1e-8

    def __len__(self) -> int:
        return len(self.rows)

    def patient_id_for_index(self, index: int) -> str:
        if self.patient_col is None:
            return "unknown"
        raw = self.rows.iloc[index][self.patient_col]
        if pd.isna(raw):
            return "unknown"
        return str(raw)

    def filepath_for_index(self, index: int) -> str:
        row = self.rows.iloc[index]
        return str(resolve_path(self.manifest_csv, str(row[self.filepath_col])))

    def labels_array(self) -> list[int]:
        return [int(v) for v in self.rows[self.label_col].tolist()]

    def _load_array(self, index: int) -> np.ndarray:
        row = self.rows.iloc[index]
        fp = resolve_path(self.manifest_csv, str(row[self.filepath_col]))
        if self.recipe is not None:
            recipe = PreprocessRecipe.from_dict(self.recipe.to_dict())
            recipe.zscore_mean = None
            recipe.zscore_std = None
            recipe.resize_width = int(self.image_size[0])
            recipe.resize_height = int(self.image_size[1])
        else:
            recipe = PreprocessRecipe(
                resize_width=int(self.image_size[0]),
                resize_height=int(self.image_size[1]),
                replicate_grayscale_to_rgb=bool(self.replicate_gray),
                input_channels=3 if self.replicate_gray else 1,
                zscore_mean=None,
                zscore_std=None,
            )
        suffix = Path(fp).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            return pil_to_float_hwc(Image.open(fp), recipe)
        return load_image_array(fp, recipe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        np_im = self._load_array(index)
        if self.jitter:
            # Brightness jitter only (geometric aug is not applied here —
            # minority oversampling covers class mix).
            np_im = np.clip(np_im * float(0.85 + 0.3 * np.random.random()), 0.0, 1.0)
        if self.zscore_mean is not None and self.zscore_std is not None:
            np_im = (np_im - self.zscore_mean) / self.zscore_std
            np_im = np.nan_to_num(np_im, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.from_numpy(np_im).permute(2, 0, 1).contiguous()

        row = self.rows.iloc[index]
        lab = row[self.label_col]
        if pd.isna(lab):
            if not self.require_label:
                lab = 0
            else:
                fp = resolve_path(self.manifest_csv, str(row[self.filepath_col]))
                raise ValueError(f"Missing label row {index} for {fp}")
        if self.regression or (
            self.loss_name == "binary_crossentropy" and self.output_type == "sigmoid"
        ):
            target = torch.tensor(float(lab), dtype=torch.float32)
        else:
            target = torch.tensor(int(lab), dtype=torch.long)
        return tensor, target


def compute_train_zscore(ds: ManifestClassificationDataset, max_images: int = 256) -> tuple[float, float]:
    """Pixel mean/std from training images (unmasked; exported PNGs have no mask)."""
    n = min(len(ds), max_images)
    acc = []
    for i in range(n):
        acc.append(ds._load_array(i).ravel())
    pix = np.concatenate(acc) if acc else np.array([0.0], dtype=np.float32)
    return float(np.mean(pix)), float(np.std(pix) + 1e-8)


class ImagePathDataset(Dataset):
    """Unlabeled PNG/DICOM paths scored with a frozen PreprocessRecipe."""

    def __init__(self, items: list[dict[str, Any]], recipe: PreprocessRecipe) -> None:
        if not items:
            raise ValueError("No images to score")
        self.items = items
        self.recipe = recipe

    def __len__(self) -> int:
        return len(self.items)

    def patient_id_for_index(self, index: int) -> str:
        return str(self.items[index].get("patient_id") or "unknown")

    def filepath_for_index(self, index: int) -> str:
        return str(Path(self.items[index]["path"]).resolve())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.items[index]["path"])
        np_im = load_image_array(path, self.recipe)
        tensor = torch.from_numpy(np_im).permute(2, 0, 1).contiguous()
        return tensor, torch.tensor(0, dtype=torch.long)


def items_from_manifest(
    manifest_csv: Path,
    *,
    filepath_col: str = FILEPATH_COLUMN,
    patient_col: str = PATIENT_ID_COLUMN,
) -> list[dict[str, Any]]:
    manifest_csv = Path(manifest_csv)
    df = pd.read_csv(manifest_csv)
    if filepath_col not in df.columns:
        raise KeyError(f"{manifest_csv.name} missing {filepath_col!r}; have {list(df.columns)}")
    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        path = resolve_path(manifest_csv, str(row[filepath_col]))
        pid = "unknown"
        if patient_col in df.columns and not pd.isna(row.get(patient_col)):
            pid = str(row[patient_col])
        items.append({"path": path, "patient_id": pid})
    if not items:
        raise ValueError(f"Manifest empty: {manifest_csv}")
    return items
