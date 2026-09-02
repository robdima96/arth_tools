# -*- coding: utf-8 -*-
"""
Pixel conversion helpers used by export and the training dataset.

Ultrasound arrays are min-max scaled to uint8. RGB is a 3-channel replicate
of the greyscale plane (control-board REPLICATE_GRAYSCALE_TO_RGB).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def ndarray_to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    while a.ndim > 2:
        a = a.squeeze()
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {a.shape}")
    amin, amax = float(a.min()), float(a.max())
    if amax <= amin + 1e-9:
        # Constant frame: keep the raw 8-bit value instead of collapsing to zeros.
        v = amin
        if 0.0 <= v <= 255.0:
            return np.full(a.shape, int(np.clip(round(v), 0, 255)), dtype=np.uint8)
        return np.full(a.shape, 128, dtype=np.uint8)
    return ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)


def ndarray_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    u8 = ndarray_to_u8(arr)
    return np.stack([u8, u8, u8], axis=-1)


def array_to_pil(arr: np.ndarray, *, rgb: bool = True) -> Image.Image:
    if rgb:
        return Image.fromarray(ndarray_to_rgb_u8(arr), mode="RGB")
    return Image.fromarray(ndarray_to_u8(arr), mode="L")
