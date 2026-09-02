# -*- coding: utf-8 -*-
"""Burn HKA axes onto a copy of the source DICOM and store the angle in tags."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom.uid import generate_uid

from arth_tools.data.preprocess import ndarray_to_u8
from arth_tools.hka.geometry import HKAResult, LimbHKA
from arth_tools.hka.landmarks import dataset_to_gray


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_limb(draw: ImageDraw.ImageDraw, limb: LimbHKA, scale: float) -> None:
    def xy(pt: Any) -> tuple[int, int]:
        return (int(round(pt.col)), int(round(pt.row)))

    width = max(2, int(round(2 * scale)))
    r = max(4, int(round(6 * scale)))
    color = 255
    draw.line([xy(limb.hip), xy(limb.knee), xy(limb.ankle)], fill=color, width=width)
    for pt in (limb.hip, limb.knee, limb.ankle):
        x, y = xy(pt)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)
    label = f"{limb.laterality[0].upper()} {limb.hka_deg:.1f}"
    tx, ty = xy(limb.knee)
    draw.text((tx + r + 2, ty - 4), label, fill=color, font=_font(max(10, int(12 * scale))))


def overlay_gray(gray: np.ndarray, result: HKAResult) -> np.ndarray:
    """Return uint8 image with axes and angle text burned in."""
    u8 = ndarray_to_u8(gray)
    h, w = u8.shape
    scale = min(h, w) / 512.0
    im = Image.fromarray(u8, mode="L")
    draw = ImageDraw.Draw(im)
    for limb in result.limbs:
        _draw_limb(draw, limb, scale)
    header = result.summary_text()
    draw.text((8, 8), header[:180], fill=255, font=_font(max(10, int(14 * scale))))
    return np.asarray(im, dtype=np.uint8)


def _scale_u8_to_original(u8: np.ndarray, original: np.ndarray) -> np.ndarray:
    src = np.asarray(original)
    if src.dtype == np.uint8:
        return u8.astype(np.uint8)
    lo, hi = float(np.min(src)), float(np.max(src))
    if hi <= lo + 1e-9:
        return np.full(u8.shape, src.flat[0] if src.size else 0, dtype=src.dtype)
    scaled = lo + (u8.astype(np.float64) / 255.0) * (hi - lo)
    if np.issubdtype(src.dtype, np.integer):
        info = np.iinfo(src.dtype)
        scaled = np.clip(np.round(scaled), info.min, info.max)
    return scaled.astype(src.dtype)


def write_hka_dicom(ds: Any, result: HKAResult, dest: Path) -> Path:
    """Copy ``ds``, burn the overlay, write angle into ImageComments / SeriesDescription."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = copy.deepcopy(ds)
    gray = dataset_to_gray(ds)
    burned = overlay_gray(gray, result)
    original = np.asarray(ds.pixel_array)
    while original.ndim > 2:
        if original.shape[-1] in (3, 4):
            original = original[..., 0]
        else:
            original = np.squeeze(original)
    pixels = _scale_u8_to_original(burned, original)

    new_uid = generate_uid()
    out.SOPInstanceUID = new_uid
    if hasattr(out, "file_meta") and out.file_meta is not None:
        out.file_meta.MediaStorageSOPInstanceUID = new_uid
    out.SeriesInstanceUID = generate_uid()
    out.SeriesDescription = "HKA annotated"
    comment = result.summary_text()[:1024]
    out.ImageComments = comment
    if hasattr(out, "DerivationDescription"):
        out.DerivationDescription = comment
    else:
        try:
            out.DerivationDescription = comment
        except Exception:
            pass
    out.Rows = int(pixels.shape[0])
    out.Columns = int(pixels.shape[1])
    if pixels.dtype == np.uint8:
        out.BitsAllocated = 8
        out.BitsStored = 8
        out.HighBit = 7
        out.PixelRepresentation = 0
    elif pixels.dtype == np.uint16:
        out.BitsAllocated = 16
        out.BitsStored = 16
        out.HighBit = 15
        out.PixelRepresentation = 0
    out.SamplesPerPixel = 1
    out.PhotometricInterpretation = "MONOCHROME2"
    out.PixelData = np.ascontiguousarray(pixels).tobytes()
    out.save_as(str(dest), write_like_original=False)
    return dest
