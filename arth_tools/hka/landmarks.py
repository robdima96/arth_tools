# -*- coding: utf-8 -*-
"""Detect hip, knee, and ankle centers on a standing AP long-leg radiograph.

Classical (no trained detector): femoral heads are bright disks in the upper
film (distance-transform peaks); knee and ankle are joint-space rows in a
vertical strip under each head. Swap in another detector later by returning
the same ``Point`` list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt, gaussian_filter, maximum_filter

from arth_tools.hka.geometry import (
    HKAResult,
    Point,
    assign_laterality,
    hka_from_points,
    pixel_spacing_mm,
)


def dataset_to_gray(ds: Any) -> np.ndarray:
    from arth_tools.data.preprocess import PreprocessRecipe, apply_stages_from_dataset

    recipe = PreprocessRecipe(preprocess_consume="decode_roi", modality_id="xray", roi_model_id="xray")
    arr = apply_stages_from_dataset(ds, recipe)
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D pixels, got shape {a.shape}")
    return a


def _local_maxima(dt: np.ndarray, *, min_radius: float, min_sep: int, max_peaks: int) -> list[Point]:
    sep = max(3, int(min_sep))
    footprint = np.ones((2 * sep + 1, 2 * sep + 1), dtype=bool)
    mx = maximum_filter(dt, footprint=footprint)
    peaks = (dt == mx) & (dt >= min_radius)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return []
    vals = dt[peaks]
    order = np.argsort(-vals)
    picked: list[Point] = []
    for idx in order:
        r, c = (int(coords[idx][0]), int(coords[idx][1]))
        if any((r - p.row) ** 2 + (c - p.col) ** 2 < sep * sep for p in picked):
            continue
        picked.append(Point(row=float(r), col=float(c)))
        if len(picked) >= max_peaks:
            break
    return picked


def detect_hips(
    gray: np.ndarray,
    *,
    band: tuple[float, float] = (0.0, 0.40),
    bone_percentile: float = 82.0,
    max_hips: int = 2,
) -> list[Point]:
    h, w = gray.shape
    r0, r1 = int(h * band[0]), max(int(h * band[1]), int(h * band[0]) + 8)
    top = gray[r0:r1]
    sigma = max(1.0, min(top.shape) / 180.0)
    blur = gaussian_filter(top, sigma=sigma)
    thr = float(np.percentile(blur, bone_percentile))
    mask = blur >= thr
    mask = binary_opening(mask, iterations=1)
    if not np.any(mask):
        return []
    dt = distance_transform_edt(mask)
    min_r = max(3.0, min(h, w) * 0.012)
    min_sep = max(8, int(w * 0.12))
    peaks = _local_maxima(dt, min_radius=min_r, min_sep=min_sep, max_peaks=max_hips)
    return [Point(row=p.row + r0, col=p.col) for p in peaks]


def _centroid_on_row(gray: np.ndarray, row: int, col: float, half_width: int, bone_thr: float) -> float:
    h, w = gray.shape
    r = int(np.clip(row, 0, h - 1))
    c0 = int(np.clip(col - half_width, 0, w - 1))
    c1 = int(np.clip(col + half_width + 1, 1, w))
    strip = gray[r, c0:c1]
    weights = np.clip(strip - bone_thr, 0.0, None)
    if float(weights.sum()) < 1e-6:
        return float(col)
    xs = np.arange(c0, c1, dtype=np.float64)
    return float(np.dot(xs, weights) / weights.sum())


def _joint_in_band(
    gray: np.ndarray,
    col: float,
    *,
    band: tuple[float, float],
    half_width: int,
    bone_percentile: float,
) -> Point | None:
    h, w = gray.shape
    r0 = int(np.clip(h * band[0], 0, h - 2))
    r1 = int(np.clip(h * band[1], r0 + 4, h))
    c0 = int(np.clip(col - half_width, 0, w - 1))
    c1 = int(np.clip(col + half_width + 1, c0 + 2, w))
    region = gray[r0:r1, c0:c1]
    if region.size == 0:
        return None
    sigma = max(1.0, min(region.shape) / 80.0)
    blur = gaussian_filter(region.astype(np.float64), sigma=sigma)
    thr = float(np.percentile(blur, bone_percentile))
    mask = blur >= thr
    if np.any(mask):
        dt = distance_transform_edt(mask)
        # Prefer a compact blob (joint) near the hip column.
        peak = np.unravel_index(int(np.argmax(dt)), dt.shape)
        if float(dt[peak]) >= 2.0:
            return Point(row=float(r0 + peak[0]), col=float(c0 + peak[1]))
    occ = (region >= thr).mean(axis=1)
    if occ.size < 4:
        return None
    smooth = gaussian_filter(occ.astype(np.float64), sigma=max(1.0, occ.size / 25.0))
    row = r0 + int(np.argmin(smooth))
    jcol = _centroid_on_row(gray, row, col, half_width=half_width * 2, bone_thr=thr)
    return Point(row=float(row), col=float(jcol))


def detect_limbs(
    gray: np.ndarray,
    *,
    hip_band: tuple[float, float] = (0.0, 0.40),
    knee_band: tuple[float, float] = (0.40, 0.72),
    ankle_band: tuple[float, float] = (0.72, 1.00),
    bone_percentile: float = 82.0,
    max_hips: int = 2,
) -> list[tuple[str, Point, Point, Point]]:
    hips = detect_hips(gray, band=hip_band, bone_percentile=bone_percentile, max_hips=max_hips)
    if not hips:
        return []
    _, w = gray.shape
    half = max(6, int(w * 0.08))
    labeled = assign_laterality(hips)
    midline = float(np.mean([p.col for p in hips]))
    found: list[tuple[str, Point, Point, Point]] = []
    for laterality, hip in labeled:
        knee = _joint_in_band(
            gray, hip.col, band=knee_band, half_width=half, bone_percentile=bone_percentile
        )
        ankle = _joint_in_band(
            gray, hip.col, band=ankle_band, half_width=half, bone_percentile=bone_percentile
        )
        if knee is None or ankle is None:
            continue
        if not (hip.row < knee.row < ankle.row):
            continue
        found.append((laterality, hip, knee, ankle))
    _ = midline
    return found


def measure_hka(ds: Any, *, max_hips: int = 2, bone_percentile: float = 82.0) -> HKAResult:
    try:
        gray = dataset_to_gray(ds)
    except Exception as exc:
        return HKAResult(ok=False, message=f"Could not read pixels: {exc}")
    row_mm, col_mm = pixel_spacing_mm(ds)
    limbs_pts = detect_limbs(gray, max_hips=max_hips, bone_percentile=bone_percentile)
    if not limbs_pts:
        return HKAResult(
            ok=False,
            message="No hip-knee-ankle chain found (need a standing AP long-leg film).",
            spacing_mm=(row_mm, col_mm),
        )
    midline = float(np.mean([hip.col for _, hip, _, _ in limbs_pts]))
    limbs = []
    for laterality, hip, knee, ankle in limbs_pts:
        limbs.append(
            hka_from_points(
                hip,
                knee,
                ankle,
                laterality=laterality,
                row_mm=row_mm,
                col_mm=col_mm,
                midline_col=midline,
            )
        )
    return HKAResult(ok=True, message="ok", limbs=limbs, spacing_mm=(row_mm, col_mm))
