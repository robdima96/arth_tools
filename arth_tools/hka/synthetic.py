# -*- coding: utf-8 -*-
"""Synthetic standing AP long-leg radiograph (no PHI) for HKA tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from arth_tools.data.dicom import write_synthetic_dicom
from arth_tools.hka.geometry import Point


def _disk(arr: np.ndarray, row: int, col: int, radius: int, value: int) -> None:
    h, w = arr.shape
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius * radius
    arr[r0:r1, c0:c1][mask] = value


def _shaft(arr: np.ndarray, r0: int, r1: int, col: int, half: int, value: int) -> None:
    h, w = arr.shape
    rr0, rr1 = max(0, r0), min(h, r1)
    cc0, cc1 = max(0, col - half), min(w, col + half + 1)
    arr[rr0:rr1, cc0:cc1] = np.maximum(arr[rr0:rr1, cc0:cc1], value)


def render_longleg(
    *,
    height: int = 480,
    width: int = 240,
    right_hip: tuple[int, int] = (70, 70),
    right_knee: tuple[int, int] = (240, 70),
    right_ankle: tuple[int, int] = (420, 70),
    left_hip: tuple[int, int] = (70, 170),
    left_knee: tuple[int, int] = (240, 170),
    left_ankle: tuple[int, int] = (420, 190),
) -> tuple[np.ndarray, dict[str, tuple[Point, Point, Point]]]:
    """Bright bones on a dark field. Coordinates are (row, col)."""
    img = np.full((height, width), 20, dtype=np.uint8)
    limbs = {
        "right": (right_hip, right_knee, right_ankle),
        "left": (left_hip, left_knee, left_ankle),
    }
    truth: dict[str, tuple[Point, Point, Point]] = {}
    for name, (hip, knee, ankle) in limbs.items():
        hr, hc = hip
        kr, kc = knee
        ar, ac = ankle
        _disk(img, hr, hc, 18, 220)
        _shaft(img, hr, kr, hc, 8, 180)
        _disk(img, kr, kc, 12, 210)
        _shaft(img, kr, ar, kc if kc == ac else (kc + ac) // 2, 7, 180)
        # slanted tibia: fill along the line
        n = max(abs(ar - kr), abs(ac - kc), 1)
        for t in range(n + 1):
            r = int(round(kr + (ar - kr) * t / n))
            c = int(round(kc + (ac - kc) * t / n))
            _disk(img, r, c, 6, 180)
        _disk(img, ar, ac, 10, 200)
        truth[name] = (
            Point(row=float(hr), col=float(hc)),
            Point(row=float(kr), col=float(kc)),
            Point(row=float(ar), col=float(ac)),
        )
    return img, truth


def write_synthetic_longleg_dicom(
    path: str | Path,
    *,
    patient_id: str = "HKA001",
    **render_kwargs: Any,
) -> Path:
    pixels, _ = render_longleg(**render_kwargs)
    return write_synthetic_dicom(
        path,
        patient_id=patient_id,
        patient_name="Fixture",
        pixels=pixels,
        modality="CR",
        study_description="Standing long-leg",
        series_description="AP",
    )
