# -*- coding: utf-8 -*-
"""Hip-knee-ankle (HKA) angle on a standing AP long-leg radiograph.

Convention
----------
Points are (row, col) with row increasing downward.

The femoral mechanical axis is hip → knee; the tibial axis is knee → ankle.
``hka_deg`` is 180° minus the deflection between those distal-pointing
axes, so a straight limb is 180°. ``deviation_deg`` is ``hka_deg - 180``
(negative = varus / bow, positive = valgus / knock). Sign uses the
hip–ankle chord: the knee lying toward the image midline is valgus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Point:
    row: float
    col: float

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.row), float(self.col))

    def xy_mm(self, row_mm: float, col_mm: float) -> np.ndarray:
        return np.array([self.col * col_mm, self.row * row_mm], dtype=np.float64)


@dataclass
class LimbHKA:
    laterality: str
    hip: Point
    knee: Point
    ankle: Point
    hka_deg: float
    deviation_deg: float
    alignment: str  # "neutral" | "varus" | "valgus"

    def to_dict(self) -> dict[str, Any]:
        return {
            "laterality": self.laterality,
            "hip_row": self.hip.row,
            "hip_col": self.hip.col,
            "knee_row": self.knee.row,
            "knee_col": self.knee.col,
            "ankle_row": self.ankle.row,
            "ankle_col": self.ankle.col,
            "hka_deg": self.hka_deg,
            "deviation_deg": self.deviation_deg,
            "alignment": self.alignment,
        }


@dataclass
class HKAResult:
    ok: bool
    message: str = ""
    limbs: list[LimbHKA] = field(default_factory=list)
    spacing_mm: tuple[float, float] = (1.0, 1.0)

    def summary_text(self) -> str:
        if not self.limbs:
            return self.message or "HKA: no limbs"
        parts = []
        for limb in self.limbs:
            parts.append(
                f"{limb.laterality} HKA={limb.hka_deg:.1f} deg "
                f"({limb.deviation_deg:+.1f} {limb.alignment})"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "spacing_row_mm": self.spacing_mm[0],
            "spacing_col_mm": self.spacing_mm[1],
            "limbs": [limb.to_dict() for limb in self.limbs],
            "summary": self.summary_text(),
        }


def pixel_spacing_mm(ds: Any) -> tuple[float, float]:
    raw = ds.get("PixelSpacing", None) if hasattr(ds, "get") else None
    if raw is None and hasattr(ds, "get"):
        raw = ds.get("ImagerPixelSpacing", None)
    if raw is None:
        return (1.0, 1.0)
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError, IndexError):
        return (1.0, 1.0)


def hka_from_points(
    hip: Point,
    knee: Point,
    ankle: Point,
    *,
    laterality: str,
    row_mm: float = 1.0,
    col_mm: float = 1.0,
    midline_col: float | None = None,
    neutral_tol_deg: float = 1.5,
) -> LimbHKA:
    hip_xy = hip.xy_mm(row_mm, col_mm)
    knee_xy = knee.xy_mm(row_mm, col_mm)
    ankle_xy = ankle.xy_mm(row_mm, col_mm)
    femur = knee_xy - hip_xy
    tibia = ankle_xy - knee_xy
    n_f = float(np.linalg.norm(femur))
    n_t = float(np.linalg.norm(tibia))
    if n_f < 1e-9 or n_t < 1e-9:
        raise ValueError("Degenerate HKA points (zero-length axis)")
    cos = float(np.clip(np.dot(femur, tibia) / (n_f * n_t), -1.0, 1.0))
    deflection = float(np.degrees(np.arccos(cos)))
    hka = 180.0 - deflection

    chord = ankle_xy - hip_xy
    n_c = float(np.linalg.norm(chord))
    if n_c < 1e-9:
        signed = 0.0
    else:
        # 2D cross (hip→ankle) × (hip→knee): sign says which side the knee sits on.
        hip_to_knee = knee_xy - hip_xy
        cross = chord[0] * hip_to_knee[1] - chord[1] * hip_to_knee[0]
        signed = deflection if cross > 0 else -deflection
        mid = float(midline_col) if midline_col is not None else float(hip.col)
        # Knee toward midline → valgus (+). Image x increases to the right.
        toward_midline = (knee.col - hip.col) * (mid - hip.col)
        if toward_midline < 0:
            signed = -abs(deflection)
        elif toward_midline > 0:
            signed = abs(deflection)
        else:
            signed = abs(deflection) if cross >= 0 else -abs(deflection)

    if abs(signed) <= neutral_tol_deg:
        alignment = "neutral"
        signed = 0.0 if abs(signed) < 1e-9 else signed
    elif signed > 0:
        alignment = "valgus"
    else:
        alignment = "varus"

    return LimbHKA(
        laterality=laterality,
        hip=hip,
        knee=knee,
        ankle=ankle,
        hka_deg=float(hka),
        deviation_deg=float(signed if alignment != "neutral" else hka - 180.0),
        alignment=alignment,
    )


def assign_laterality(hips: list[Point]) -> list[tuple[str, Point]]:
    """AP film: viewer's left is the patient's right (smaller column)."""
    ordered = sorted(hips, key=lambda p: p.col)
    if len(ordered) == 1:
        return [("unknown", ordered[0])]
    out: list[tuple[str, Point]] = []
    if len(ordered) >= 1:
        out.append(("right", ordered[0]))
    if len(ordered) >= 2:
        out.append(("left", ordered[-1]))
    return out
