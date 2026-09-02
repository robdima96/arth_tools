# -*- coding: utf-8 -*-
"""
Staged pixel operators: decode → header YOLO → intensity → CNN tensor.

Each catalog tool declares preprocess_consume so a stage never runs twice.
The same recipe is stored on the model bundle so infer matches train.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

ConsumeMode = Literal["cnn_full", "decode_roi", "external"]
StageName = Literal["decode", "roi", "intensity", "tensor"]

STAGES: tuple[str, ...] = ("decode", "roi", "intensity", "tensor")
CONSUME_STAGES: dict[str, tuple[str, ...]] = {
    "cnn_full": ("decode", "roi", "intensity", "tensor"),
    "decode_roi": ("decode", "roi"),
    "external": (),
}
CNN_BAKED_STAGES: tuple[str, ...] = ("decode", "roi", "intensity")

DICOM_TO_CATALOG: dict[str, str] = {
    "CR": "xray",
    "DX": "xray",
    "RF": "xray",
    "XA": "xray",
    "US": "ultrasound",
    "MR": "mri",
}

MODALITY_INTENSITY: dict[str, dict[str, Any]] = {
    "xray": {
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "window_mode": "voi_or_percentile",
        "foreground_eps": None,
    },
    "ultrasound": {
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "window_mode": "foreground_percentile",
        "foreground_eps": 1e-3,
    },
    "mri": {
        "percentile_low": 1.0,
        "percentile_high": 99.0,
        "window_mode": "percentile",
        "foreground_eps": None,
    },
}

RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def catalog_modality_from_dicom(mod: str | None) -> str:
    key = str(mod or "").strip().upper()
    return DICOM_TO_CATALOG.get(key, str(mod or "").strip().lower() or "xray")


def stages_for_mode(mode: str | None) -> tuple[str, ...]:
    return CONSUME_STAGES.get(str(mode or "cnn_full"), CONSUME_STAGES["cnn_full"])


def remaining_stages(mode: str | None, baked: list[str] | tuple[str, ...] | None) -> list[str]:
    done = {str(s) for s in (baked or []) if str(s)}
    return [s for s in stages_for_mode(mode) if s not in done]


def parse_baked_stages(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(s) for s in raw if str(s)]
    text = str(raw).strip()
    if not text:
        return []
    return [p.strip() for p in text.replace(",", ";").split(";") if p.strip()]


def _as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 2:
        if a.shape[-1] in (3, 4):
            break
        a = np.squeeze(a)
        if a.ndim > 2 and a.shape[0] == 1:
            a = a[0]
        elif a.ndim > 2:
            break
    return a


def ndarray_to_u8(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    a = _as_2d(a)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {a.shape}")
    amin, amax = float(a.min()), float(a.max())
    if amax <= amin + 1e-9:
        v = amin
        if 0.0 <= v <= 255.0:
            return np.full(a.shape, int(np.clip(round(v), 0, 255)), dtype=np.uint8)
        return np.full(a.shape, 128, dtype=np.uint8)
    return ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)


def percentile_to_u8(
    arr: np.ndarray,
    *,
    low: float = 0.5,
    high: float = 99.5,
    foreground_eps: float | None = None,
) -> np.ndarray:
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    a = _as_2d(a)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    sample = a
    if foreground_eps is not None:
        fg = a[a > float(foreground_eps)]
        if fg.size > 16:
            sample = fg
    p_lo, p_hi = np.percentile(sample, [float(low), float(high)])
    if p_hi <= p_lo + 1e-9:
        return ndarray_to_u8(a)
    scaled = np.clip((a - p_lo) / (p_hi - p_lo), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def ndarray_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    u8 = arr if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 2) else ndarray_to_u8(arr)
    if u8.ndim == 3 and u8.shape[-1] == 3:
        return u8.astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def array_to_pil(arr: np.ndarray, *, rgb: bool = True) -> Image.Image:
    if rgb:
        return Image.fromarray(ndarray_to_rgb_u8(arr), mode="RGB")
    u8 = arr if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 2) else ndarray_to_u8(arr)
    return Image.fromarray(u8, mode="L")


@dataclass
class PreprocessRecipe:
    """Deterministic train/serve image operators (saved on the model bundle)."""

    resize_width: int = 224
    resize_height: int = 224
    replicate_grayscale_to_rgb: bool = True
    input_channels: int = 3
    zscore_mean: float | None = None
    zscore_std: float | None = None
    export_mode: str = "dicom_display_u8_rgb"
    preprocess_consume: str = "cnn_full"
    baked_stages: list[str] = field(default_factory=lambda: list(CNN_BAKED_STAGES))
    modality_id: str = ""
    roi_model_id: str = ""
    roi_weight_hash: str | None = None
    roi_conf: float = 0.25
    percentile_low: float | None = None
    percentile_high: float | None = None
    window_mode: str = ""
    frame_policy: str = "middle"

    def image_size(self) -> tuple[int, int]:
        return (int(self.resize_width), int(self.resize_height))

    def consume_mode(self) -> str:
        mode = str(self.preprocess_consume or "cnn_full")
        return mode if mode in CONSUME_STAGES else "cnn_full"

    def catalog_modality(self) -> str:
        return str(self.modality_id or self.roi_model_id or "xray")

    def roi_id(self) -> str:
        return str(self.roi_model_id or self.modality_id or "xray")

    def is_legacy_minmax(self) -> bool:
        return str(self.export_mode or "") == "minmax_u8_rgb"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["baked_stages"] = list(self.baked_stages or [])
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PreprocessRecipe:
        if not raw:
            return cls()
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}  # type: ignore[attr-defined]
        if "baked_stages" in known:
            known["baked_stages"] = parse_baked_stages(known["baked_stages"])
        recipe = cls(**known)
        if recipe.is_legacy_minmax() and "baked_stages" not in (raw or {}):
            recipe.baked_stages = list(CNN_BAKED_STAGES)
        return recipe

    @classmethod
    def from_training_config(cls, cfg: Any, *, zscore: tuple[float, float] | None = None) -> PreprocessRecipe:
        mean = std = None
        if zscore is not None:
            mean, std = float(zscore[0]), float(zscore[1])
        modality = str(getattr(cfg, "modality", "") or "")
        consume = str(getattr(cfg, "preprocess_consume", "") or "cnn_full")
        defaults = MODALITY_INTENSITY.get(modality, MODALITY_INTENSITY["xray"])
        p_lo = getattr(cfg, "percentile_low", None)
        p_hi = getattr(cfg, "percentile_high", None)
        window_mode = str(getattr(cfg, "window_mode", "") or "") or str(defaults["window_mode"])
        roi_id = str(getattr(cfg, "roi_model_id", "") or "") or modality
        baked = parse_baked_stages(getattr(cfg, "baked_stages", None))
        if consume == "cnn_full" and not baked:
            baked = list(CNN_BAKED_STAGES)
        export_mode = str(getattr(cfg, "export_mode", "") or "") or "dicom_display_u8_rgb"
        hash_val = None
        try:
            from arth_tools.roi.infer import weight_hash_for

            hash_val = weight_hash_for(roi_id or modality)
        except Exception:
            hash_val = None
        return cls(
            resize_width=int(cfg.input_width),
            resize_height=int(cfg.input_height),
            replicate_grayscale_to_rgb=bool(cfg.replicate_grayscale_to_rgb),
            input_channels=int(cfg.input_channels),
            zscore_mean=mean,
            zscore_std=std,
            export_mode=export_mode,
            preprocess_consume=consume,
            baked_stages=baked,
            modality_id=modality,
            roi_model_id=roi_id,
            roi_weight_hash=hash_val,
            roi_conf=float(getattr(cfg, "roi_conf", 0.25) or 0.25),
            percentile_low=float(p_lo) if p_lo is not None else float(defaults["percentile_low"]),
            percentile_high=float(p_hi) if p_hi is not None else float(defaults["percentile_high"]),
            window_mode=window_mode,
            frame_policy=str(getattr(cfg, "frame_policy", "") or "middle"),
        )


def _pick_frame(arr: np.ndarray, *, policy: str = "middle") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        return a
    if a.ndim >= 3:
        n = int(a.shape[0])
        if n <= 1:
            return _pick_frame(np.squeeze(a, axis=0), policy=policy)
        idx = 0 if policy == "first" else n // 2
        return _pick_frame(a[idx], policy=policy)
    return np.squeeze(a)


def decode_dicom(ds: Any, *, apply_voi: bool = False, frame_policy: str = "middle") -> tuple[np.ndarray, dict[str, Any]]:
    """Display-referred array + metadata. Photometric invert lives only here."""
    if not hasattr(ds, "pixel_array"):
        raise ValueError("No pixel data")
    raw = ds.pixel_array
    arr = _pick_frame(raw, policy=frame_policy)

    window_applied = False
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut

        arr = apply_modality_lut(arr, ds)
        has_window = ds.get("WindowCenter") is not None and ds.get("WindowWidth") is not None
        has_voi = ds.get("VOILUTSequence") is not None
        if apply_voi and (has_window or has_voi):
            arr = apply_voi_lut(arr, ds)
            window_applied = True
    except Exception:
        arr = np.asarray(arr)

    arr = np.asarray(arr, dtype=np.float64)
    photo = str(ds.get("PhotometricInterpretation", "MONOCHROME2") or "MONOCHROME2").upper()
    if "PALETTE COLOR" in photo:
        try:
            from pydicom.pixels import apply_color_lut

            arr = np.asarray(apply_color_lut(ds.pixel_array, ds), dtype=np.float64)
            arr = _pick_frame(arr, policy=frame_policy)
            photo = "RGB"
        except Exception:
            pass
    elif photo.startswith("YBR"):
        try:
            from pydicom.pixels.processing import convert_color_space

            rgb = convert_color_space(arr.astype(np.uint8, copy=False), photo, "RGB")
            arr = np.asarray(rgb, dtype=np.float64)
            photo = "RGB"
        except Exception:
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        pass
    else:
        arr = _as_2d(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D pixels, got shape {arr.shape}")
        if "MONOCHROME1" in photo:
            arr = float(np.nanmax(arr)) - arr

    bits = ds.get("BitsStored", ds.get("BitsAllocated"))
    try:
        bits_stored = int(bits) if bits is not None else None
    except (TypeError, ValueError):
        bits_stored = None
    meta = {
        "photometric": photo,
        "bits_stored": bits_stored,
        "manufacturer": str(ds.get("Manufacturer", "") or ""),
        "manufacturer_model": str(ds.get("ManufacturerModelName", "") or ""),
        "window_applied": window_applied,
        "modality": str(ds.get("Modality", "") or ""),
        "catalog_modality": catalog_modality_from_dicom(str(ds.get("Modality", "") or "")),
    }
    return arr, meta


def intensity_to_u8(arr: np.ndarray, recipe: PreprocessRecipe, *, bits_stored: int | None = None) -> np.ndarray:
    if recipe.is_legacy_minmax():
        return ndarray_to_u8(arr)
    modality = recipe.catalog_modality()
    defaults = MODALITY_INTENSITY.get(modality, MODALITY_INTENSITY["xray"])
    low = float(recipe.percentile_low if recipe.percentile_low is not None else defaults["percentile_low"])
    high = float(recipe.percentile_high if recipe.percentile_high is not None else defaults["percentile_high"])
    window_mode = str(recipe.window_mode or defaults["window_mode"])
    eps = defaults.get("foreground_eps")
    a = np.asarray(arr, dtype=np.float64)

    if window_mode == "foreground_percentile":
        if bits_stored is not None and int(bits_stored) <= 8:
            amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
            if amin >= 0.0 and amax <= 255.0 + 1e-6:
                plane = _as_2d(a)
                if plane.ndim == 3:
                    plane = 0.299 * plane[..., 0] + 0.587 * plane[..., 1] + 0.114 * plane[..., 2]
                return np.clip(np.rint(plane), 0, 255).astype(np.uint8)
        return percentile_to_u8(a, low=low, high=high, foreground_eps=float(eps or 1e-3))
    return percentile_to_u8(a, low=low, high=high, foreground_eps=None)


def _apply_roi(arr: np.ndarray, recipe: PreprocessRecipe) -> np.ndarray:
    from arth_tools.roi.infer import crop_image

    return crop_image(arr, recipe.roi_id(), conf=float(recipe.roi_conf))


def apply_stages_from_dataset(ds: Any, recipe: PreprocessRecipe) -> np.ndarray:
    """Run unbaked stages for a live DICOM dataset. external → unmodified pixel_array."""
    mode = recipe.consume_mode()
    if mode == "external":
        return np.asarray(ds.pixel_array)
    todo = remaining_stages(mode, [])
    apply_voi = "intensity" in stages_for_mode(mode)
    if "decode" in todo:
        arr, meta = decode_dicom(ds, apply_voi=apply_voi, frame_policy=recipe.frame_policy)
    else:
        arr = np.asarray(ds.pixel_array, dtype=np.float64)
        meta = {"bits_stored": ds.get("BitsStored")}
    if "roi" in todo:
        arr = _apply_roi(arr, recipe)
    if "intensity" in todo:
        arr = intensity_to_u8(arr, recipe, bits_stored=meta.get("bits_stored") if isinstance(meta, dict) else None)
    if "tensor" in todo:
        pil = array_to_pil(arr, rgb=bool(recipe.replicate_grayscale_to_rgb))
        return pil_to_float_hwc(pil, recipe)
    return arr


def dicom_to_display_u8(ds: Any, recipe: PreprocessRecipe) -> tuple[np.ndarray, dict[str, Any]]:
    """decode + roi + intensity → uint8 plane (CNN export)."""
    display = PreprocessRecipe.from_dict(recipe.to_dict())
    display.preprocess_consume = "cnn_full"
    display.baked_stages = []
    apply_voi = True
    arr, meta = decode_dicom(ds, apply_voi=apply_voi, frame_policy=display.frame_policy)
    arr = _apply_roi(arr, display)
    u8 = intensity_to_u8(arr, display, bits_stored=meta.get("bits_stored"))
    meta["baked_stages"] = list(CNN_BAKED_STAGES)
    meta["window_applied"] = bool(meta.get("window_applied"))
    return u8, meta


def pil_to_float_hwc(pil: Image.Image, recipe: PreprocessRecipe) -> np.ndarray:
    mode = "RGB" if recipe.replicate_grayscale_to_rgb else "L"
    pil = pil.convert(mode)
    pil = pil.resize(recipe.image_size(), Image.BILINEAR)
    np_im = np.asarray(pil, dtype=np.float32) / 255.0
    if np_im.ndim == 2:
        np_im = np_im[:, :, None]
    if recipe.zscore_mean is not None and recipe.zscore_std is not None:
        std = float(recipe.zscore_std) if float(recipe.zscore_std) > 1e-8 else 1e-8
        np_im = (np_im - float(recipe.zscore_mean)) / std
        np_im = np.nan_to_num(np_im, nan=0.0, posinf=0.0, neginf=0.0)
    return np_im


def dicom_pixels_to_pil(arr: np.ndarray, recipe: PreprocessRecipe) -> Image.Image:
    rgb = recipe.replicate_grayscale_to_rgb or str(recipe.export_mode).endswith("rgb")
    if recipe.is_legacy_minmax():
        return array_to_pil(arr, rgb=rgb)
    u8 = intensity_to_u8(arr, recipe)
    return array_to_pil(u8, rgb=rgb)


def load_image_array(path: Path, recipe: PreprocessRecipe) -> np.ndarray:
    """PNG or DICOM path → HWC float32. Raster files skip baked stages."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        pil = Image.open(path)
        return pil_to_float_hwc(pil, recipe)
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"No pixel data: {path}")
    if recipe.consume_mode() == "external":
        return np.asarray(ds.pixel_array)
    if recipe.is_legacy_minmax():
        pil = dicom_pixels_to_pil(ds.pixel_array, recipe)
        return pil_to_float_hwc(pil, recipe)
    live = PreprocessRecipe.from_dict(recipe.to_dict())
    live.baked_stages = []
    return apply_stages_from_dataset(ds, live)
