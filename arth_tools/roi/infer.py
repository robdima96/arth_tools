# -*- coding: utf-8 -*-
"""Top-1 header-region crop. Missing weights → identity (never a guessed box)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from arth_tools.roi.config import MODALITIES, RoiConfig, load_roi_config
from arth_tools.roi.model import YoloV3Tiny, decode_heads

_MODELS: dict[str, tuple[YoloV3Tiny, RoiConfig]] = {}


def weights_path_for(modality: str) -> Path:
    cfg = load_roi_config(str(modality).strip().lower() or "xray")
    return cfg.best_weights()


def weight_hash_for(modality: str) -> str | None:
    path = weights_path_for(modality)
    if not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def crop_xyxy(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    a = np.asarray(arr)
    h = int(a.shape[0])
    w = int(a.shape[1])
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return a
    return a[y1:y2, x1:x2, ...] if a.ndim >= 2 else a


def _load_detector(modality: str) -> tuple[YoloV3Tiny, RoiConfig] | None:
    mid = str(modality or "xray").strip().lower()
    if mid not in MODALITIES:
        mid = "xray"
    cfg = load_roi_config(mid)
    path = cfg.best_weights()
    if not path.is_file():
        return None
    key = f"{mid}:{path}:{path.stat().st_mtime}"
    cached = _MODELS.get(mid)
    if cached is not None and key in _MODELS:
        return cached
    model = YoloV3Tiny(num_classes=1)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()
    pair = (model, cfg)
    _MODELS[mid] = pair
    _MODELS[key] = pair
    return pair


def detect_top1(
    arr: np.ndarray,
    modality: str,
    *,
    conf: float | None = None,
) -> tuple[tuple[int, int, int, int], float] | None:
    loaded = _load_detector(modality)
    if loaded is None:
        return None
    model, cfg = loaded
    thresh = float(conf if conf is not None else cfg.conf_threshold)
    plane = np.asarray(arr)
    if plane.ndim == 2:
        rgb = np.stack([plane, plane, plane], axis=-1)
    elif plane.ndim == 3 and plane.shape[-1] >= 3:
        rgb = plane[..., :3]
    else:
        return None
    rgb = rgb.astype(np.float32)
    rmin, rmax = float(np.nanmin(rgb)), float(np.nanmax(rgb))
    if rmax > 1.5:
        rgb = (rgb - rmin) / (rmax - rmin + 1e-8)
    from PIL import Image

    pil = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    resized = pil.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        heads = model(tensor)
        boxes = decode_heads(heads, img_size=cfg.img_size, anchors=cfg.anchors)[0]
    score = boxes[:, 4] * boxes[:, 5]
    best = int(torch.argmax(score).item())
    if float(score[best].item()) < thresh:
        return None
    x1, y1, x2, y2 = boxes[best, :4].tolist()
    sx = float(arr.shape[1]) / float(cfg.img_size)
    sy = float(arr.shape[0]) / float(cfg.img_size)
    box = (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )
    return box, float(score[best].item())


def crop_image(
    arr: np.ndarray,
    modality: str,
    *,
    conf: float | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Crop the image region. No weights / low conf / failed detect → identity."""
    a = np.asarray(arr)
    if box is not None:
        return crop_xyxy(a, box)
    found = detect_top1(a, modality, conf=conf)
    if found is None:
        return a
    return crop_xyxy(a, found[0])
