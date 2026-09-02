# -*- coding: utf-8 -*-
"""YOLO-format image + label loader for header-region boxes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_yolo_label(path: Path) -> np.ndarray:
    """Return (N, 5) cls xc yc w h. Empty file → one dummy skip row is not used."""
    text = Path(path).read_text(encoding="utf-8").strip()
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def write_yolo_label(path: Path, boxes: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in np.asarray(boxes, dtype=np.float32).reshape(-1, 5):
        cls, xc, yc, w, h = row.tolist()
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> np.ndarray:
    w = max(1, int(width))
    h = max(1, int(height))
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return np.asarray([[0.0, xc, yc, bw, bh]], dtype=np.float32)


class RoiYoloDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        label_dir: Path,
        *,
        img_size: int = 416,
        rotation_deg: float = 0.0,
        hflip: bool = False,
        train: bool = False,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.label_dir = Path(label_dir)
        self.img_size = int(img_size)
        self.rotation_deg = float(rotation_deg)
        self.hflip = bool(hflip)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image: Path) -> Path:
        return self.label_dir / f"{image.stem}.txt"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        pil = Image.open(path).convert("RGB")
        boxes = read_yolo_label(self._label_path(path))
        if self.train and self.hflip and float(np.random.random()) < 0.5:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.size:
                boxes = boxes.copy()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if self.train and self.rotation_deg > 0:
            angle = float((np.random.random() * 2 - 1) * self.rotation_deg)
            pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            if boxes.size and abs(angle) > 1e-3:
                boxes = _rotate_yolo_boxes(boxes, angle)
        pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if boxes.size == 0:
            target = torch.tensor([0.0, 0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        else:
            target = torch.from_numpy(boxes[0].astype(np.float32))
        return tensor, target


def _rotate_yolo_boxes(boxes: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = math.radians(-angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    out = boxes.copy()
    for i, row in enumerate(boxes):
        xc, yc, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        x, y = xc - 0.5, yc - 0.5
        xr = c * x - s * y + 0.5
        yr = s * x + c * y + 0.5
        out[i, 1] = float(np.clip(xr, 0, 1))
        out[i, 2] = float(np.clip(yr, 0, 1))
        out[i, 3] = float(np.clip(w, 1e-3, 1))
        out[i, 4] = float(np.clip(h, 1e-3, 1))
    return out


def list_labeled_images(image_dir: Path, label_dir: Path) -> list[Path]:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    out: list[Path] = []
    if not image_dir.is_dir():
        return out
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        if (label_dir / f"{path.stem}.txt").is_file():
            out.append(path)
    return out
