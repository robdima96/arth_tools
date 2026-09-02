# -*- coding: utf-8 -*-
"""
Build YOLO label files from boxes or cropped-image pairs.

    python -m arth_tools roi-prepare --modality xray --image-dir ... --label-dir ...
    python -m arth_tools roi-prepare --modality xray --box-csv boxes.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from arth_tools.roi.config import load_roi_config
from arth_tools.roi.dataset import write_yolo_label, xyxy_to_yolo


def _write_splits(image_paths: list[Path], split_dir: Path, *, seed: int = 0) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = len(paths)
    n_train = max(1, int(round(0.7 * n))) if n > 3 else n
    n_val = max(0, int(round(0.15 * n))) if n > 3 else 0
    train, val, test = paths[:n_train], paths[n_train : n_train + n_val], paths[n_train + n_val :]
    for name, subset in (("train.txt", train), ("val.txt", val), ("test.txt", test)):
        (split_dir / name).write_text("\n".join(str(p.resolve()) for p in subset) + ("\n" if subset else ""), encoding="utf-8")


def prepare_from_yolo_dir(image_dir: Path, label_dir: Path, split_dir: Path, *, seed: int = 0) -> list[Path]:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    images = [
        p
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        and (label_dir / f"{p.stem}.txt").is_file()
    ]
    _write_splits(images, split_dir, seed=seed)
    print(f"ROI prepare: {len(images)} labelled images → {split_dir}")
    return images


def prepare_from_box_csv(csv_path: Path, image_dir: Path, label_dir: Path, split_dir: Path, *, seed: int = 0) -> list[Path]:
    label_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("filepath") or row.get("path") or row.get("image") or ""
            path = Path(str(raw).strip())
            if not path.is_file():
                cand = image_dir / path.name
                if cand.is_file():
                    path = cand
            if not path.is_file():
                continue
            with Image.open(path) as pil:
                w, h = pil.size
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            boxes = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            dest = label_dir / f"{path.stem}.txt"
            write_yolo_label(dest, boxes)
            written.append(path)
    _write_splits(written, split_dir, seed=seed)
    print(f"ROI prepare: wrote {len(written)} labels from {csv_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare YOLO labels for header-crop training.")
    ap.add_argument("--modality", default="xray")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--label-dir", type=Path)
    ap.add_argument("--box-csv", type=Path, help="CSV with filepath,x1,y1,x2,y2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_roi_config(args.config or args.modality)
    image_dir = args.image_dir or cfg.image_dir()
    label_dir = args.label_dir or cfg.label_dir()
    split_dir = cfg.split_dir()
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.box_csv is not None:
        prepare_from_box_csv(args.box_csv, image_dir, label_dir, split_dir, seed=args.seed)
    else:
        prepare_from_yolo_dir(image_dir, label_dir, split_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
