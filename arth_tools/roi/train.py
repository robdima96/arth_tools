# -*- coding: utf-8 -*-
"""
Supervised YOLOv3-Tiny train/eval for header crops.

    python -m arth_tools roi-train --modality xray
    python -m arth_tools roi-eval  --modality xray
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from arth_tools.roi.config import RoiConfig, load_roi_config
from arth_tools.roi.dataset import RoiYoloDataset, list_labeled_images
from arth_tools.roi.model import YoloV3Tiny, decode_heads, yolo_loss
from arth_tools.roi.parse_cfg import cfg_num_classes, parse_cfg


def _split_paths(cfg: RoiConfig) -> tuple[list[Path], list[Path]]:
    split_dir = cfg.split_dir()
    train_list = split_dir / "train.txt"
    val_list = split_dir / "val.txt"
    if train_list.is_file():
        train_paths = [Path(p.strip()) for p in train_list.read_text(encoding="utf-8").splitlines() if p.strip()]
        val_paths = []
        if val_list.is_file():
            val_paths = [Path(p.strip()) for p in val_list.read_text(encoding="utf-8").splitlines() if p.strip()]
        return train_paths, val_paths
    images = list_labeled_images(cfg.image_dir(), cfg.label_dir())
    n_val = max(1, len(images) // 5) if len(images) >= 5 else 0
    return images[n_val:], images[:n_val]


def train_roi(cfg: RoiConfig, *, epochs: int | None = None, device: str = "cpu") -> Path:
    cfg.weights_dir.mkdir(parents=True, exist_ok=True)
    train_paths, val_paths = _split_paths(cfg)
    if not train_paths:
        raise FileNotFoundError(
            f"No labelled ROI images under {cfg.image_dir()} (need sibling labels in {cfg.label_dir()})."
        )
    n_cls = 1
    if cfg.cfg_path.is_file():
        n_cls = cfg_num_classes(parse_cfg(cfg.cfg_path), default=1)
    model = YoloV3Tiny(num_classes=n_cls)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    ds = RoiYoloDataset(
        train_paths,
        cfg.label_dir(),
        img_size=cfg.img_size,
        rotation_deg=cfg.rotation_deg,
        hflip=cfg.hflip,
        train=True,
    )
    loader = DataLoader(ds, batch_size=max(1, int(cfg.batch_size)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    n_epochs = int(epochs if epochs is not None else cfg.epochs)
    best_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for images, targets in loader:
            images = images.to(dev)
            targets = targets.to(dev)
            heads = model(images)
            loss = yolo_loss(heads, targets, img_size=cfg.img_size, anchors=cfg.anchors)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        mean_loss = running / max(n_batches, 1)
        blob = {"model": model.state_dict(), "modality": cfg.modality, "epoch": epoch, "loss": mean_loss}
        torch.save(blob, cfg.last_weights())
        if mean_loss <= best_loss:
            best_loss = mean_loss
            torch.save(blob, cfg.best_weights())
        print(f"epoch {epoch + 1}/{n_epochs} loss={mean_loss:.4f}")
    print(f"Wrote {cfg.best_weights()}")
    return cfg.best_weights()


def eval_roi(cfg: RoiConfig, *, device: str = "cpu") -> dict[str, float]:
    _, val_paths = _split_paths(cfg)
    paths = val_paths or list_labeled_images(cfg.image_dir(), cfg.label_dir())
    if not paths:
        raise FileNotFoundError(f"No ROI val images under {cfg.image_dir()}")
    if not cfg.best_weights().is_file():
        return {"n": 0.0, "mean_iou": 0.0, "recall": 0.0}
    model = YoloV3Tiny(num_classes=1)
    blob = torch.load(cfg.best_weights(), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()
    ds = RoiYoloDataset(paths, cfg.label_dir(), img_size=cfg.img_size, train=False)
    ious = []
    with torch.no_grad():
        for image, target in ds:
            heads = model(image.unsqueeze(0))
            boxes = decode_heads(heads, img_size=cfg.img_size, anchors=cfg.anchors)[0]
            score = boxes[:, 4] * boxes[:, 5]
            best = int(torch.argmax(score).item())
            pred = boxes[best, :4]
            gx = float(target[1]) * cfg.img_size
            gy = float(target[2]) * cfg.img_size
            gw = float(target[3]) * cfg.img_size
            gh = float(target[4]) * cfg.img_size
            gt = torch.tensor([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2])
            tl = torch.maximum(pred[:2], gt[:2])
            br = torch.minimum(pred[2:], gt[2:])
            wh = (br - tl).clamp(min=0)
            inter = float(wh[0] * wh[1])
            area_p = float((pred[2] - pred[0]).clamp(min=0) * (pred[3] - pred[1]).clamp(min=0))
            area_g = float((gt[2] - gt[0]).clamp(min=0) * (gt[3] - gt[1]).clamp(min=0))
            ious.append(inter / (area_p + area_g - inter + 1e-8))
    mean_iou = float(sum(ious) / max(len(ious), 1))
    recall = float(sum(1 for v in ious if v >= 0.5) / max(len(ious), 1))
    print(f"n={len(ious)} mean_iou={mean_iou:.3f} recall@0.5={recall:.3f}")
    return {"n": float(len(ious)), "mean_iou": mean_iou, "recall": recall}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a per-modality header YOLO (internal).")
    ap.add_argument("--modality", default="xray", help="xray | ultrasound | mri")
    ap.add_argument("--config", type=Path, help="Optional configs/roi_<modality>.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = load_roi_config(args.config or args.modality)
    train_roi(cfg, epochs=args.epochs, device=args.device)
    return 0


def main_eval(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a per-modality header YOLO (internal).")
    ap.add_argument("--modality", default="xray")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = load_roi_config(args.config or args.modality)
    eval_roi(cfg, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
