# -*- coding: utf-8 -*-
"""Header YOLO infrastructure (identity until weights exist)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arth_tools.roi.config import load_roi_config
from arth_tools.roi.dataset import RoiYoloDataset, write_yolo_label, xyxy_to_yolo
from arth_tools.roi.infer import crop_image, crop_xyxy, weight_hash_for
from arth_tools.roi.model import YoloV3Tiny, decode_heads, yolo_loss
from arth_tools.roi.parse_cfg import cfg_num_classes, parse_cfg
from arth_tools.roi.train import train_roi
from arth_tools.paths import PACKAGE_DIR


def test_parse_cfg_one_class() -> None:
    cfg_path = PACKAGE_DIR / "roi" / "cfgs" / "yolov3-tiny-1cls.cfg"
    blocks = parse_cfg(cfg_path)
    assert cfg_num_classes(blocks) == 1
    assert any(b.get("type") == "yolo" for b in blocks)


def test_yolo_forward_shapes() -> None:
    model = YoloV3Tiny(num_classes=1)
    x = torch.zeros(1, 3, 416, 416)
    large, small = model(x)
    assert large.shape[0] == 1
    assert small.shape[0] == 1
    assert large.shape[1] == 18
    assert small.shape[1] == 18
    boxes = decode_heads((large, small), img_size=416, anchors=(10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319))
    assert boxes.shape[-1] == 6


def test_crop_identity_and_box() -> None:
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.testing.assert_array_equal(crop_image(arr, "mri"), arr)
    np.testing.assert_array_equal(crop_image(arr, "ultrasound"), arr)
    cut = crop_xyxy(arr, (2, 2, 6, 6))
    assert cut.shape == (4, 4)
    assert weight_hash_for("xray") is None


def test_roi_configs_exist() -> None:
    for mid in ("xray", "ultrasound", "mri"):
        cfg = load_roi_config(mid)
        assert cfg.modality == mid
        assert cfg.best_weights().parent.name == mid


def test_train_one_step(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    lab_dir = tmp_path / "labels"
    img_dir.mkdir()
    lab_dir.mkdir()
    canvas = np.zeros((32, 32), dtype=np.uint8)
    canvas[8:24, 8:24] = 200
    Image.fromarray(canvas, mode="L").convert("RGB").save(img_dir / "a.png")
    write_yolo_label(lab_dir / "a.txt", xyxy_to_yolo(8, 8, 24, 24, 32, 32))
    cfg = load_roi_config("xray")
    cfg.data_root = tmp_path
    cfg.weights_dir = tmp_path / "weights"
    cfg.img_size = 64
    cfg.batch_size = 1
    cfg.rotation_deg = 0.0
    (tmp_path / "xray" / "images").mkdir(parents=True)
    (tmp_path / "xray" / "labels").mkdir(parents=True)
    Image.fromarray(canvas, mode="L").convert("RGB").save(tmp_path / "xray" / "images" / "a.png")
    write_yolo_label(tmp_path / "xray" / "labels" / "a.txt", xyxy_to_yolo(8, 8, 24, 24, 32, 32))
    path = train_roi(cfg, epochs=1, device="cpu")
    assert path.is_file()
    ds = RoiYoloDataset([tmp_path / "xray" / "images" / "a.png"], tmp_path / "xray" / "labels", img_size=64)
    image, target = ds[0]
    model = YoloV3Tiny(1)
    loss = yolo_loss(model(image.unsqueeze(0)), target.unsqueeze(0), img_size=64, anchors=cfg.anchors)
    assert torch.isfinite(loss)


def test_forced_box_is_non_identity() -> None:
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    cropped = crop_image(arr, "xray", box=(1, 2, 8, 9))
    assert cropped.shape == (7, 7)
    assert not np.array_equal(cropped, arr)
