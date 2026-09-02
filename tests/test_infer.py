# -*- coding: utf-8 -*-
"""Bundle reload and unlabeled infer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from arth_tools.inference.infer import collect_image_items, run_infer
from arth_tools.models.load import load_classifier, write_run_bundle
from arth_tools.training.backbones import build_model
from arth_tools.training.config import TrainingConfig


def _tiny_cfg() -> TrainingConfig:
    return TrainingConfig(
        architecture_id="cnn",
        num_classes=2,
        input_height=16,
        input_width=16,
        num_kernels=4,
        device="cpu",
        train_zscore=False,
        batch_size=2,
        dataloader_num_workers=0,
    )


def test_write_and_load_bundle(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = build_model(cfg)
    dest = write_run_bundle(run_dir, cfg, model=model)
    assert (dest / "bundle.json").is_file()
    assert (dest / "state_dict.pt").is_file()
    assert (dest / "preprocess.json").is_file()
    assert (dest / "label_map.json").is_file()
    loaded = load_classifier(run_dir)
    assert loaded.cfg.architecture_id == "cnn"
    assert loaded.cfg.num_classes == 2
    loaded_bundle = load_classifier(dest)
    assert loaded_bundle.weights_path.is_file()


def test_infer_unlabeled_png(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_bundle(run_dir, cfg, model=build_model(cfg))
    img_dir = tmp_path / "images" / "P001"
    img_dir.mkdir(parents=True)
    Image.fromarray(np.full((8, 8), 80, dtype=np.uint8), mode="L").convert("RGB").save(img_dir / "a.png")
    out_csv = tmp_path / "predictions.csv"
    summary = run_infer(run_dir, items=collect_image_items(img_dir.parent), out_csv=out_csv, device="cpu")
    assert summary["n_images"] == 1
    df = pd.read_csv(out_csv)
    assert "y_pred" in df.columns
    assert "y_true" not in df.columns
    assert "score_0" in df.columns
