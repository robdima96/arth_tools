# -*- coding: utf-8 -*-
"""Registry and build_model dispatch."""

from __future__ import annotations

import pytest

from arth_tools.models.registry import get_builder, normalize_architecture_id, registered_architectures
from arth_tools.training.backbones import build_model
from arth_tools.training.config import TrainingConfig


def test_cnn_aliases_normalize() -> None:
    assert normalize_architecture_id("tiny_cnn") == "cnn"
    assert normalize_architecture_id("CNN") == "cnn"
    assert "cnn" in registered_architectures()
    assert "resnet50" in registered_architectures()


def test_build_model_cnn() -> None:
    cfg = TrainingConfig(
        architecture_id="tiny_cnn",
        num_classes=3,
        input_height=16,
        input_width=16,
        num_kernels=4,
        device="cpu",
    )
    model = build_model(cfg)
    assert model is not None
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_unknown_architecture_raises() -> None:
    cfg = TrainingConfig(architecture_id="vit_never_registered", num_classes=2)
    with pytest.raises(ValueError, match="Unknown architecture_id"):
        build_model(cfg)
    with pytest.raises(ValueError, match="Unknown architecture_id"):
        get_builder("not_a_model")
