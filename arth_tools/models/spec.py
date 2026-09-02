"""Model specification schema (architecture, task, freeze criteria).

The live knobs live on the CONTROL BOARD (`arth_tools.training.config`). This
module is the JSON/YAML dump used in run reports and the frozen model bundle.
Architectures are registered in `arth_tools.models.registry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from arth_tools.training.config import TrainingConfig


class Citation(BaseModel):
    title: str = ""
    authors: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    code_url: str = ""
    license: str = ""


class FreezeCriteria(BaseModel):
    primary_metric: str = "accuracy"
    min_value: float = 0.0
    min_epoch: int = 1
    mode: Literal["max", "min"] = "max"


class ModelSpec(BaseModel):
    architecture_id: str = "cnn"
    architecture_name: str = "VGG-inspired CNN (explicit build_cnn_model)"
    version: str = "0.1.0"
    citation: Citation = Field(default_factory=Citation)
    pretrained_id: str = ""
    task_id: str = ""
    task_name: str = ""
    modality: str = "unspecified"
    anatomy: str = "unspecified"
    input_height: int = 224
    input_width: int = 224
    input_channels: int = 3
    replicate_grayscale_to_rgb: bool = True
    num_classes: int = 2
    loss_name: Literal["sparse_categorical_crossentropy", "binary_crossentropy", "mse"] = (
        "sparse_categorical_crossentropy"
    )
    output_type: Literal["softmax", "sigmoid", "linear"] = "softmax"
    primary_metric: str = "accuracy"
    freeze: FreezeCriteria = Field(default_factory=FreezeCriteria)

    def image_size(self) -> tuple[int, int]:
        return (self.input_width, self.input_height)


def spec_from_training_config(cfg: TrainingConfig) -> ModelSpec:
    mode: Literal["max", "min"] = "min" if cfg.freeze_mode == "min" else "max"
    loss: Literal["sparse_categorical_crossentropy", "binary_crossentropy", "mse"]
    if cfg.loss_name in {"sparse_categorical_crossentropy", "binary_crossentropy", "mse"}:
        loss = cfg.loss_name  # type: ignore[assignment]
    else:
        loss = "sparse_categorical_crossentropy"
    output: Literal["softmax", "sigmoid", "linear"]
    if cfg.output_type in {"softmax", "sigmoid", "linear"}:
        output = cfg.output_type  # type: ignore[assignment]
    else:
        output = "softmax"
    return ModelSpec(
        architecture_id=cfg.architecture_id,
        architecture_name=cfg.architecture_name,
        pretrained_id=cfg.pretrained_id,
        task_id=cfg.task_id,
        task_name=cfg.task_name,
        modality=cfg.modality or "unspecified",
        anatomy=cfg.anatomy or "unspecified",
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        input_channels=cfg.input_channels,
        replicate_grayscale_to_rgb=cfg.replicate_grayscale_to_rgb,
        num_classes=cfg.num_classes,
        loss_name=loss,
        output_type=output,
        primary_metric=cfg.primary_metric,
        freeze=FreezeCriteria(
            primary_metric=cfg.primary_metric,
            min_value=cfg.freeze_min_value,
            min_epoch=cfg.freeze_min_epoch,
            mode=mode,
        ),
    )


def load_spec(path: Path) -> ModelSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ModelSpec.model_validate(raw)


def dump_spec(spec: ModelSpec, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(spec.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
