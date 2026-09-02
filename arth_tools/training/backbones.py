# -*- coding: utf-8 -*-
"""
Trainable backbones.

Default: explicit VGG-inspired CNN (Keras-style block layout, ported to PyTorch).
Optional: Hugging Face ResNet-50.

Add a new architecture by registering it in ``register_default_architectures()``
(or calling ``arth_tools.models.registry.register`` from your module). Train,
eval, and infer all construct models through ``build_model`` — do not add
``if architecture_id`` branches in those files.
"""

from __future__ import annotations

import torch.nn as nn

from arth_tools.models.registry import freeze_fn_for, get_builder, register
from arth_tools.training.config import TrainingConfig

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


def _activation(name: str) -> nn.Module:
    name_l = (name or "relu").lower()
    if name_l in {"relu"}:
        return nn.ReLU(inplace=True)
    if name_l in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(0.1, inplace=True)
    if name_l in {"gelu"}:
        return nn.GELU()
    if name_l in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Unknown activation_func: {name}")


def logits_dim(cfg: TrainingConfig) -> int:
    if cfg.loss_name == "binary_crossentropy" and cfg.output_type == "sigmoid":
        return 1
    if cfg.loss_name == "mse" or cfg.output_type == "linear":
        return 1
    return int(cfg.num_classes)


class ExplicitCNN(nn.Module):
    """VGG-16 inspired CNN. Constructed by build_cnn_model() so every block is visible."""

    def __init__(self, features: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, pixel_values):  # type: ignore[no-untyped-def]
        x = self.features(pixel_values)
        x = x.flatten(1)
        return self.classifier(x)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = trainable


def _conv_bn(
    in_ch: int,
    out_ch: int,
    kernel_size: tuple[int, int],
    conv_stride: tuple[int, int],
    activation_func: str,
) -> list[nn.Module]:
    """One Conv2D(activation=...) + BatchNormalization, matching the Keras order."""
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=conv_stride, padding="same", bias=True),
        _activation(activation_func),
        nn.BatchNorm2d(out_ch),
    ]


def build_cnn_model(
    output_type: str,
    num_kernels: int,
    kernel_size: tuple[int, int],
    conv_stride: tuple[int, int],
    pool_stride: tuple[int, int],
    activation_func: str,
    input_height: int,
    input_width: int,
    *,
    input_channels: int = 3,
    num_classes: int = 2,
    drop: float = 0.2,
    spatial_drop: float = 0.2,
    n_logits: int | None = None,
) -> ExplicitCNN:
    """
    Build VGG-16 inspired CNN architecture with a flexible output head.

    Input is channel-first (B, C, H, W). Keras used (H, W, C); SpatialDropout2D
    is Dropout2d here; GlobalAveragePooling2D is AdaptiveAvgPool2d(1).

    The last Dense has no sigmoid/softmax — train.py uses *_with_logits losses.
    """
    del input_height, input_width  # size is whatever the loader yields; convs are fully conv
    if n_logits is None:
        if output_type == "softmax":
            n_logits = int(num_classes)
        else:
            n_logits = 1

    k = tuple(kernel_size)
    cs = tuple(conv_stride)
    ps = tuple(pool_stride)

    # Layer 1: 2 conv + max pool
    layer1: list[nn.Module] = []
    layer1.extend(_conv_bn(input_channels, num_kernels, k, cs, activation_func))
    layer1.extend(_conv_bn(num_kernels, num_kernels, k, cs, activation_func))
    layer1.append(nn.MaxPool2d(kernel_size=2, stride=ps, padding=0))
    layer1.append(nn.Dropout2d(spatial_drop))

    # Layer 2: 2 conv + max pool
    layer2: list[nn.Module] = []
    layer2.extend(_conv_bn(num_kernels, num_kernels * 2, k, cs, activation_func))
    layer2.extend(_conv_bn(num_kernels * 2, num_kernels * 2, k, cs, activation_func))
    layer2.append(nn.MaxPool2d(kernel_size=2, stride=ps, padding=0))
    layer2.append(nn.Dropout2d(spatial_drop))

    # Layer 3: 2 conv + max pool
    layer3: list[nn.Module] = []
    layer3.extend(_conv_bn(num_kernels * 2, num_kernels * 4, k, cs, activation_func))
    layer3.extend(_conv_bn(num_kernels * 4, num_kernels * 4, k, cs, activation_func))
    layer3.append(nn.MaxPool2d(kernel_size=2, stride=ps, padding=0))
    layer3.append(nn.Dropout2d(spatial_drop))

    # Layer 4: 2 conv + max pool
    layer4: list[nn.Module] = []
    layer4.extend(_conv_bn(num_kernels * 4, num_kernels * 8, k, cs, activation_func))
    layer4.extend(_conv_bn(num_kernels * 8, num_kernels * 8, k, cs, activation_func))
    layer4.append(nn.MaxPool2d(kernel_size=2, stride=ps, padding=0))
    layer4.append(nn.Dropout2d(spatial_drop))

    # Global average pooling
    features = nn.Sequential(
        *layer1,
        *layer2,
        *layer3,
        *layer4,
        nn.AdaptiveAvgPool2d(1),
    )

    # Output layer options (Dense stacks)
    head: list[nn.Module] = []
    if output_type == "sigmoid":
        head.extend(
            [
                nn.Linear(num_kernels * 8, 256),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(256, 64),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(64, n_logits),
            ]
        )
    elif output_type == "softmax":
        head.extend(
            [
                nn.Linear(num_kernels * 8, 512),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(512, 256),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(256, n_logits),
            ]
        )
    elif output_type == "linear":
        head.extend(
            [
                nn.Linear(num_kernels * 8, 256),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(256, 64),
                _activation(activation_func),
                nn.Dropout(drop),
                nn.Linear(64, n_logits),
            ]
        )
    else:
        raise ValueError(f"Unknown output type: {output_type}")

    return ExplicitCNN(features, nn.Sequential(*head))


def build_classifier_stack(hidden_last: int, num_logits: int, cfg: TrainingConfig) -> nn.Sequential:
    """Optional MLP on a Hugging Face ResNet classifier (ML Pipeline US DICOMs)."""
    mlp = int(cfg.head_mlp_hidden)
    dout = float(cfg.dropout_classifier_head)
    layers: list[nn.Module] = [nn.Flatten()]
    if mlp > 0:
        layers.extend(
            [
                nn.Linear(hidden_last, mlp),
                nn.ReLU(inplace=True),
                nn.Dropout(dout),
                nn.Linear(mlp, num_logits),
            ]
        )
    else:
        layers.append(nn.Linear(hidden_last, num_logits))
    return nn.Sequential(*layers)


def _build_cnn(cfg: TrainingConfig) -> ExplicitCNN:
    return build_cnn_model(
        output_type=cfg.output_type,
        num_kernels=cfg.num_kernels,
        kernel_size=cfg.kernel_size,
        conv_stride=cfg.conv_stride,
        pool_stride=cfg.pool_stride,
        activation_func=cfg.activation_func,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        input_channels=cfg.input_channels,
        num_classes=cfg.num_classes,
        drop=cfg.drop,
        spatial_drop=cfg.spatial_drop,
        n_logits=logits_dim(cfg),
    )


def _build_resnet50(cfg: TrainingConfig) -> nn.Module:
    from transformers import ResNetForImageClassification

    n_logits = logits_dim(cfg)
    model = ResNetForImageClassification.from_pretrained(
        cfg.pretrained_id or "microsoft/resnet-50",
        num_labels=n_logits,
        ignore_mismatched_sizes=True,
    )
    model.classifier = build_classifier_stack(model.config.hidden_sizes[-1], n_logits, cfg)
    return model


def _freeze_cnn(model: nn.Module, trainable: bool) -> None:
    setter = getattr(model, "set_backbone_trainable", None)
    if callable(setter):
        setter(trainable)
        return
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = trainable


def _freeze_resnet(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "resnet"):
        for p in model.resnet.parameters():
            p.requires_grad = trainable
        return
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = trainable


def register_default_architectures() -> None:
    register("cnn", _build_cnn, aliases=("tiny_cnn", "tinycnn", "vgg_cnn"), freeze_fn=_freeze_cnn)
    register(
        "resnet50",
        _build_resnet50,
        aliases=("microsoft/resnet-50",),
        freeze_fn=_freeze_resnet,
    )


register_default_architectures()


def build_model(cfg: TrainingConfig) -> nn.Module:
    return get_builder(cfg.architecture_id)(cfg)


def forward_logits(model: nn.Module, pixel_values):  # type: ignore[no-untyped-def]
    out = model(pixel_values)
    if hasattr(out, "logits"):
        return out.logits
    return out


def set_backbone_trainable(model: nn.Module, trainable: bool, architecture_id: str | None = None) -> None:
    if architecture_id:
        fn = freeze_fn_for(architecture_id)
        if fn is not None:
            fn(model, trainable)
            return
    setter = getattr(model, "set_backbone_trainable", None)
    if callable(setter):
        setter(trainable)
        return
    if hasattr(model, "resnet"):
        for p in model.resnet.parameters():
            p.requires_grad = trainable
        return
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = trainable
