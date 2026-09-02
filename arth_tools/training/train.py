# -*- coding: utf-8 -*-
"""
Train a spec-defined model from CSV manifests.

Edit arth_tools/training/config.py (the CONTROL BOARD), then::

    python -m arth_tools.training.train

Smoke tests pass a TrainingConfig with fixture paths instead of E:\\ manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from arth_tools.data.dataset import ManifestClassificationDataset, compute_train_zscore
from arth_tools.models.spec import dump_spec, spec_from_training_config
from arth_tools.training.backbones import (
    build_cnn_model,
    build_model,
    forward_logits,
    set_backbone_trainable,
)
from arth_tools.training.metrics import batch_loss, run_eval
from arth_tools.training.config import (
    TrainingConfig,
    load_config_yaml,
    new_run_id,
    save_hyperparameters_txt,
)
from arth_tools.training.freeze import decide_freeze, freeze_if_criteria_met, write_freeze_decision
from arth_tools.training.logging_setup import setup_run_logging
from arth_tools.training.plots import plot_curves

# ============================================================================
# SEEDS / DEVICE
# ============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def environment_versions() -> dict[str, str]:
    import importlib.metadata as md
    import platform

    names = [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "pydantic",
        "optuna",
        "pydicom",
        "pillow",
        "pyyaml",
        "matplotlib",
        "transformers",
    ]
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not-installed"
    return out


# ============================================================================
# CLASS WEIGHTS / OVERSAMPLING
# ============================================================================


def print_dataset_info(name: str, labels: list[int] | np.ndarray) -> None:
    y = np.asarray(labels).ravel()
    counts = Counter(y.tolist())
    total = len(y)
    print(f"\n{name} set:")
    print(f"  Samples: {total}")
    for cls, cnt in sorted(counts.items()):
        frac = (cnt / total) if total else 0.0
        print(f"  Class {cls}: {cnt} ({frac:.3f})")


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    print("\n" + "=" * 70)
    print("CLASS WEIGHT COMPUTATION")
    print("=" * 70)
    labels_array = np.array(labels, dtype=int)
    classes = np.unique(labels_array)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_array)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights, strict=True)}
    print("Class weights computed successfully!")
    print(f"Classes found: {classes}")
    print(f"Class weights: {class_weight_dict}")
    print(
        f"Summary: {len(classes)} classes with weights ranging from "
        f"{min(weights):.4f} to {max(weights):.4f}"
    )
    return class_weight_dict


def _class_weight_tensor(
    mapping: dict[int, float],
    device: torch.device,
) -> torch.Tensor:
    max_c = int(max(mapping))
    vec = torch.ones(max_c + 1, dtype=torch.float32, device=device)
    for c, wi in mapping.items():
        vec[c] = wi
    return vec


def _oversample_sampler(labels: list[int]) -> WeightedRandomSampler:
    ys = np.array(labels, dtype=int)
    counts = {int(c): int((ys == c).sum()) for c in np.unique(ys)}
    sample_weights = [1.0 / counts[int(y)] for y in ys]
    print(f"Selective augmentation ENABLED (WeightedRandomSampler)")
    print(f"Class counts: {counts}")
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _write_model_details(model: nn.Module, cfg: TrainingConfig, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("===== MODEL SUMMARY =====\n")
        fh.write(repr(model))
        fh.write("\n\n===== LAYER HYPERPARAMETERS =====\n")
        for name, module in model.named_modules():
            if not name:
                continue
            fh.write(f"Layer: {name} ({module.__class__.__name__})\n")
            if isinstance(module, nn.Conv2d):
                fh.write(f"  filters: {module.out_channels}\n")
                fh.write(f"  kernel_size: {tuple(module.kernel_size)}\n")
                fh.write(f"  strides: {tuple(module.stride)}\n")
                fh.write(f"  padding: {module.padding}\n")
            if isinstance(module, nn.Linear):
                fh.write(f"  units: {module.out_features}\n")
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                fh.write(f"  rate: {module.p}\n")
            fh.write("-" * 30 + "\n")
        fh.write("\n\n===== LOSS & METRICS =====\n")
        fh.write(f"Loss: {cfg.loss_name}\n")
        fh.write(f"Metrics: {cfg.primary_metric}\n")
        fh.write(f"Output type: {cfg.output_type}\n")


# ============================================================================
# TRAIN
# ============================================================================


def train(cfg: TrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainingConfig()
    if cfg.train_manifest is None or cfg.val_manifest is None:
        raise ValueError("train_manifest and val_manifest are required")

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_id = cfg.run_id or new_run_id()
    run_dir = Path(cfg.report_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logging(run_dir)

    spec = spec_from_training_config(cfg)
    dump_spec(spec, run_dir / "model_spec.yaml")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    save_hyperparameters_txt(cfg, run_dir / "hyperparameters.txt")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_versions(), indent=2),
        encoding="utf-8",
    )

    regression = cfg.loss_name == "mse"
    ds_train = ManifestClassificationDataset(
        cfg.train_manifest,
        filepath_col=cfg.filepath_column,
        label_col=cfg.label_column,
        patient_col=cfg.patient_id_column,
        replicate_gray=cfg.replicate_grayscale_to_rgb,
        image_size=cfg.image_size(),
        jitter=cfg.train_jitter,
        regression=regression,
        output_type=cfg.output_type,
        loss_name=cfg.loss_name,
    )
    ds_val = ManifestClassificationDataset(
        cfg.val_manifest,
        filepath_col=cfg.filepath_column,
        label_col=cfg.label_column,
        patient_col=cfg.patient_id_column,
        replicate_gray=cfg.replicate_grayscale_to_rgb,
        image_size=cfg.image_size(),
        jitter=False,
        regression=regression,
        output_type=cfg.output_type,
        loss_name=cfg.loss_name,
    )

    print("\n=== Dataset summary ===")
    print_dataset_info("Train", ds_train.labels_array())
    print_dataset_info("Validation", ds_val.labels_array())

    if cfg.train_zscore:
        print("\n" + "=" * 70)
        print("Z-SCORE NORMALIZATION (training stats only)")
        print("=" * 70)
        train_mean, train_std = compute_train_zscore(ds_train)
        print(f"  Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        if train_std < 1e-6:
            print("  train_std is too small -- skipping z-score (constant or empty pixels).")
        else:
            ds_train.set_zscore(train_mean, train_std)
            ds_val.set_zscore(train_mean, train_std)
            (run_dir / "zscore.json").write_text(
                json.dumps({"mean": train_mean, "std": train_std}, indent=2),
                encoding="utf-8",
            )

    sampler = None
    if cfg.oversample_minority:
        sampler = _oversample_sampler(ds_train.labels_array())
    else:
        print("Selective augmentation DISABLED")

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.dataloader_num_workers,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
    )

    # Build model (explicit CNN by default)
    print("\n" + "=" * 70)
    print("BUILD MODEL")
    print("=" * 70)
    if cfg.architecture_id.lower() in {"cnn", "tiny_cnn", "tinycnn", "vgg_cnn"}:
        model = build_cnn_model(
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
        )
    else:
        model = build_model(cfg)
    model = model.to(device)
    print(model)
    _write_model_details(model, cfg, run_dir / "model_details.txt")
    print("\nModel compiled with:")
    print(f"  Loss function: {cfg.loss_name}")
    print(f"  Optimizer: AdamW (lr={cfg.learning_rate}, wd={cfg.weight_decay})")
    print(f"  Metrics: {cfg.primary_metric}")
    print(f"  Device: {device}")

    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    higher_is_better = cfg.freeze_mode == "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max" if higher_is_better else "min",
        patience=cfg.reduce_lr_patience,
        factor=cfg.reduce_lr_factor,
        min_lr=cfg.reduce_lr_min_lr,
    )

    class_w = None
    if cfg.class_weights:
        class_w = _class_weight_tensor(compute_class_weights(ds_train.labels_array()), device)
    else:
        print("\nClass weights DISABLED")

    epochs_csv = run_dir / "epochs.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        f"train_{cfg.primary_metric}",
        "val_loss",
        f"val_{cfg.primary_metric}",
        "lr",
    ]
    with epochs_csv.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    best_metric: float | None = None
    best_epoch = 0
    best_path = run_dir / "checkpoint_best.pt"
    disk_ckpt = cfg.checkpoint_dir / cfg.checkpoint_best_name
    stagnation = 0
    eps = cfg.early_stopping_min_delta

    print("\n" + "=" * 70)
    print("FIT")
    print("=" * 70)

    for epoch in range(1, cfg.epochs + 1):
        freeze_bb = cfg.freeze_backbone and epoch <= cfg.freeze_backbone_epochs
        set_backbone_trainable(model, trainable=not freeze_bb)
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = forward_logits(model, xb)
            if class_w is not None and cfg.loss_name == "sparse_categorical_crossentropy":
                loss = nn.functional.cross_entropy(logits, yb.long(), weight=class_w)
            else:
                loss = batch_loss(cfg.loss_name, logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(statistics.mean(epoch_losses)) if epoch_losses else float("nan")
        _, tr_metric = run_eval(
            model, train_loader, device, cfg.loss_name, cfg.primary_metric, threshold=cfg.binary_threshold
        )
        val_loss, val_metric = run_eval(
            model, val_loader, device, cfg.loss_name, cfg.primary_metric, threshold=cfg.binary_threshold
        )
        if cfg.use_reduce_lr:
            scheduler.step(val_metric if higher_is_better else val_loss)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            f"train_{cfg.primary_metric}": tr_metric,
            "val_loss": val_loss,
            f"val_{cfg.primary_metric}": val_metric,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        with epochs_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

        logger.info(
            "Epoch %s/%s train_loss=%.6f val_loss=%.6f val_%s=%.6f",
            epoch,
            cfg.epochs,
            tr_loss,
            val_loss,
            cfg.primary_metric,
            val_metric,
        )

        improved = False
        if best_metric is None:
            improved = True
        elif higher_is_better and val_metric > best_metric + eps:
            improved = True
        elif (not higher_is_better) and val_metric < best_metric - eps:
            improved = True

        if improved:
            best_metric = float(val_metric)
            best_epoch = epoch
            stagnation = 0
            blob = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_primary": val_metric,
                "val_loss": val_loss,
                "architecture_id": cfg.architecture_id,
                "seed": cfg.seed,
            }
            if cfg.use_model_checkpoint:
                torch.save(blob, best_path)
                cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    torch.save(blob, disk_ckpt)
                except OSError as exc:
                    logger.warning("Could not write E: checkpoint %s (%s)", disk_ckpt, exc)
                logger.info("Saved best checkpoint -> %s", best_path)
        else:
            stagnation += 1
            if cfg.use_early_stopping and stagnation >= cfg.early_stopping_patience:
                logger.info("Early stopping (patience %s).", cfg.early_stopping_patience)
                break

    plot_curves(epochs_csv, run_dir, cfg.primary_metric)
    (run_dir / "best_epoch.json").write_text(
        json.dumps(
            {"epoch": best_epoch, "val_primary": best_metric, "metric": cfg.primary_metric},
            indent=2,
        ),
        encoding="utf-8",
    )

    decision = decide_freeze(
        best_epoch=best_epoch,
        best_metric=best_metric if best_metric is not None else float("nan"),
        primary_metric=cfg.primary_metric,
        min_value=cfg.freeze_min_value,
        min_epoch=cfg.freeze_min_epoch,
        mode=cfg.freeze_mode,
    )
    metadata = {
        "run_id": run_id,
        "seed": cfg.seed,
        "architecture_id": cfg.architecture_id,
        "train_manifest": str(cfg.train_manifest),
        "val_manifest": str(cfg.val_manifest),
        "split_ids": {
            "train": str(cfg.train_manifest),
            "val": str(cfg.val_manifest),
            "test": str(cfg.test_manifest) if cfg.test_manifest else "",
        },
        "spec_dump": spec.model_dump(),
        "hyperparameters": cfg.hyperparameters_dict(),
    }
    frozen_path = None
    if best_path.is_file() and best_metric is not None:
        frozen_path = freeze_if_criteria_met(
            run_dir,
            best_path,
            metadata,
            decision,
            extra_frozen_dir=cfg.frozen_dir / run_id,
        )
    else:
        write_freeze_decision(run_dir, {**decision, "frozen": False, "reason": "no checkpoint"})

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "frozen": bool(decision["frozen"] and frozen_path is not None),
        "frozen_path": str(frozen_path) if frozen_path else None,
        "eval": None,
    }
    test_path = Path(cfg.test_manifest) if cfg.test_manifest else None
    if cfg.eval_on_test and test_path is not None and test_path.is_file() and best_path.is_file():
        from arth_tools.training.evaluate import evaluate_run

        result["eval"] = evaluate_run(
            run_dir,
            split_name="test",
            manifest=test_path,
            checkpoint=best_path,
            cfg=cfg,
            device=device,
        )
    elif cfg.eval_on_test:
        logger.info("Skipping held-out eval (missing test manifest or checkpoint).")

    logger.info("Training finished: %s", result)
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train from the control board (optional YAML overlay).")
    ap.add_argument("--config", type=Path, help="Optional YAML overlay on TrainingConfig")
    ap.add_argument("--train-manifest", type=Path)
    ap.add_argument("--val-manifest", type=Path)
    ap.add_argument("--test-manifest", type=Path)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--run-id", type=str)
    ap.add_argument("--no-eval", action="store_true", help="Skip held-out test eval after fit")
    args = ap.parse_args(argv)

    cfg = load_config_yaml(args.config) if args.config else TrainingConfig()
    if args.train_manifest:
        cfg.train_manifest = args.train_manifest
    if args.val_manifest:
        cfg.val_manifest = args.val_manifest
    if args.test_manifest:
        cfg.test_manifest = args.test_manifest
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.run_id:
        cfg.run_id = args.run_id
    if args.no_eval:
        cfg.eval_on_test = False
    print(json.dumps(train(cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
