# -*- coding: utf-8 -*-
"""
Held-out evaluation for a finished training run.

Always reloads the run's config_snapshot.yaml + zscore.json + checkpoint so
test-time preprocessing matches training. Do not score with whatever happens
to be on the control board unless you pass --use-control-board.

    python -m arth_tools eval --run-dir reporting/training/<run_id>
    python -m arth_tools eval --run-dir reporting/training/<run_id> --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from arth_tools.data.dataset import ManifestClassificationDataset
from arth_tools.training.backbones import build_cnn_model, build_model
from arth_tools.training.config import TrainingConfig, load_config_yaml
from arth_tools.training.metrics import (
    aggregate_by_patient,
    collect_predictions,
    metric_from_scores,
    metric_suite,
    predict_from_scores,
)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_run_config(run_dir: Path) -> TrainingConfig:
    snap = Path(run_dir) / "config_snapshot.yaml"
    if not snap.is_file():
        raise FileNotFoundError(f"No config_snapshot.yaml in {run_dir}")
    return load_config_yaml(snap)


def find_checkpoint(run_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for cand in (
        Path(run_dir) / "checkpoint_best.pt",
        Path(run_dir) / "frozen" / "model.pt",
        Path(run_dir) / "frozen" / "state_dict.pt",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def build_eval_model(cfg: TrainingConfig) -> torch.nn.Module:
    if cfg.architecture_id.lower() in {"cnn", "tiny_cnn", "tinycnn", "vgg_cnn"}:
        return build_cnn_model(
            output_type=cfg.output_type,
            num_kernels=cfg.num_kernels,
            kernel_size=tuple(cfg.kernel_size),
            conv_stride=tuple(cfg.conv_stride),
            pool_stride=tuple(cfg.pool_stride),
            activation_func=cfg.activation_func,
            input_height=cfg.input_height,
            input_width=cfg.input_width,
            input_channels=cfg.input_channels,
            num_classes=cfg.num_classes,
            drop=cfg.drop,
            spatial_drop=cfg.spatial_drop,
        )
    return build_model(cfg)


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and all(isinstance(k, str) for k in blob):
        state = blob
    else:
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint}")
    model.load_state_dict(state)


def apply_saved_zscore(ds: ManifestClassificationDataset, run_dir: Path) -> None:
    zpath = Path(run_dir) / "zscore.json"
    if not zpath.is_file():
        return
    raw = json.loads(zpath.read_text(encoding="utf-8"))
    mean = raw.get("mean")
    std = raw.get("std")
    if mean is None or std is None:
        return
    if float(std) < 1e-6:
        print("Saved train_std is degenerate; eval will not apply z-score.")
        return
    ds.set_zscore(float(mean), float(std))


def _manifest_for_split(cfg: TrainingConfig, split_name: str, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    mapping = {
        "train": cfg.train_manifest,
        "val": cfg.val_manifest,
        "test": cfg.test_manifest,
    }
    path = mapping.get(split_name)
    if path is None:
        raise ValueError(f"Unknown split {split_name!r}; use train, val, or test.")
    if not Path(path).is_file():
        raise FileNotFoundError(f"{split_name} manifest missing: {path}")
    return Path(path)


def _plot_eval(out_dir: Path, suite: dict[str, Any], y_true: np.ndarray, scores: np.ndarray, loss_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if loss_name == "mse":
        return
    cm = np.asarray(suite.get("confusion_matrix", []), dtype=float)
    if cm.size:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

    y = y_true.astype(int)
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_curve

        if scores.ndim == 2 and scores.shape[1] == 2:
            pos = scores[:, 1]
        else:
            pos = scores.reshape(-1)
        fpr, tpr, _ = roc_curve(y, pos)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {suite.get('auc', float('nan')):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC (held-out)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "roc.png", dpi=120)
        plt.close(fig)


def evaluate_run(
    run_dir: Path,
    *,
    split_name: str = "test",
    manifest: Path | None = None,
    checkpoint: Path | None = None,
    cfg: TrainingConfig | None = None,
    device: torch.device | str | None = None,
    threshold: float | None = None,
    patient_aggregate: bool | None = None,
    use_control_board: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print(f"Run dir: {run_dir}")
    print(f"Split: {split_name}")

    if use_control_board:
        cfg = cfg or TrainingConfig()
        print("Using live CONTROL BOARD (not the run snapshot).")
    else:
        cfg = cfg or load_run_config(run_dir)

    thresh = float(cfg.binary_threshold if threshold is None else threshold)
    do_patient = cfg.eval_patient_aggregate if patient_aggregate is None else patient_aggregate
    man = _manifest_for_split(cfg, split_name, manifest)
    ckpt = find_checkpoint(run_dir, checkpoint)
    dev = device if isinstance(device, torch.device) else resolve_device(device or cfg.device)

    print(f"Manifest: {man}")
    print(f"Checkpoint: {ckpt}")
    print(f"Device: {dev}")

    regression = cfg.loss_name == "mse"
    ds = ManifestClassificationDataset(
        man,
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
    apply_saved_zscore(ds, run_dir)

    loader = DataLoader(
        ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=False,
        num_workers=int(cfg.dataloader_num_workers),
    )
    model = build_eval_model(cfg).to(dev)
    load_weights(model, ckpt)
    model.eval()

    y, scores, avg_loss = collect_predictions(model, loader, dev, cfg.loss_name)
    pred = (
        scores.reshape(-1)
        if cfg.loss_name == "mse"
        else predict_from_scores(scores, threshold=thresh)
    )
    image_suite = metric_suite(y, scores, cfg.loss_name, threshold=thresh)
    image_suite["loss"] = avg_loss
    image_suite["primary_metric"] = cfg.primary_metric
    image_suite["primary"] = metric_from_scores(
        y, scores, cfg.primary_metric, cfg.loss_name, threshold=thresh
    )

    pids = [ds.patient_id_for_index(i) for i in range(len(ds))]
    paths = [ds.filepath_for_index(i) for i in range(len(ds))]

    pred_df = pd.DataFrame(
        {
            "filepath": paths,
            "patient_id": pids,
            "y_true": y.tolist(),
            "y_pred": pred.tolist(),
        }
    )
    if scores.ndim == 2:
        for c in range(scores.shape[1]):
            pred_df[f"score_{c}"] = scores[:, c]
    else:
        pred_df["score_0"] = scores.reshape(-1)

    patient_suite = None
    if do_patient and len(set(pids)) > 1:
        y_p, s_p = aggregate_by_patient(pids, y, scores, cfg.loss_name, threshold=thresh)
        patient_suite = metric_suite(y_p, s_p, cfg.loss_name, threshold=thresh)
        patient_suite["primary"] = metric_from_scores(
            y_p, s_p, cfg.primary_metric, cfg.loss_name, threshold=thresh
        )
        print(f"Patient-level rows: {patient_suite['n']} (from {len(pids)} images)")

    dest = run_dir / f"eval_{split_name}"
    dest.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(dest / "predictions.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "split": split_name,
        "manifest": str(man),
        "checkpoint": str(ckpt),
        "n_images": int(len(y)),
        "n_patients": int(len(set(pids))),
        "loss": avg_loss,
        "threshold": thresh,
        "image": image_suite,
        "patient": patient_suite,
    }
    (dest / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _plot_eval(dest, image_suite, y, scores, cfg.loss_name)

    print("\n=== Image-level ===")
    _print_suite(image_suite, cfg.loss_name)
    if patient_suite is not None:
        print("\n=== Patient-level (mean scores / majority via mean) ===")
        _print_suite(patient_suite, cfg.loss_name)
    print(f"\nWrote {dest}")
    return summary


def _print_suite(suite: dict[str, Any], loss_name: str) -> None:
    if loss_name == "mse":
        print(f"  RMSE : {suite.get('rmse', float('nan')):.4f}")
        print(f"  MAE  : {suite.get('mae', float('nan')):.4f}")
        print(f"  R2   : {suite.get('r2', float('nan')):.4f}")
        return
    print(f"  Accuracy : {suite.get('accuracy', float('nan')):.4f}")
    print(f"  Macro F1 : {suite.get('macro_f1', float('nan')):.4f}")
    print(f"  Precision: {suite.get('precision', float('nan')):.4f}")
    print(f"  Recall   : {suite.get('recall', float('nan')):.4f}")
    print(f"  AUC      : {suite.get('auc', float('nan')):.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a training run on a held-out manifest.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--manifest", type=Path, help="Override the snapshot manifest for this split")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--device", type=str)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--no-patient-agg", action="store_true")
    ap.add_argument(
        "--use-control-board",
        action="store_true",
        help="Ignore config_snapshot.yaml and use arth_tools/training/config.py",
    )
    args = ap.parse_args(argv)
    summary = evaluate_run(
        args.run_dir,
        split_name=args.split,
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        patient_aggregate=False if args.no_patient_agg else None,
        use_control_board=args.use_control_board,
    )
    print(json.dumps({k: summary[k] for k in ("split", "n_images", "n_patients", "loss") if k in summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
