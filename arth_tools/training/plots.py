"""Epoch curve graphs written next to epochs.csv (IPFP plotted these after fit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_curves(epochs_csv: Path, out_dir: Path, primary_metric: str) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(epochs_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = out_dir / "curves_loss.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=120)
    plt.close(fig)

    metric_col = f"val_{primary_metric}"
    metric_path = out_dir / "curves_metric.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_" + primary_metric in df.columns:
        ax.plot(df["epoch"], df["train_" + primary_metric], label="train")
    if metric_col in df.columns:
        ax.plot(df["epoch"], df[metric_col], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(primary_metric.upper())
    ax.set_title(f"Training and Validation {primary_metric.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(metric_path, dpi=120)
    plt.close(fig)
    return loss_path, metric_path
