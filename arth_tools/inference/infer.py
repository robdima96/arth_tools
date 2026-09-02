# -*- coding: utf-8 -*-
"""Unlabeled inference: DICOM or PNG folder → predictions.csv (no ground truth)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from arth_tools.data.dataset import ImagePathDataset, items_from_manifest
from arth_tools.data.labels import patient_id_from_dataset, source_path
from arth_tools.models.load import LoadedClassifier, load_classifier
from arth_tools.training.metrics import collect_predictions, predict_from_scores

SKIP_EXTS = {
    ".txt",
    ".xml",
    ".json",
    ".csv",
    ".gif",
    ".etl",
    ".log",
    ".exe",
    ".md",
    ".yaml",
    ".yml",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _patient_id_for_file(path: Path) -> str:
    if path.suffix.lower() in IMAGE_EXTS:
        return path.parent.name or path.stem
    try:
        import pydicom

        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        pid = patient_id_from_dataset(ds)
        if pid:
            return pid
        src = source_path(ds)
        if src is not None:
            return src.parent.name or path.stem
    except Exception:
        pass
    return path.parent.name or path.stem


def collect_image_items(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if root.is_file():
        return [{"path": root, "patient_id": _patient_id_for_file(root)}]
    if not root.is_dir():
        raise FileNotFoundError(root)
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in SKIP_EXTS:
            continue
        if path.name.upper() == "DICOMDIR":
            continue
        items.append({"path": path, "patient_id": _patient_id_for_file(path)})
    if not items:
        raise FileNotFoundError(f"No images or DICOM files under {root}")
    return items


def run_infer(
    source: Path,
    *,
    items: list[dict[str, Any]],
    out_csv: Path,
    device: str = "auto",
    batch_size: int | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    loaded: LoadedClassifier = load_classifier(source, checkpoint=checkpoint)
    cfg = loaded.cfg
    dev = resolve_device(device)
    loaded.model.to(dev)
    ds = ImagePathDataset(items, loaded.recipe)
    loader = DataLoader(
        ds,
        batch_size=max(1, int(batch_size if batch_size is not None else cfg.batch_size)),
        shuffle=False,
        num_workers=int(cfg.dataloader_num_workers),
    )
    _, scores, _ = collect_predictions(loaded.model, loader, dev, cfg.loss_name, with_loss=False)
    pred = (
        scores.reshape(-1)
        if cfg.loss_name == "mse"
        else predict_from_scores(scores, threshold=float(cfg.binary_threshold))
    )
    names = loaded.class_names()
    rows = []
    for i, item in enumerate(items):
        y_pred = pred[i]
        rec: dict[str, Any] = {
            "filepath": str(Path(item["path"]).resolve()),
            "patient_id": item.get("patient_id", "unknown"),
            "y_pred": int(y_pred) if cfg.loss_name != "mse" else float(y_pred),
        }
        if cfg.loss_name != "mse":
            rec["y_pred_label"] = names.get(int(y_pred), str(int(y_pred)))
        if scores.ndim == 2:
            for c in range(scores.shape[1]):
                rec[f"score_{c}"] = float(scores[i, c])
        else:
            rec["score_0"] = float(scores.reshape(-1)[i])
        rows.append(rec)
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = {
        "source": str(Path(source)),
        "bundle_dir": str(loaded.bundle_dir),
        "weights": str(loaded.weights_path),
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if not df.empty else 0,
        "predictions": str(out_csv),
        "architecture_id": cfg.architecture_id,
    }
    sidecar = out_csv.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score unlabeled DICOM/PNG files with a trained run or frozen bundle."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Training run directory (uses run/bundle)")
    src.add_argument("--bundle", type=Path, help="Frozen bundle directory (bundle.json)")
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--dicom-root", type=Path, help="Folder of unlabeled DICOM files")
    inp.add_argument("--images", type=Path, help="Folder (or file) of PNG/JPEG/DICOM")
    inp.add_argument("--manifest", type=Path, help="CSV with a filepath column (labels ignored)")
    ap.add_argument("--out", type=Path, help="predictions.csv path")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args(argv)

    source = args.bundle if args.bundle is not None else args.run_dir
    if args.manifest is not None:
        items = items_from_manifest(args.manifest)
    else:
        root = args.dicom_root if args.dicom_root is not None else args.images
        items = collect_image_items(root)
    out_csv = args.out
    if out_csv is None:
        out_csv = Path(source) / "infer_predictions.csv"
    run_infer(
        source,
        items=items,
        out_csv=out_csv,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
