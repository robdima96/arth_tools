"""End-to-end smoke: synthetic DICOM -> PNG -> patient splits -> train -> held-out eval.

Uses local fixture paths (not E:\\) so the smoke run does not need the removable disk.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from arth_tools.data.dicom import load_patient_images_from_root, write_synthetic_dicom
from arth_tools.data.export import export_from_dicom_root
from arth_tools.data.splits import split_manifest_by_patient, write_split_manifests
from arth_tools.paths import FIXTURES_DIR, TRAINING_REPORT_DIR
from arth_tools.training.config import HPOConfig, TrainingConfig
from arth_tools.training.evaluate import evaluate_run
from arth_tools.training.train import train


def write_png_fixtures(image_dir: Path, n_patients: int = 6, n_per: int = 2) -> Path:
    """Also write PNGs so training can run without DICOM if needed."""
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    import pandas as pd

    for p in range(n_patients):
        pid = f"P{p:03d}"
        label = p % 2
        for i in range(n_per):
            arr = np.full((16, 16), 40 + 20 * label + i * 5, dtype=np.uint8)
            dest = image_dir / pid / f"slice_{i:04d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode="L").convert("RGB").save(dest)
            rows.append({"filepath": str(dest.resolve()), "patient_id": pid, "label": label})
    manifest = image_dir / "manifest_master.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def write_dicom_fixtures(dicom_dir: Path, n_patients: int = 6, n_per: int = 2) -> Path:
    dicom_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_patients):
        pid = f"P{p:03d}"
        for i in range(n_per):
            pixels = np.full((8, 8), 30 + 10 * (p % 2) + i, dtype=np.uint8)
            write_synthetic_dicom(
                dicom_dir / pid / f"img_{i:04d}.dcm",
                patient_id=pid,
                patient_name="Fixture",
                pixels=pixels,
                modality="US",
            )
    return dicom_dir


def run_smoke(*, work_dir: Path | None = None) -> dict:
    work_dir = Path(work_dir) if work_dir else FIXTURES_DIR / "smoke"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    dicom_dir = write_dicom_fixtures(work_dir / "dicoms")
    loaded = load_patient_images_from_root(dicom_dir, load_pixels=True)
    image_root = work_dir / "png"
    master = work_dir / "manifest_master.csv"
    export_from_dicom_root(dicom_dir, image_root, master, default_label=0)

    import pandas as pd

    df = pd.read_csv(master)
    df["label"] = df["patient_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int) % 2
    df.to_csv(master, index=False)

    assigned, report = split_manifest_by_patient(
        df,
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        seed=0,
        split_force_pt_strat=True,
        split_force_eq_dist=False,
    )
    split_dir = work_dir / "splits"
    paths = write_split_manifests(assigned, split_dir, report=report)

    cfg = TrainingConfig(
        architecture_id="cnn",
        architecture_name="VGG-inspired CNN (smoke)",
        num_classes=2,
        input_height=16,
        input_width=16,
        num_kernels=8,
        primary_metric="accuracy",
        freeze_min_value=0.0,
        freeze_min_epoch=1,
        freeze_mode="max",
        report_root=TRAINING_REPORT_DIR,
        run_id="smoke_tiny_cnn",
        seed=0,
        device="cpu",
        train_manifest=paths["train"],
        val_manifest=paths["val"],
        test_manifest=paths["test"],
        checkpoint_dir=work_dir / "checkpoints",
        frozen_dir=work_dir / "frozen",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        train_jitter=True,
        train_zscore=True,
        class_weights=True,
        oversample_minority=True,
        freeze_backbone=True,
        freeze_backbone_epochs=1,
        use_early_stopping=True,
        use_reduce_lr=True,
        eval_on_test=True,
        eval_patient_aggregate=True,
        hpo=HPOConfig(enabled=False),
    )
    train_result = train(cfg)

    # Re-run eval from disk to prove the snapshot path (not just the in-memory hook).
    disk_eval = evaluate_run(Path(train_result["run_dir"]), split_name="test")

    out = {
        "n_dicom_patients": len(loaded),
        "split_ok": report["ok"],
        "split_report": report,
        "train": {
            k: train_result[k]
            for k in ("run_id", "run_dir", "best_epoch", "best_metric", "frozen")
            if k in train_result
        },
        "eval": {
            "n_images": disk_eval.get("n_images"),
            "n_patients": disk_eval.get("n_patients"),
            "image_accuracy": (disk_eval.get("image") or {}).get("accuracy"),
            "patient_accuracy": (disk_eval.get("patient") or {}).get("accuracy"),
        },
    }
    (work_dir / "smoke_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    result = run_smoke()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
