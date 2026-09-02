"""End-to-end smoke: synthetic DICOM -> prepare -> train -> eval -> infer, plus HKA.

Uses local fixture paths so the smoke run does not need a data disk.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from arth_tools.data.dicom import load_patient_images_from_root, write_synthetic_dicom
from arth_tools.data.prepare import prepare_dicom_root
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
                study_description=str(p % 2),
            )
    return dicom_dir


def run_hka_smoke(work_dir: Path) -> dict:
    from arth_tools.hka.config import HKAConfig
    from arth_tools.hka.run import run_hka
    from arth_tools.hka.synthetic import write_synthetic_longleg_dicom

    dicom_dir = work_dir / "hka_dicoms"
    write_synthetic_longleg_dicom(dicom_dir / "HKA001" / "longleg.dcm")
    payload = run_hka(HKAConfig(dicom_root=dicom_dir, output_dir=work_dir / "hka_out"))
    rec = next((r for r in payload["results"] if r.get("ok")), {})
    return {
        "n_ok": payload["n_ok"],
        "summary": rec.get("summary"),
        "output_dicom": rec.get("output_dicom"),
    }


def run_smoke(*, work_dir: Path | None = None) -> dict:
    work_dir = Path(work_dir) if work_dir else FIXTURES_DIR / "smoke"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    hka = run_hka_smoke(work_dir)

    dicom_dir = write_dicom_fixtures(work_dir / "dicoms")
    loaded = load_patient_images_from_root(dicom_dir, load_pixels=True)
    split_dir = work_dir / "splits"
    paths = prepare_dicom_root(
        dicom_dir,
        image_root=work_dir / "png",
        manifest_dir=split_dir,
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        seed=0,
        split_force_pt_strat=True,
        split_force_eq_dist=False,
    )
    report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))

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

    from arth_tools.inference.infer import collect_image_items, run_infer

    infer_summary = run_infer(
        Path(train_result["run_dir"]),
        items=collect_image_items(dicom_dir),
        out_csv=work_dir / "infer_predictions.csv",
        device="cpu",
    )

    out = {
        "n_dicom_patients": len(loaded),
        "split_ok": report["ok"],
        "split_report": report,
        "train": {
            k: train_result[k]
            for k in ("run_id", "run_dir", "best_epoch", "best_metric", "frozen", "bundle_dir")
            if k in train_result
        },
        "eval": {
            "n_images": disk_eval.get("n_images"),
            "n_patients": disk_eval.get("n_patients"),
            "image_accuracy": (disk_eval.get("image") or {}).get("accuracy"),
            "patient_accuracy": (disk_eval.get("patient") or {}).get("accuracy"),
        },
        "infer": {
            "n_images": infer_summary.get("n_images"),
            "predictions": infer_summary.get("predictions"),
        },
        "hka": hka,
    }
    (work_dir / "smoke_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    result = run_smoke()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
