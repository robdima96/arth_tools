# -*- coding: utf-8 -*-
"""Turn a DICOM folder into train/val/test manifests (export + patient split)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arth_tools.data.export import export_from_dicom_root
from arth_tools.data.labels import LabelError
from arth_tools.data.preprocess import PreprocessRecipe
from arth_tools.data.splits import split_manifest_by_patient, write_split_manifests
from arth_tools.training.config import TrainingConfig, load_config_yaml


def prepare_dicom_root(
    dicom_root: Path,
    *,
    image_root: Path,
    manifest_dir: Path,
    master_manifest: Path | None = None,
    label_source: str = "auto",
    label_tag: str = "StudyDescription",
    label_map: dict[str, int] | None = None,
    label_map_path: Path | None = None,
    allowed_modalities: list[str] | None = None,
    default_label: int | None = None,
    label_csv: Path | None = None,
    label_csv_join: str = "patient_id",
    label_csv_column: str = "label",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    split_force_pt_strat: bool = True,
    split_force_eq_dist: bool = False,
    recipe: Any | None = None,
) -> dict[str, Path]:
    from arth_tools.data.preprocess import PreprocessRecipe

    master = Path(master_manifest) if master_manifest is not None else Path(manifest_dir) / "manifest_master.csv"
    df = export_from_dicom_root(
        Path(dicom_root),
        Path(image_root),
        master,
        label_source=label_source,  # type: ignore[arg-type]
        label_tag=label_tag,
        label_map=label_map,
        default_label=default_label,
        allowed_modalities=allowed_modalities,
        label_map_path=label_map_path,
        label_csv=label_csv,
        label_csv_join=label_csv_join,
        label_csv_column=label_csv_column,
        recipe=recipe if recipe is not None else PreprocessRecipe(),
    )
    assigned, report = split_manifest_by_patient(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        split_force_pt_strat=split_force_pt_strat,
        split_force_eq_dist=split_force_eq_dist,
    )
    paths = write_split_manifests(assigned, Path(manifest_dir), report=report)
    paths["master"] = master
    if label_map_path is not None:
        paths["label_map"] = Path(label_map_path)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(json.dumps(report, indent=2, default=str))
    return paths


def prepare_from_config(cfg: TrainingConfig, *, default_label: int | None = None) -> dict[str, Path]:
    if cfg.task_id:
        print(f"Task: {cfg.task_id} ({cfg.task_name or cfg.modality or 'unspecified'})")
    paths = prepare_dicom_root(
        cfg.dicom_root,
        image_root=cfg.image_root,  # type: ignore[arg-type]
        manifest_dir=cfg.manifest_dir,  # type: ignore[arg-type]
        master_manifest=cfg.master_manifest,
        label_source=cfg.label_source,
        label_tag=cfg.label_dicom_tag,
        label_map=cfg.label_map or None,
        label_map_path=cfg.label_map_path,
        allowed_modalities=cfg.allowed_modalities or None,
        default_label=default_label,
        label_csv=cfg.label_csv,
        label_csv_join=cfg.label_csv_join,
        label_csv_column=cfg.label_csv_column,
        train_frac=cfg.train_split,
        val_frac=cfg.val_split,
        test_frac=cfg.test_split,
        seed=cfg.seed,
        split_force_pt_strat=cfg.split_force_pt_strat,
        split_force_eq_dist=cfg.split_force_eq_dist,
        recipe=PreprocessRecipe.from_training_config(cfg),
    )
    expected = cfg.num_classes
    inferred = cfg.apply_label_map_file()
    if inferred is not None and inferred != expected:
        print(
            f"Note: task YAML num_classes={expected} but label_map has {inferred}. "
            "train --config will size the head from the label map."
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a DICOM folder and write patient-grouped train/val/test manifests."
    )
    ap.add_argument("--config", type=Path, help="Task YAML (configs/kl_grade.yaml or kl_grade)")
    ap.add_argument("--dicom-root", type=Path, help="Override DICOM folder")
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--out-dir", type=Path, help="Override manifest directory")
    ap.add_argument("--label-source", choices=("auto", "tag", "folder", "csv"))
    ap.add_argument("--label-tag")
    ap.add_argument("--label-csv", type=Path)
    ap.add_argument("--default-label", type=int, default=None)
    ap.add_argument("--train", type=float)
    ap.add_argument("--val", type=float)
    ap.add_argument("--test", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stratify", action="store_true", default=None)
    ap.add_argument("--no-patient-strat", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config_yaml(args.config) if args.config else TrainingConfig()
    if args.dicom_root:
        cfg.dicom_root = args.dicom_root
    if args.image_root:
        cfg.image_root = args.image_root
    if args.out_dir:
        cfg.manifest_dir = args.out_dir
        cfg.master_manifest = Path(args.out_dir) / "manifest_master.csv"
        cfg.label_map_path = Path(args.out_dir) / "label_map.json"
    if args.label_source:
        cfg.label_source = args.label_source
    if args.label_tag:
        cfg.label_dicom_tag = args.label_tag
    if args.label_csv:
        cfg.label_csv = args.label_csv
        cfg.label_source = args.label_source or "csv"
    if args.train is not None:
        cfg.train_split = args.train
    if args.val is not None:
        cfg.val_split = args.val
    if args.test is not None:
        cfg.test_split = args.test
    if args.seed is not None:
        cfg.seed = args.seed
    if args.stratify:
        cfg.split_force_eq_dist = True
    if args.no_patient_strat:
        cfg.split_force_pt_strat = False

    try:
        prepare_from_config(cfg, default_label=args.default_label)
    except (LabelError, FileNotFoundError, RuntimeError) as exc:
        print(f"Prepare failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
