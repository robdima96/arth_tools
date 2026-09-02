# -*- coding: utf-8 -*-
"""
DICOM folder → PNG + master manifest.

Patient IDs and labels are read from the files (DICOM tags or class folders).
You do not hand-author the CSV.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from arth_tools.data.dicom import load_from_disk, load_patient_images_from_root
from arth_tools.data.labels import (
    LabelError,
    assign_labels,
    csv_vals_for_keys,
    folder_component,
    load_csv_label_table,
    patient_id_from_dataset,
    source_path,
    tag_value,
)
from arth_tools.data.preprocess import (
    CNN_BAKED_STAGES,
    PreprocessRecipe,
    array_to_pil,
    dicom_to_display_u8,
)
from arth_tools.training.config import (
    DICOM_ROOT,
    FILEPATH_COLUMN,
    IMAGE_ROOT,
    LABEL_COLUMN,
    LABEL_DICOM_TAG,
    LABEL_MAP,
    LABEL_MAP_PATH,
    LABEL_SOURCE,
    MANIFEST_MASTER,
    PATIENT_ID_COLUMN,
    TrainingConfig,
    load_config_yaml,
)

FILEPATH_COL = FILEPATH_COLUMN
PATIENT_COL = PATIENT_ID_COLUMN
LABEL_COL = LABEL_COLUMN
LABEL_RAW_COL = "label_raw"
MODALITY_COL = "modality"
CHANNELS_COL = "channels"
SOURCE_COL = "source_dicom"
STUDY_UID_COL = "study_uid"
SERIES_UID_COL = "series_uid"
PHOTOMETRIC_COL = "photometric"
BITS_STORED_COL = "bits_stored"
MANUFACTURER_COL = "manufacturer"
MANUFACTURER_MODEL_COL = "manufacturer_model"
WINDOW_APPLIED_COL = "window_applied"
BAKED_STAGES_COL = "baked_stages"


def _pid_clean(key: str) -> str:
    return str(key).replace(os.sep, "_").replace(" ", "_")


def _slice_to_png(ds: Any, dest: Path, recipe: PreprocessRecipe | None = None) -> dict[str, Any] | None:
    recipe = recipe or PreprocessRecipe()
    try:
        if not hasattr(ds, "pixel_array"):
            return None
        u8, meta = dicom_to_display_u8(ds, recipe)
        pil = array_to_pil(u8, rgb=True)
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil.save(dest)
    baked = meta.get("baked_stages") or list(CNN_BAKED_STAGES)
    baked_s = ";".join(str(s) for s in baked)
    return {
        FILEPATH_COL: str(dest.resolve()),
        MODALITY_COL: str(ds.get("Modality", "") or ""),
        CHANNELS_COL: 3,
        STUDY_UID_COL: str(ds.get("StudyInstanceUID", "") or ""),
        SERIES_UID_COL: str(ds.get("SeriesInstanceUID", "") or ""),
        SOURCE_COL: str(source_path(ds) or ""),
        PHOTOMETRIC_COL: str(meta.get("photometric") or ds.get("PhotometricInterpretation", "") or ""),
        BITS_STORED_COL: meta.get("bits_stored") if meta.get("bits_stored") is not None else ds.get("BitsStored", ""),
        MANUFACTURER_COL: str(meta.get("manufacturer") or ds.get("Manufacturer", "") or ""),
        MANUFACTURER_MODEL_COL: str(meta.get("manufacturer_model") or ds.get("ManufacturerModelName", "") or ""),
        WINDOW_APPLIED_COL: bool(meta.get("window_applied")),
        BAKED_STAGES_COL: baked_s,
    }


def export_patient_dict(
    patient_dict: dict[str, list[Any]],
    image_root: Path,
    master_manifest: Path,
    *,
    dicom_root: Path | None = None,
    label_source: str = LABEL_SOURCE,
    label_tag: str = LABEL_DICOM_TAG,
    label_map: dict[str, int] | None = None,
    default_label: int | None = None,
    allowed_modalities: list[str] | None = None,
    label_map_path: Path | None = None,
    label_csv: Path | None = None,
    label_csv_join: str = "patient_id",
    label_csv_column: str = "label",
    filepath_col: str = FILEPATH_COL,
    patient_col: str = PATIENT_COL,
    label_col: str = LABEL_COL,
    recipe: PreprocessRecipe | None = None,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("EXPORT SLICES TO PNG")
    print("=" * 70)

    rows: list[dict[str, Any]] = []
    tag_vals: list[str | None] = []
    folder_vals: list[str | None] = []
    patient_ids: list[str] = []
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    root = Path(dicom_root) if dicom_root is not None else None
    allowed = {m.strip().upper() for m in (allowed_modalities or []) if str(m).strip()}
    n_skip_mod = 0
    recipe = recipe or PreprocessRecipe()

    for pkey, series in patient_dict.items():
        fallback_pid = _pid_clean(pkey)
        if not isinstance(series, list):
            continue
        n_ok = 0
        for idx, ds in enumerate(series):
            mod = str(ds.get("Modality", "") or "").strip().upper()
            if allowed and mod not in allowed:
                n_skip_mod += 1
                continue
            pid = patient_id_from_dataset(ds, dicom_root=root) or fallback_pid
            pid = _pid_clean(pid)
            png = image_root / pid / f"slice_{idx:04d}.png"
            meta = _slice_to_png(ds, png, recipe)
            if meta is None:
                continue
            rows.append(
                {
                    filepath_col: meta[FILEPATH_COL],
                    patient_col: pid,
                    MODALITY_COL: meta[MODALITY_COL],
                    CHANNELS_COL: meta[CHANNELS_COL],
                    STUDY_UID_COL: meta[STUDY_UID_COL],
                    SERIES_UID_COL: meta[SERIES_UID_COL],
                    SOURCE_COL: meta[SOURCE_COL],
                    PHOTOMETRIC_COL: meta.get(PHOTOMETRIC_COL, ""),
                    BITS_STORED_COL: meta.get(BITS_STORED_COL, ""),
                    MANUFACTURER_COL: meta.get(MANUFACTURER_COL, ""),
                    MANUFACTURER_MODEL_COL: meta.get(MANUFACTURER_MODEL_COL, ""),
                    WINDOW_APPLIED_COL: meta.get(WINDOW_APPLIED_COL, False),
                    BAKED_STAGES_COL: meta.get(BAKED_STAGES_COL, ";".join(CNN_BAKED_STAGES)),
                }
            )
            tag_vals.append(tag_value(ds, label_tag))
            folder_vals.append(folder_component(ds, root))
            patient_ids.append(pid)
            n_ok += 1
        print(f"  {fallback_pid}: {n_ok} slices")

    if n_skip_mod:
        print(f"Skipped {n_skip_mod} instances whose Modality is not in {sorted(allowed)}")
    if not rows:
        hint = "check pixel data"
        if allowed:
            hint = f"check pixel data and Modality filter {sorted(allowed)}"
        raise RuntimeError(f"No slices exported — {hint}.")

    csv_vals: list[str | None] | None = None
    if str(label_source) == "csv":
        if label_csv is None:
            raise LabelError("label_source=csv requires label_csv (a spreadsheet of grades).")
        join_col = str(label_csv_join or "patient_id")
        table = load_csv_label_table(label_csv, join_col=join_col, label_col=label_csv_column or "label")
        if join_col == "study_uid":
            keys = [str(row.get(STUDY_UID_COL, "") or "") for row in rows]
        else:
            keys = list(patient_ids)
        csv_vals = csv_vals_for_keys(keys, table)

    encoded, mapping, used_source, raws = assign_labels(
        tag_vals,
        folder_vals,
        patient_ids,
        source=label_source,  # type: ignore[arg-type]
        tag_name=label_tag,
        label_map=label_map,
        default_label=default_label,
        csv_vals=csv_vals,
    )
    for row, lab, raw in zip(rows, encoded, raws, strict=True):
        row[label_col] = int(lab)
        row[LABEL_RAW_COL] = raw

    df = pd.DataFrame(rows)
    cols = [
        filepath_col,
        patient_col,
        label_col,
        LABEL_RAW_COL,
        MODALITY_COL,
        CHANNELS_COL,
        STUDY_UID_COL,
        SERIES_UID_COL,
        SOURCE_COL,
        PHOTOMETRIC_COL,
        BITS_STORED_COL,
        MANUFACTURER_COL,
        MANUFACTURER_MODEL_COL,
        WINDOW_APPLIED_COL,
        BAKED_STAGES_COL,
    ]
    df = df[[c for c in cols if c in df.columns]]

    master_manifest = Path(master_manifest)
    master_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(master_manifest, index=False)
    sidecar = {
        "source": used_source,
        "tag": label_tag if used_source == "tag" else None,
        "map": mapping,
        "num_classes": int(len(set(mapping.values()))),
    }
    label_dest = Path(label_map_path) if label_map_path is not None else master_manifest.parent / LABEL_MAP_PATH.name
    label_dest.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote master manifest ({len(df)} rows) -> {master_manifest}")
    print(f"Labels from {used_source!r}: {mapping}  (num_classes={sidecar['num_classes']})")
    print(f"Wrote label map -> {label_dest}")
    return df


def export_from_dicom_root(
    dicom_root: Path,
    image_root: Path,
    master_manifest: Path,
    *,
    load_pixels: bool = True,
    label_source: str = LABEL_SOURCE,
    label_tag: str = LABEL_DICOM_TAG,
    label_map: dict[str, int] | None = None,
    default_label: int | None = None,
    allowed_modalities: list[str] | None = None,
    label_map_path: Path | None = None,
    label_csv: Path | None = None,
    label_csv_join: str = "patient_id",
    label_csv_column: str = "label",
    recipe: PreprocessRecipe | None = None,
) -> pd.DataFrame:
    dicom_root = Path(dicom_root)
    if not dicom_root.is_dir():
        raise FileNotFoundError(f"DICOM root is not a directory: {dicom_root}")
    patient_dict = load_patient_images_from_root(dicom_root, load_pixels=load_pixels)
    return export_patient_dict(
        patient_dict,
        image_root,
        master_manifest,
        dicom_root=dicom_root,
        label_source=label_source,
        label_tag=label_tag,
        label_map=label_map,
        default_label=default_label,
        allowed_modalities=allowed_modalities,
        label_map_path=label_map_path,
        label_csv=label_csv,
        label_csv_join=label_csv_join,
        label_csv_column=label_csv_column,
        recipe=recipe,
    )


def export_from_pickle(
    pickle_path: Path,
    image_root: Path,
    master_manifest: Path,
    *,
    label_source: str = LABEL_SOURCE,
    label_tag: str = LABEL_DICOM_TAG,
    label_map: dict[str, int] | None = None,
    default_label: int | None = None,
    allowed_modalities: list[str] | None = None,
    label_map_path: Path | None = None,
    label_csv: Path | None = None,
    label_csv_join: str = "patient_id",
    label_csv_column: str = "label",
    recipe: PreprocessRecipe | None = None,
) -> pd.DataFrame:
    patient_dict = load_from_disk(pickle_path)
    return export_patient_dict(
        patient_dict,
        image_root,
        master_manifest,
        dicom_root=None,
        label_source=label_source,
        label_tag=label_tag,
        label_map=label_map,
        default_label=default_label,
        allowed_modalities=allowed_modalities,
        label_map_path=label_map_path,
        label_csv=label_csv,
        label_csv_join=label_csv_join,
        label_csv_column=label_csv_column,
        recipe=recipe,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a DICOM folder to PNG + manifest (IDs and labels from the files)."
    )
    ap.add_argument("--config", type=Path, help="Task YAML (configs/kl_grade.yaml or kl_grade)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--dicom-root", type=Path, help="Folder of DICOM files (any nesting)")
    src.add_argument("--pickle", type=Path, help="Optional pickled patient_dict (advanced)")
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--label-source", choices=("auto", "tag", "folder", "csv"))
    ap.add_argument("--label-tag", help="DICOM keyword or 0x hex tag")
    ap.add_argument("--label-csv", type=Path, help="Sidecar grades CSV (label_source=csv)")
    ap.add_argument(
        "--default-label",
        type=int,
        default=None,
        help="Fallback integer label if the files have no class tag or class folders",
    )
    args = ap.parse_args(argv)

    cfg = load_config_yaml(args.config) if args.config else TrainingConfig()
    kwargs = dict(
        label_source=args.label_source or cfg.label_source,
        label_tag=args.label_tag or cfg.label_dicom_tag,
        label_map=cfg.label_map or None,
        default_label=args.default_label,
        allowed_modalities=cfg.allowed_modalities or None,
        label_map_path=cfg.label_map_path,
        label_csv=args.label_csv or cfg.label_csv,
        label_csv_join=cfg.label_csv_join,
        label_csv_column=cfg.label_csv_column,
        recipe=PreprocessRecipe.from_training_config(cfg),
    )
    image_root = args.image_root or cfg.image_root
    manifest = args.manifest or cfg.master_manifest
    try:
        if args.pickle is not None:
            df = export_from_pickle(args.pickle, image_root, manifest, **kwargs)
        else:
            df = export_from_dicom_root(
                args.dicom_root or cfg.dicom_root, image_root, manifest, **kwargs
            )
    except (LabelError, FileNotFoundError, RuntimeError) as exc:
        print(f"Export failed: {exc}")
        return 1
    print(f"Wrote {len(df)} rows -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
