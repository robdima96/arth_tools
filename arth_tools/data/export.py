# -*- coding: utf-8 -*-
"""
Non-interactive DICOM / pickle → PNG + master manifest.

Default destinations are the CONTROL BOARD E:\\ArthAgent paths. Override with
CLI flags. Labels in the master CSV are placeholders (default_label) until you
edit the file or join a clinical table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from arth_tools.data.dicom import load_from_disk, load_patient_images_from_root
from arth_tools.data.preprocess import array_to_pil
from arth_tools.training.config import (
    FILEPATH_COLUMN,
    IMAGE_ROOT,
    LABEL_COLUMN,
    MANIFEST_MASTER,
    PATIENT_ID_COLUMN,
)

FILEPATH_COL = FILEPATH_COLUMN
PATIENT_COL = PATIENT_ID_COLUMN
LABEL_COL = LABEL_COLUMN
MODALITY_COL = "modality"
CHANNELS_COL = "channels"


def _pid_clean(key: str) -> str:
    return str(key).replace(os.sep, "_").replace(" ", "_")


def _slice_to_png(ds: Any, dest: Path) -> dict[str, Any] | None:
    try:
        if not hasattr(ds, "pixel_array"):
            return None
        arr = ds.pixel_array
        pil = array_to_pil(arr, rgb=True)
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil.save(dest)
    modality = str(ds.get("Modality", "") or "")
    return {
        FILEPATH_COL: str(dest.resolve()),
        MODALITY_COL: modality,
        CHANNELS_COL: 3,
    }


def export_patient_dict(
    patient_dict: dict[str, list[Any]],
    image_root: Path,
    master_manifest: Path,
    *,
    default_label: int = 0,
    filepath_col: str = FILEPATH_COL,
    patient_col: str = PATIENT_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("EXPORT SLICES TO PNG")
    print("=" * 70)

    rows: list[dict[str, Any]] = []
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    for pkey, series in patient_dict.items():
        pid = _pid_clean(pkey)
        if not isinstance(series, list):
            continue
        n_ok = 0
        for idx, ds in enumerate(series):
            png = image_root / pid / f"slice_{idx:04d}.png"
            meta = _slice_to_png(ds, png)
            if meta is None:
                continue
            rows.append(
                {
                    filepath_col: meta[FILEPATH_COL],
                    patient_col: pid,
                    label_col: int(default_label),
                    MODALITY_COL: meta[MODALITY_COL],
                    CHANNELS_COL: meta[CHANNELS_COL],
                }
            )
            n_ok += 1
        print(f"  {pid}: {n_ok} slices")

    if not rows:
        raise RuntimeError("No slices exported — check pixel_array availability.")

    df = pd.DataFrame(rows)
    master_manifest = Path(master_manifest)
    master_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(master_manifest, index=False)
    print(f"Wrote master manifest ({len(df)} rows) -> {master_manifest}")
    return df


def export_from_dicom_root(
    dicom_root: Path,
    image_root: Path,
    master_manifest: Path,
    *,
    default_label: int = 0,
    load_pixels: bool = True,
) -> pd.DataFrame:
    patient_dict = load_patient_images_from_root(dicom_root, load_pixels=load_pixels)
    return export_patient_dict(
        patient_dict,
        image_root,
        master_manifest,
        default_label=default_label,
    )


def export_from_pickle(
    pickle_path: Path,
    image_root: Path,
    master_manifest: Path,
    *,
    default_label: int = 0,
) -> pd.DataFrame:
    patient_dict = load_from_disk(pickle_path)
    return export_patient_dict(
        patient_dict,
        image_root,
        master_manifest,
        default_label=default_label,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export DICOM tree or pickle to PNG + manifest.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--dicom-root", type=Path)
    src.add_argument("--pickle", type=Path)
    ap.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_MASTER)
    ap.add_argument("--default-label", type=int, default=0)
    args = ap.parse_args(argv)

    if args.dicom_root is not None:
        df = export_from_dicom_root(
            args.dicom_root, args.image_root, args.manifest, default_label=args.default_label
        )
    elif args.pickle is not None:
        df = export_from_pickle(args.pickle, args.image_root, args.manifest, default_label=args.default_label)
    else:
        from arth_tools.training.config import DICOM_ROOT

        df = export_from_dicom_root(
            DICOM_ROOT, args.image_root, args.manifest, default_label=args.default_label
        )
    print(f"Wrote {len(df)} rows -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
