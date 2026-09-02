# -*- coding: utf-8 -*-
"""
Standing AP x-ray DICOM → annotated DICOM + HKA angle.

    python -m arth_tools hka --config hka --dicom-root /path/to/xrays
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError

from arth_tools.data.labels import patient_id_from_dataset
from arth_tools.hka.annotate import write_hka_dicom
from arth_tools.hka.config import HKAConfig, load_hka_config
from arth_tools.hka.landmarks import measure_hka


SKIP_EXTS = {".txt", ".xml", ".json", ".jpg", ".png", ".gif", ".bmp", ".etl", ".log", ".exe"}


def iter_dicom_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, _, files in os.walk(str(root)):
        for name in files:
            if name.upper() == "DICOMDIR":
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in SKIP_EXTS:
                continue
            found.append(Path(dirpath) / name)
    return found


def _relative_stem(path: Path, root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        rel = Path(path.name)
    return rel.with_suffix("")


def process_file(
    path: Path,
    cfg: HKAConfig,
    *,
    dicom_root: Path,
) -> dict[str, Any]:
    try:
        ds = pydicom.dcmread(str(path), force=True)
    except (InvalidDicomError, Exception) as exc:
        return {"source": str(path), "ok": False, "message": f"unreadable: {exc}"}
    mod = str(ds.get("Modality", "") or "").strip().upper()
    if cfg.allowed_modalities and mod and mod not in cfg.allowed_modalities:
        return {
            "source": str(path),
            "ok": False,
            "skipped": True,
            "message": f"modality {mod or 'empty'} not in {cfg.allowed_modalities}",
        }
    result = measure_hka(ds, max_hips=cfg.max_hips, bone_percentile=cfg.bone_percentile)
    pid = patient_id_from_dataset(ds, dicom_root=dicom_root) or "unknown"
    out_rel = _relative_stem(path, dicom_root)
    dest = cfg.output_dir / str(pid) / f"{out_rel.name}_hka.dcm"
    row: dict[str, Any] = {
        "source": str(path),
        "patient_id": pid,
        "modality": mod,
        "ok": result.ok,
        "message": result.message,
        "summary": result.summary_text(),
        "output_dicom": "",
        "n_limbs": len(result.limbs),
    }
    if result.ok and cfg.overlay:
        dest = write_hka_dicom(ds, result, dest)
        row["output_dicom"] = str(dest)
    elif result.ok:
        dest.parent.mkdir(parents=True, exist_ok=True)
        row["output_dicom"] = ""
    for i, limb in enumerate(result.limbs):
        prefix = limb.laterality if limb.laterality != "unknown" else f"limb{i}"
        row[f"{prefix}_hka_deg"] = round(limb.hka_deg, 3)
        row[f"{prefix}_deviation_deg"] = round(limb.deviation_deg, 3)
        row[f"{prefix}_alignment"] = limb.alignment
    row["limbs"] = [limb.to_dict() for limb in result.limbs]
    return row


def run_hka(cfg: HKAConfig) -> dict[str, Any]:
    root = Path(cfg.dicom_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM root is not a directory: {root}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_dicom_paths(root)
    print("\n" + "=" * 70)
    print("HKA (standing AP x-ray)")
    print("=" * 70)
    print(f"Input:  {root}")
    print(f"Output: {cfg.output_dir}")
    print(f"Files:  {len(paths)}")

    rows: list[dict[str, Any]] = []
    n_ok = 0
    n_skip = 0
    for path in paths:
        rec = process_file(path, cfg, dicom_root=root)
        rows.append(rec)
        if rec.get("skipped"):
            n_skip += 1
            continue
        status = "OK" if rec.get("ok") else "FAIL"
        print(f"  [{status}] {path.name}: {rec.get('summary') or rec.get('message')}")
        if rec.get("ok"):
            n_ok += 1

    csv_path = cfg.output_dir / "hka_angles.csv"
    json_path = cfg.output_dir / "hka_angles.json"
    flat_keys = [
        "source",
        "patient_id",
        "modality",
        "ok",
        "n_limbs",
        "summary",
        "message",
        "output_dicom",
        "right_hka_deg",
        "right_deviation_deg",
        "right_alignment",
        "left_hka_deg",
        "left_deviation_deg",
        "left_alignment",
        "unknown_hka_deg",
        "unknown_deviation_deg",
        "unknown_alignment",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dicom_root": str(root),
        "output_dir": str(cfg.output_dir),
        "n_files": len(paths),
        "n_ok": n_ok,
        "n_skipped_modality": n_skip,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {n_ok} annotated DICOMs; {n_skip} skipped (modality)")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Hip-knee-ankle angle on standing AP x-ray DICOMs (writes annotated DICOM + CSV)."
    )
    ap.add_argument("--config", type=Path, help="Task YAML (configs/hka.yaml or hka)")
    ap.add_argument("--dicom-root", type=Path, help="Folder of standing AP x-ray DICOMs")
    ap.add_argument("--out-dir", type=Path, help="Annotated DICOM + angle tables")
    ap.add_argument("--no-overlay", action="store_true", help="Measure only; do not write annotated DICOM")
    args = ap.parse_args(argv)

    cfg = load_hka_config(args.config)
    if args.dicom_root:
        cfg.dicom_root = args.dicom_root
    if args.out_dir:
        cfg.output_dir = args.out_dir
    if args.no_overlay:
        cfg.overlay = False
    try:
        run_hka(cfg)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"HKA failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
