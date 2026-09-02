# -*- coding: utf-8 -*-
"""Resolve patient IDs and class labels from DICOM tags, class folders, or a CSV join."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from arth_tools.training.config import LABEL_DICOM_TAG, LABEL_MAP, LABEL_SOURCE


class LabelError(ValueError):
    pass


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def tag_value(ds: Any, tag: str) -> str | None:
    """Read a DICOM keyword or 0x hex tag as a stripped string."""
    if not tag:
        return None
    raw = tag.strip()
    try:
        if raw.lower().startswith("0x"):
            value = ds.get(int(raw, 16), None)
        else:
            value = ds.get(raw, None)
    except (TypeError, ValueError, AttributeError):
        return None
    if value is None:
        return None
    elem_value = getattr(value, "value", value)
    return _nonempty_str(elem_value)


def source_path(ds: Any) -> Path | None:
    name = getattr(ds, "filename", None)
    if not name:
        return None
    return Path(name)


def folder_component(ds: Any, dicom_root: Path | None) -> str | None:
    """First path component under dicom_root, or the parent folder name."""
    path = source_path(ds)
    if path is None:
        return None
    if dicom_root is not None:
        try:
            rel = path.resolve().relative_to(Path(dicom_root).resolve())
        except ValueError:
            rel = Path(path.name)
        if len(rel.parts) >= 2:
            return rel.parts[0]
        return None
    parent = path.parent.name
    return parent or None


def patient_id_from_dataset(ds: Any, *, dicom_root: Path | None = None) -> str | None:
    """PatientID, else StudyInstanceUID, else a folder name. Never PatientName."""
    pid = _nonempty_str(ds.get("PatientID", None) if hasattr(ds, "get") else None)
    if pid:
        return pid
    study = _nonempty_str(ds.get("StudyInstanceUID", None) if hasattr(ds, "get") else None)
    if study:
        return study
    path = source_path(ds)
    if path is None:
        return None
    if dicom_root is not None:
        try:
            rel = path.resolve().relative_to(Path(dicom_root).resolve())
            if len(rel.parts) >= 2:
                return rel.parts[-2]
        except ValueError:
            pass
    parent = path.parent.name
    return parent or path.stem


def encode_label_strings(raws: list[str], mapping: dict[str, int] | None = None) -> tuple[list[int], dict[str, int]]:
    if mapping:
        encoded: list[int] = []
        for raw in raws:
            if raw not in mapping:
                raise LabelError(f"Unmapped label {raw!r}; known keys: {sorted(mapping)}")
            encoded.append(int(mapping[raw]))
        return encoded, dict(mapping)

    unique = sorted(set(raws))
    if unique and all(_is_int(u) for u in unique):
        resolved = {u: int(u) for u in unique}
    else:
        resolved = {u: i for i, u in enumerate(unique)}
    return [resolved[r] for r in raws], resolved


def _is_int(text: str) -> bool:
    try:
        int(text)
    except (TypeError, ValueError):
        return False
    return True


def folders_look_like_classes(folder_vals: list[str | None], patient_ids: list[str]) -> bool:
    folders = {f for f in folder_vals if f}
    patients = {p for p in patient_ids if p}
    if len(folders) < 2:
        return False
    if folders == patients:
        return False
    return True


def load_csv_label_table(
    path: Path,
    *,
    join_col: str,
    label_col: str = "label",
) -> dict[str, str]:
    dest = Path(path)
    if not dest.is_file():
        raise LabelError(f"label_source=csv but label CSV is missing: {dest}")
    df = pd.read_csv(dest)
    if join_col not in df.columns:
        raise LabelError(f"Label CSV {dest.name} missing join column {join_col!r}; have {list(df.columns)}")
    if label_col not in df.columns:
        raise LabelError(f"Label CSV {dest.name} missing label column {label_col!r}; have {list(df.columns)}")
    table: dict[str, str] = {}
    for _, row in df.iterrows():
        key = _nonempty_str(row[join_col])
        lab = _nonempty_str(row[label_col])
        if key is None or lab is None:
            continue
        table[key] = lab
    if not table:
        raise LabelError(f"Label CSV {dest} has no usable {join_col}/{label_col} rows.")
    return table


def csv_vals_for_keys(keys: list[str], table: dict[str, str]) -> list[str | None]:
    return [table.get(k) for k in keys]


def decide_label_source(
    tag_vals: list[str | None],
    folder_vals: list[str | None],
    patient_ids: list[str],
    *,
    source: Literal["auto", "tag", "folder", "csv"] = LABEL_SOURCE,  # type: ignore[assignment]
    tag_name: str = LABEL_DICOM_TAG,
    csv_vals: list[str | None] | None = None,
) -> Literal["tag", "folder", "csv"]:
    if source == "csv":
        if not csv_vals or not any(csv_vals):
            raise LabelError(
                "LABEL_SOURCE='csv' but no rows matched the join key. "
                "Check label_csv, label_csv_join (patient_id or study_uid), and the CSV columns."
            )
        return "csv"
    if source == "tag":
        if not any(tag_vals):
            raise LabelError(
                f"No values in DICOM tag {tag_name!r}. "
                "Set LABEL_DICOM_TAG to a tag that holds the class, or use LABEL_SOURCE='folder'."
            )
        return "tag"
    if source == "folder":
        if not any(folder_vals):
            raise LabelError(
                "LABEL_SOURCE='folder' but files are not nested under class folders. "
                "Expected dicom_root/<class>/... (e.g. dicoms/0/patient/slice.dcm)."
            )
        return "folder"

    if any(tag_vals):
        return "tag"
    if folders_look_like_classes(folder_vals, patient_ids):
        return "folder"
    raise LabelError(
        "Could not read class labels from the DICOM tree. "
        f"Tag {tag_name!r} is empty and folders look like one directory per patient. "
        "Put files in class folders (dicoms/0/..., dicoms/1/...), "
        "set LABEL_DICOM_TAG to a tag that holds the class, "
        "or pass --default-label only for a throwaway run."
    )


def assign_labels(
    tag_vals: list[str | None],
    folder_vals: list[str | None],
    patient_ids: list[str],
    *,
    source: Literal["auto", "tag", "folder", "csv"] = LABEL_SOURCE,  # type: ignore[assignment]
    tag_name: str = LABEL_DICOM_TAG,
    label_map: dict[str, int] | None = None,
    default_label: int | None = None,
    csv_vals: list[str | None] | None = None,
) -> tuple[list[int], dict[str, int], str, list[str]]:
    mapping = dict(label_map or LABEL_MAP)
    if source == "auto" and not any(tag_vals) and not folders_look_like_classes(folder_vals, patient_ids):
        if default_label is not None:
            n = len(tag_vals)
            raws = [str(default_label)] * n
            return [int(default_label)] * n, {str(default_label): int(default_label)}, "default", raws
        decide_label_source(tag_vals, folder_vals, patient_ids, source=source, tag_name=tag_name)

    chosen = decide_label_source(
        tag_vals,
        folder_vals,
        patient_ids,
        source=source,
        tag_name=tag_name,
        csv_vals=csv_vals,
    )
    raws: list[str] = []
    if chosen == "csv":
        pool = list(csv_vals or [])
    elif chosen == "tag":
        pool = tag_vals
    else:
        pool = folder_vals
    for i, raw in enumerate(pool):
        if raw is None:
            if default_label is not None:
                raws.append(str(default_label))
                continue
            raise LabelError(
                f"Missing {chosen} label on slice {i} (patient {patient_ids[i]!r}). "
                "Fill the DICOM tag, use class folders, a label CSV, or pass --default-label."
            )
        raws.append(raw)
    encoded, resolved = encode_label_strings(raws, mapping or None)
    return encoded, resolved, chosen, raws
