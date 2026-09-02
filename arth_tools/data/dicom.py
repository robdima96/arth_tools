# -*- coding: utf-8 -*-
"""
Load DICOM datasets from a folder tree (optional DICOMDIR).

Patient IDs come from the files themselves (PatientID, else StudyInstanceUID).
The CONTROL BOARD DICOM_ROOT is used when run as __main__.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.fileset import FileSet

from arth_tools.data.labels import patient_id_from_dataset

warnings.filterwarnings("ignore", category=UserWarning)

PatientDict = dict[str, list[Any]]


def load_dicomdir_info(dicomdir_path: str | Path) -> PatientDict:
    """
    Reads the DICOMDIR file and creates a patient dictionary.

    Args:
        dicomdir_path: Path to the directory containing 'DICOMDIR'.

    Returns:
        dict: Keys are PatientID strings, values are empty lists.
    """
    dicomdir_path = Path(dicomdir_path)
    dicomdir_file = dicomdir_path / "DICOMDIR" if dicomdir_path.is_dir() else dicomdir_path
    if not dicomdir_file.is_file():
        raise FileNotFoundError(f"No DICOMDIR file found at: {dicomdir_file}")

    # pydicom 3.x: FileSet replaces read_dicomdir / DicomDir
    fs = FileSet(str(dicomdir_file))
    patient_dict: PatientDict = {}

    for node in fs._tree:
        if node.record_type != "PATIENT":
            continue
        rec = node._record
        try:
            patient_id = rec.PatientID
            if not patient_id:
                continue
            patient_dict[str(patient_id).strip()] = []
        except AttributeError:
            continue  # skip incomplete records

    return patient_dict


def patient_key_from_dataset(ds: Any, *, dicom_root: str | Path | None = None) -> str | None:
    root = Path(dicom_root) if dicom_root is not None else None
    return patient_id_from_dataset(ds, dicom_root=root)


def load_patient_images_from_root(
    dicom_root_dir: str | Path,
    patient_dict: PatientDict | None = None,
    *,
    load_pixels: bool = False,
    require_known_keys: bool = False,
) -> PatientDict:
    """
    Walks through all subfolders of dicom_root_dir and loads DICOM datasets.

    Args:
        dicom_root_dir: Root directory containing any number of subfolders with DICOM files.
        patient_dict: Dictionary from load_dicomdir_info, mapping patient IDs to lists.
            If None, keys are created from each file's PatientID.
        load_pixels: If True, loads pixel data; otherwise skips for speed.
        require_known_keys: If True, only keys already in patient_dict are kept.

    Returns:
        dict: Updated patient_dict with DICOM datasets appended per patient key.
    """
    patient_dict = patient_dict if patient_dict is not None else {}
    valid_keys = set(patient_dict.keys())
    skip_exts = {".txt", ".xml", ".json", ".jpg", ".png", ".gif", ".bmp", ".etl", ".log", ".exe"}

    root_path = Path(dicom_root_dir)
    for root, _, files in os.walk(str(dicom_root_dir)):
        for file in files:
            if file.upper() == "DICOMDIR":
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in skip_exts:
                continue
            full_path = os.path.join(root, file)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ds = pydicom.dcmread(full_path, stop_before_pixels=not load_pixels, force=True)
                key = patient_key_from_dataset(ds, dicom_root=root_path)
                if key is None:
                    continue
                if require_known_keys and valid_keys and key not in valid_keys:
                    continue
                patient_dict.setdefault(key, []).append(ds)
            except (InvalidDicomError, Exception):
                continue  # ignore non-DICOM or unreadable files

    for key, images in patient_dict.items():
        print(f"{key}: {len(images)} images loaded")

    return patient_dict


def save_to_disk(patient_dict: PatientDict, filepath: str | Path) -> Path:
    """Persist patient_dict (lists of pydicom datasets) with pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as fh:
        pickle.dump(patient_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved patient_dict with {len(patient_dict)} keys to {str(filepath)!r}")
    return filepath


def load_from_disk(filepath: str | Path) -> PatientDict:
    """Load patient_dict previously written by save_to_disk."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"No such file: {filepath}")
    with filepath.open("rb") as fh:
        patient_dict = pickle.load(fh)
    print(f"Loaded patient_dict with {len(patient_dict)} keys from {str(filepath)!r}")
    return patient_dict


def write_synthetic_dicom(
    path: str | Path,
    *,
    patient_id: str,
    patient_name: str = "Fixture",
    pixels: Any | None = None,
    modality: str = "US",
    study_description: str | None = None,
    series_description: str | None = None,
    photometric: str = "MONOCHROME2",
    window_center: float | None = None,
    window_width: float | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> Path:
    """Write a tiny DICOM for smoke tests (no PHI)."""
    import numpy as np
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixels is None:
        pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    pixels = np.asarray(pixels)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.Modality = modality
    if study_description is not None:
        ds.StudyDescription = study_description
    if series_description is not None:
        ds.SeriesDescription = series_description
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Rows = int(pixels.shape[0])
    ds.Columns = int(pixels.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    if pixels.dtype == np.uint16 or int(np.max(pixels)) > 255:
        pix = np.asarray(pixels, dtype=np.uint16)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pix.tobytes()
    else:
        pix = np.asarray(pixels, dtype=np.uint8)
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pix.tobytes()
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)
    return path


if __name__ == "__main__":
    from arth_tools.training.config import DICOM_ROOT, DICOMDIR_PATH, PICKLE_DIR

    dicomdir_path = DICOMDIR_PATH
    dicom_root_dir = DICOM_ROOT

    if (Path(dicomdir_path) / "DICOMDIR").is_file() or Path(dicomdir_path).is_file():
        patient_dict = load_dicomdir_info(dicomdir_path)
        patient_dict = load_patient_images_from_root(
            dicom_root_dir,
            patient_dict,
            load_pixels=True,
            require_known_keys=True,
        )
    else:
        print(f"No DICOMDIR at {dicomdir_path}; walking {dicom_root_dir} by PatientID only.")
        patient_dict = load_patient_images_from_root(dicom_root_dir, load_pixels=True)

    total_images = sum(len(datasets) for datasets in patient_dict.values())
    print(
        f"Success: patient_dict loaded in memory -- "
        f"{len(patient_dict)} patients, {total_images} DICOM instances total."
    )
    pickle_path = PICKLE_DIR / "patient_dict.pkl"
    try:
        save_to_disk(patient_dict, pickle_path)
    except OSError as exc:
        print(f"Could not pickle to {pickle_path} ({exc})")
