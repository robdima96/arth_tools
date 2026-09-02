# -*- coding: utf-8 -*-
"""CSV label join and encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arth_tools.data.labels import (
    LabelError,
    assign_labels,
    csv_vals_for_keys,
    encode_label_strings,
    load_csv_label_table,
)


def test_encode_int_strings_stay_ints() -> None:
    encoded, mapping = encode_label_strings(["0", "4", "2"])
    assert mapping == {"0": 0, "2": 2, "4": 4}
    assert encoded == [0, 4, 2]


def test_csv_label_table_patient_join(tmp_path: Path) -> None:
    csv_path = tmp_path / "grades.csv"
    pd.DataFrame(
        {"patient_id": ["P1", "P2", "P3"], "label": ["0", "3", "4"]}
    ).to_csv(csv_path, index=False)
    table = load_csv_label_table(csv_path, join_col="patient_id", label_col="label")
    assert table["P2"] == "3"
    vals = csv_vals_for_keys(["P1", "P9", "P3"], table)
    assert vals == ["0", None, "4"]


def test_assign_labels_from_csv() -> None:
    encoded, mapping, source, raws = assign_labels(
        [None, None],
        [None, None],
        ["P1", "P2"],
        source="csv",
        csv_vals=["2", "4"],
        label_map={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
    )
    assert source == "csv"
    assert encoded == [2, 4]
    assert raws == ["2", "4"]
    assert mapping["4"] == 4


def test_csv_missing_join_raises() -> None:
    with pytest.raises(LabelError, match="csv"):
        assign_labels(
            [None],
            [None],
            ["P1"],
            source="csv",
            csv_vals=[None],
        )


def test_csv_export_join_on_patient_id(tmp_path: Path) -> None:
    from arth_tools.data.dicom import write_synthetic_dicom
    from arth_tools.data.export import export_from_dicom_root

    dicom_dir = tmp_path / "dicoms"
    for pid, desc in (("P1", "ignored"), ("P2", "ignored")):
        write_synthetic_dicom(
            dicom_dir / pid / "img.dcm",
            patient_id=pid,
            patient_name="Fixture",
            pixels=np.full((8, 8), 40, dtype=np.uint8),
            modality="CR",
            study_description=desc,
        )
    csv_path = tmp_path / "grades.csv"
    pd.DataFrame({"patient_id": ["P1", "P2"], "label": ["0", "3"]}).to_csv(csv_path, index=False)
    df = export_from_dicom_root(
        dicom_dir,
        tmp_path / "png",
        tmp_path / "manifest.csv",
        label_source="csv",
        label_csv=csv_path,
        label_csv_join="patient_id",
        label_map={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
    )
    assert set(df["label"].tolist()) == {0, 3}
    assert set(df["label_raw"].tolist()) == {"0", "3"}
