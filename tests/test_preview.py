# -*- coding: utf-8 -*-
"""Paired loaded vs preprocessed image previews."""

from __future__ import annotations

import numpy as np

from arth_tools.data.dicom import write_synthetic_dicom
from arth_tools.data.preprocess import PreprocessRecipe
from arth_tools.data.preview import sample_preview_rows


def _recipe(*, width: int = 8, height: int = 8) -> PreprocessRecipe:
    return PreprocessRecipe(
        resize_width=width,
        resize_height=height,
        replicate_grayscale_to_rgb=True,
        input_channels=3,
        zscore_mean=None,
        zscore_std=None,
        export_mode="dicom_display_u8_rgb",
        preprocess_consume="cnn_full",
        baked_stages=[],
        modality_id="xray",
        roi_model_id="xray",
    )


def test_sample_fewer_than_n_returns_all_paired(tmp_path) -> None:
    paths = []
    for i in range(4):
        paths.append(
            write_synthetic_dicom(
                tmp_path / f"p{i}.dcm",
                patient_id=f"P{i}",
                pixels=np.arange(16, dtype=np.uint8).reshape(4, 4) + i,
                modality="CR",
            )
        )
    rows = sample_preview_rows(paths, recipe=_recipe(), n=10, seed=0)
    assert len(rows) == 4
    assert {r.path.resolve() for r in rows} == {p.resolve() for p in paths}
    for row in rows:
        assert row.processed_u8 is not None
        assert row.loaded_u8.dtype == np.uint8
        assert row.processed_u8.dtype == np.uint8
        assert row.patient_id.startswith("P")


def test_processed_size_matches_recipe(tmp_path) -> None:
    path = write_synthetic_dicom(
        tmp_path / "a.dcm",
        patient_id="P",
        pixels=np.arange(32 * 16, dtype=np.uint8).reshape(32, 16),
        modality="CR",
    )
    recipe = _recipe(width=8, height=12)
    rows = sample_preview_rows([path], recipe=recipe, n=10, seed=1)
    assert len(rows) == 1
    loaded = rows[0].loaded_u8
    processed = rows[0].processed_u8
    assert loaded.shape[:2] == (32, 16)
    assert processed is not None
    assert processed.shape[0] == 12
    assert processed.shape[1] == 8


def test_skips_unreadable_paths(tmp_path) -> None:
    good = write_synthetic_dicom(
        tmp_path / "ok.dcm",
        patient_id="P",
        pixels=np.arange(16, dtype=np.uint8).reshape(4, 4),
        modality="CR",
    )
    junk = tmp_path / "junk.txt"
    junk.write_text("not an image", encoding="utf-8")
    rows = sample_preview_rows([junk, good], recipe=_recipe(), n=10, seed=0, shuffle=False)
    assert len(rows) == 1
    assert rows[0].path.resolve() == good.resolve()


def test_infer_preview_has_no_processed(tmp_path) -> None:
    path = write_synthetic_dicom(
        tmp_path / "b.dcm",
        patient_id="P",
        pixels=np.arange(16, dtype=np.uint8).reshape(4, 4),
        modality="CR",
    )
    rows = sample_preview_rows([path], recipe=None, n=10, seed=0)
    assert len(rows) == 1
    assert rows[0].processed_u8 is None
    assert rows[0].loaded_u8.shape == (4, 4)
