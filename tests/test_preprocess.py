# -*- coding: utf-8 -*-
"""Train/serve preprocess recipe and staged decode."""

from __future__ import annotations

import numpy as np
import pydicom

from arth_tools.data.dicom import write_synthetic_dicom
from arth_tools.data.preprocess import (
    PreprocessRecipe,
    apply_stages_from_dataset,
    array_to_pil,
    decode_dicom,
    dicom_pixels_to_pil,
    dicom_to_display_u8,
    remaining_stages,
    stages_for_mode,
    pil_to_float_hwc,
)
from arth_tools.hka.landmarks import dataset_to_gray
from arth_tools.roi.infer import crop_image, crop_xyxy


def test_recipe_roundtrip() -> None:
    recipe = PreprocessRecipe(
        resize_width=32,
        resize_height=16,
        zscore_mean=0.4,
        zscore_std=0.1,
        export_mode="minmax_u8_rgb",
        preprocess_consume="cnn_full",
        baked_stages=["decode", "roi", "intensity"],
        modality_id="xray",
    )
    restored = PreprocessRecipe.from_dict(recipe.to_dict())
    assert restored.resize_width == 32
    assert restored.resize_height == 16
    assert restored.zscore_mean == 0.4
    assert restored.export_mode == "minmax_u8_rgb"
    assert restored.preprocess_consume == "cnn_full"
    assert restored.baked_stages == ["decode", "roi", "intensity"]


def test_old_preprocess_json_loads() -> None:
    recipe = PreprocessRecipe.from_dict(
        {
            "resize_width": 16,
            "resize_height": 16,
            "export_mode": "minmax_u8_rgb",
            "zscore_mean": 0.2,
            "zscore_std": 0.1,
        }
    )
    assert recipe.is_legacy_minmax()
    assert recipe.preprocess_consume == "cnn_full"
    assert "decode" in recipe.baked_stages


def test_dicom_pixels_match_exported_png_operators() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    recipe = PreprocessRecipe(
        resize_width=4,
        resize_height=4,
        replicate_grayscale_to_rgb=True,
        zscore_mean=None,
        zscore_std=None,
        export_mode="minmax_u8_rgb",
    )
    from_png = pil_to_float_hwc(array_to_pil(arr, rgb=True), recipe)
    from_dicom = pil_to_float_hwc(dicom_pixels_to_pil(arr, recipe), recipe)
    np.testing.assert_allclose(from_png, from_dicom)
    assert from_png.shape == (4, 4, 3)
    assert from_png.max() <= 1.0 + 1e-6


def test_png_skips_baked_stages_no_double_minmax() -> None:
    arr = np.linspace(10, 200, 16, dtype=np.float64).reshape(4, 4)
    recipe = PreprocessRecipe(
        resize_width=4,
        resize_height=4,
        export_mode="dicom_display_u8_rgb",
        preprocess_consume="cnn_full",
        baked_stages=["decode", "roi", "intensity"],
        modality_id="xray",
        zscore_mean=None,
        zscore_std=None,
    )
    u8 = array_to_pil(arr, rgb=True)
    once = pil_to_float_hwc(u8, recipe)
    twice = pil_to_float_hwc(array_to_pil((once * 255.0).astype(np.uint8)[..., 0], rgb=True), recipe)
    np.testing.assert_allclose(once, twice, atol=2 / 255.0)


def test_monochrome1_inverted_once(tmp_path) -> None:
    pixels = np.array([[0, 50], [100, 200]], dtype=np.uint8)
    path = write_synthetic_dicom(
        tmp_path / "m1.dcm",
        patient_id="P",
        pixels=pixels,
        modality="CR",
        photometric="MONOCHROME1",
    )
    ds = pydicom.dcmread(str(path), force=True)
    arr, _meta = decode_dicom(ds, apply_voi=False)
    np.testing.assert_allclose(arr, 200.0 - pixels.astype(np.float64))
    gray = dataset_to_gray(ds)
    np.testing.assert_allclose(gray, arr)


def test_external_mode_leaves_pixels(tmp_path) -> None:
    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    path = write_synthetic_dicom(tmp_path / "e.dcm", patient_id="P", pixels=pixels, modality="CR")
    ds = pydicom.dcmread(str(path), force=True)
    recipe = PreprocessRecipe(preprocess_consume="external")
    out = apply_stages_from_dataset(ds, recipe)
    np.testing.assert_array_equal(out, pixels)
    assert remaining_stages("external", []) == []
    assert stages_for_mode("cnn_full")[-1] == "tensor"


def test_hka_decode_roi_is_not_uint8(tmp_path) -> None:
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    path = write_synthetic_dicom(tmp_path / "h.dcm", patient_id="P", pixels=pixels, modality="CR")
    ds = pydicom.dcmread(str(path), force=True)
    gray = dataset_to_gray(ds)
    assert gray.dtype == np.float64
    assert gray.shape == (2, 2)


def test_windowed_xray_decode(tmp_path) -> None:
    pixels = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
    path = write_synthetic_dicom(
        tmp_path / "w.dcm",
        patient_id="P",
        pixels=pixels,
        modality="DX",
        window_center=800,
        window_width=400,
        rescale_slope=1.0,
        rescale_intercept=0.0,
    )
    ds = pydicom.dcmread(str(path), force=True)
    recipe = PreprocessRecipe(preprocess_consume="cnn_full", modality_id="xray", baked_stages=[])
    u8, meta = dicom_to_display_u8(ds, recipe)
    assert u8.dtype == np.uint8
    assert u8.shape == (4, 4)
    assert meta["modality"] == "DX"
    assert "manufacturer" in meta


def test_identity_crop_without_weights() -> None:
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = crop_image(arr, "xray")
    np.testing.assert_array_equal(out, arr)
    cropped = crop_xyxy(arr, (1, 1, 4, 4))
    assert cropped.shape == (3, 3)
