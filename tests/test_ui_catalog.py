# -*- coding: utf-8 -*-
"""Hierarchical tool catalog for the Streamlit UI."""

from __future__ import annotations

from arth_tools.paths import CONFIGS_DIR
from arth_tools.ui.catalog import load_catalog, normalize_info


def test_catalog_tree_has_knee_modalities() -> None:
    catalog = load_catalog()
    assert [a.anatomy_id for a in catalog.anatomies] == ["knee"]
    knee = catalog.anatomies[0]
    assert [m.modality_id for m in knee.modalities] == ["xray", "ultrasound", "mri"]
    assert knee.modalities[0].modality_name == "X-ray"
    assert knee.modalities[1].modality_name == "Ultrasound"
    assert knee.modalities[2].modality_name == "MRI"
    assert knee.modalities[2].tasks == []


def test_kl_grade_has_cnn_and_gu_tools() -> None:
    catalog = load_catalog()
    xray = catalog.anatomies[0].modalities[0]
    tasks = {t.task_id: t for t in xray.tasks}
    assert "kl_grade" in tasks
    tools = {t.tool_id: t for t in tasks["kl_grade"].tools}
    assert set(tools) == {"kl_grade_cnn", "kl_grade_gu2022"}
    assert tools["kl_grade_cnn"].train_enabled is True
    assert tools["kl_grade_cnn"].preprocess_consume == "cnn_full"
    assert tools["kl_grade_gu2022"].preprocess_consume == "external"
    assert tools["kl_grade_cnn"].infer_backend == "run_or_bundle"
    assert tools["kl_grade_cnn"].num_classes == 5
    assert tools["kl_grade_gu2022"].train_enabled is False
    assert tools["kl_grade_gu2022"].infer_backend == "stub"
    assert tools["kl_grade_gu2022"].info["citations"]


def test_hka_and_omeract_placement() -> None:
    catalog = load_catalog()
    knee = catalog.anatomies[0]
    xray = next(m for m in knee.modalities if m.modality_id == "xray")
    us = next(m for m in knee.modalities if m.modality_id == "ultrasound")
    hka_task = next(t for t in xray.tasks if t.task_id == "hka")
    assert len(hka_task.tools) == 1
    hka = hka_task.tools[0]
    assert hka.tool_id == "hka_classical"
    assert hka.infer_backend == "classical"
    assert hka.preprocess_consume == "decode_roi"
    assert hka.train_enabled is False
    omeract_task = next(t for t in us.tasks if t.task_id == "omeract_synovitis")
    assert len(omeract_task.tools) == 1
    omeract = omeract_task.tools[0]
    assert omeract.tool_id == "omeract_cnn"
    assert omeract.train_enabled is True
    assert omeract.infer_backend == "run_or_bundle"
    assert omeract.preprocess_consume == "cnn_full"
    assert omeract.num_classes == 4


def test_catalog_yaml_is_not_a_tool_recipe() -> None:
    catalog = load_catalog()
    tools = catalog.all_tools()
    assert all(t.tool_id != "catalog" for t in tools)
    assert all(t.yaml_path.name != "catalog.yaml" for t in tools)
    assert (CONFIGS_DIR / "catalog.yaml").is_file()


def test_each_tool_has_info_lists() -> None:
    for tool in load_catalog().all_tools():
        assert isinstance(tool.info, dict)
        assert isinstance(tool.info.get("citations"), list)
        assert isinstance(tool.info.get("models"), list)
        assert "summary" in tool.info
        assert "methods" in tool.info
        assert "parameters" in tool.info
        assert tool.info["summary"]


def test_mri_still_present_when_omitted_from_yaml(tmp_path) -> None:
    (tmp_path / "catalog.yaml").write_text(
        "anatomies:\n"
        "  - id: knee\n"
        "    name: Knee\n"
        "    modalities:\n"
        "      - id: xray\n"
        "        name: X-ray\n"
        "        tasks: []\n",
        encoding="utf-8",
    )
    catalog = load_catalog(configs_dir=tmp_path)
    knee = catalog.anatomies[0]
    assert [m.modality_id for m in knee.modalities] == ["xray", "ultrasound", "mri"]
    assert all(m.tasks == [] for m in knee.modalities)


def test_normalize_info_defaults_summary() -> None:
    info = normalize_info({}, task_name="Demo task")
    assert info["summary"] == "Demo task"
    assert info["citations"] == []
    assert info["models"] == []
