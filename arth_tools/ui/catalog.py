# -*- coding: utf-8 -*-
"""Load the Streamlit library tree from configs/catalog.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from arth_tools.paths import CONFIGS_DIR

EMPTY_INFO: dict[str, Any] = {
    "summary": "",
    "methods": "",
    "parameters": {},
    "citations": [],
    "models": [],
    "notes": "",
}

_PARAM_KEYS = (
    "architecture_id",
    "num_classes",
    "modality",
    "anatomy",
    "loss_name",
    "output_type",
    "input_height",
    "input_width",
    "preprocess_consume",
)

InferBackend = Literal["run_or_bundle", "classical", "stub"]

REQUIRED_MODALITIES: tuple[tuple[str, str], ...] = (
    ("xray", "X-ray"),
    ("ultrasound", "Ultrasound"),
    ("mri", "MRI"),
)

CATALOG_FILENAME = "catalog.yaml"
VALID_INFER = frozenset({"run_or_bundle", "classical", "stub"})


@dataclass
class ToolEntry:
    tool_id: str
    tool_name: str
    task_id: str
    task_name: str
    anatomy_id: str
    anatomy_name: str
    modality_id: str
    modality_name: str
    yaml_path: Path
    train_enabled: bool
    infer_backend: InferBackend
    num_classes: int | None = None
    architecture_id: str | None = None
    preprocess_consume: str = "cnn_full"
    info: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_INFO))


@dataclass
class TaskEntry:
    task_id: str
    task_name: str
    tools: list[ToolEntry]


@dataclass
class ModalityEntry:
    modality_id: str
    modality_name: str
    tasks: list[TaskEntry]


@dataclass
class AnatomyEntry:
    anatomy_id: str
    anatomy_name: str
    modalities: list[ModalityEntry]


@dataclass
class Catalog:
    anatomies: list[AnatomyEntry]

    def all_tools(self) -> list[ToolEntry]:
        tools: list[ToolEntry] = []
        for anatomy in self.anatomies:
            for modality in anatomy.modalities:
                for task in modality.tasks:
                    tools.extend(task.tools)
        return tools


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_info(raw: Any, *, task_name: str = "") -> dict[str, Any]:
    data = dict(EMPTY_INFO)
    if isinstance(raw, dict):
        data["summary"] = str(raw.get("summary") or "")
        data["methods"] = str(raw.get("methods") or "")
        data["notes"] = str(raw.get("notes") or "")
        params = raw.get("parameters")
        data["parameters"] = dict(params) if isinstance(params, dict) else {}
        cites = raw.get("citations")
        data["citations"] = list(cites) if isinstance(cites, list) else []
        models = raw.get("models")
        data["models"] = list(models) if isinstance(models, list) else []
    if not data["summary"] and task_name:
        data["summary"] = task_name
    return data


def yaml_parameters(raw: dict[str, Any], info_params: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _PARAM_KEYS:
        if key in raw and raw[key] not in (None, ""):
            out[key] = raw[key]
    if info_params:
        out.update(info_params)
    return out


def _load_recipe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Catalog recipe not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Catalog recipe is not a mapping: {path}")
    return raw


def _parse_infer(raw: Any, *, tool_id: str) -> InferBackend:
    value = str(raw or "").strip()
    if value not in VALID_INFER:
        raise ValueError(
            f"Tool {tool_id!r} has infer={raw!r}; expected one of {sorted(VALID_INFER)}"
        )
    return value  # type: ignore[return-value]


def _ensure_modalities(parsed: list[ModalityEntry]) -> list[ModalityEntry]:
    by_id = {m.modality_id: m for m in parsed}
    out: list[ModalityEntry] = []
    seen: set[str] = set()
    for mid, default_name in REQUIRED_MODALITIES:
        seen.add(mid)
        if mid in by_id:
            entry = by_id[mid]
            name = entry.modality_name or default_name
            out.append(ModalityEntry(modality_id=mid, modality_name=name, tasks=entry.tasks))
        else:
            out.append(ModalityEntry(modality_id=mid, modality_name=default_name, tasks=[]))
    for entry in parsed:
        if entry.modality_id not in seen:
            out.append(entry)
    return out


def _build_tool(
    spec: dict[str, Any],
    *,
    configs_dir: Path,
    task_id: str,
    task_name: str,
    anatomy_id: str,
    anatomy_name: str,
    modality_id: str,
    modality_name: str,
) -> ToolEntry:
    tool_id = str(spec.get("id") or "").strip()
    if not tool_id:
        raise ValueError(f"Tool under task {task_id!r} is missing id")
    recipe_name = str(spec.get("recipe") or "").strip()
    if not recipe_name:
        raise ValueError(f"Tool {tool_id!r} is missing recipe")
    yaml_path = configs_dir / recipe_name
    raw = _load_recipe(yaml_path)
    display_task = str(raw.get("task_name") or task_name)
    info = normalize_info(raw.get("info"), task_name=display_task or tool_id)
    info["parameters"] = yaml_parameters(raw, info.get("parameters") or {})
    infer_backend = _parse_infer(spec.get("infer"), tool_id=tool_id)
    consume = str(raw.get("preprocess_consume") or "").strip()
    if not consume:
        consume = {"run_or_bundle": "cnn_full", "classical": "decode_roi", "stub": "external"}.get(
            infer_backend, "cnn_full"
        )
    return ToolEntry(
        tool_id=tool_id,
        tool_name=str(spec.get("name") or tool_id),
        task_id=task_id,
        task_name=task_name,
        anatomy_id=anatomy_id,
        anatomy_name=anatomy_name,
        modality_id=modality_id,
        modality_name=modality_name,
        yaml_path=yaml_path,
        train_enabled=bool(spec.get("train")),
        infer_backend=infer_backend,
        num_classes=_as_int(raw.get("num_classes")),
        architecture_id=str(raw["architecture_id"]) if raw.get("architecture_id") else None,
        preprocess_consume=consume,
        info=info,
    )


def load_catalog(
    *,
    configs_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> Catalog:
    root = Path(configs_dir) if configs_dir is not None else CONFIGS_DIR
    path = Path(catalog_path) if catalog_path is not None else root / CATALOG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No catalog at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Catalog is not a mapping: {path}")
    anatomies_raw = raw.get("anatomies")
    if not isinstance(anatomies_raw, list) or not anatomies_raw:
        raise ValueError(f"Catalog {path} has no anatomies")

    anatomies: list[AnatomyEntry] = []
    for anatomy_raw in anatomies_raw:
        if not isinstance(anatomy_raw, dict):
            continue
        anatomy_id = str(anatomy_raw.get("id") or "").strip()
        if not anatomy_id:
            raise ValueError("Anatomy entry is missing id")
        anatomy_name = str(anatomy_raw.get("name") or anatomy_id)
        modalities_raw = anatomy_raw.get("modalities") or []
        parsed_modalities: list[ModalityEntry] = []
        if isinstance(modalities_raw, list):
            for modality_raw in modalities_raw:
                if not isinstance(modality_raw, dict):
                    continue
                modality_id = str(modality_raw.get("id") or "").strip()
                if not modality_id:
                    raise ValueError(f"Modality under {anatomy_id!r} is missing id")
                modality_name = str(modality_raw.get("name") or modality_id)
                tasks: list[TaskEntry] = []
                tasks_raw = modality_raw.get("tasks") or []
                if isinstance(tasks_raw, list):
                    for task_raw in tasks_raw:
                        if not isinstance(task_raw, dict):
                            continue
                        task_id = str(task_raw.get("id") or "").strip()
                        if not task_id:
                            raise ValueError(f"Task under {anatomy_id}/{modality_id} is missing id")
                        task_name = str(task_raw.get("name") or task_id)
                        tools: list[ToolEntry] = []
                        tools_raw = task_raw.get("tools") or []
                        if isinstance(tools_raw, list):
                            for tool_raw in tools_raw:
                                if not isinstance(tool_raw, dict):
                                    continue
                                tools.append(
                                    _build_tool(
                                        tool_raw,
                                        configs_dir=root,
                                        task_id=task_id,
                                        task_name=task_name,
                                        anatomy_id=anatomy_id,
                                        anatomy_name=anatomy_name,
                                        modality_id=modality_id,
                                        modality_name=modality_name,
                                    )
                                )
                        tasks.append(TaskEntry(task_id=task_id, task_name=task_name, tools=tools))
                parsed_modalities.append(
                    ModalityEntry(
                        modality_id=modality_id,
                        modality_name=modality_name,
                        tasks=tasks,
                    )
                )
        anatomies.append(
            AnatomyEntry(
                anatomy_id=anatomy_id,
                anatomy_name=anatomy_name,
                modalities=_ensure_modalities(parsed_modalities),
            )
        )
    return Catalog(anatomies=anatomies)


def list_tools(
    *,
    configs_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> list[ToolEntry]:
    """Flattened tools from the catalog (UI index, not a glob of configs/)."""
    return load_catalog(configs_dir=configs_dir, catalog_path=catalog_path).all_tools()
