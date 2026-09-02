# -*- coding: utf-8 -*-
"""Reload a trained classifier from a run directory or a frozen bundle."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from arth_tools.data.preprocess import PreprocessRecipe
from arth_tools.models.spec import dump_spec, spec_from_training_config
from arth_tools.training.backbones import build_model
from arth_tools.training.config import TrainingConfig, load_config_yaml

BUNDLE_SCHEMA = "arth_tools.bundle.v1"


@dataclass
class LoadedClassifier:
    model: nn.Module
    cfg: TrainingConfig
    recipe: PreprocessRecipe
    label_map: dict[str, int]
    bundle_dir: Path
    source_dir: Path
    weights_path: Path

    def class_names(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for raw, idx in self.label_map.items():
            out.setdefault(int(idx), str(raw))
        return out


def extract_state_dict(checkpoint: Path) -> dict[str, Any]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob):
        return blob
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint}")


def load_weights(model: nn.Module, checkpoint: Path) -> None:
    model.load_state_dict(extract_state_dict(checkpoint))


def resolve_bundle_dir(source: Path) -> Path:
    source = Path(source)
    if (source / "bundle.json").is_file():
        return source
    if (source / "bundle" / "bundle.json").is_file():
        return source / "bundle"
    if (source / "frozen" / "bundle.json").is_file():
        return source / "frozen"
    return source


def _find_weights(bundle_dir: Path, source_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for cand in (
        bundle_dir / "state_dict.pt",
        bundle_dir / "model.pt",
        source_dir / "bundle" / "state_dict.pt",
        source_dir / "checkpoint_best.pt",
        source_dir / "frozen" / "model.pt",
        source_dir / "frozen" / "state_dict.pt",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No weights under {source_dir} or {bundle_dir}")


def _read_label_map(path: Path) -> tuple[dict[str, int], int | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("map"), dict):
        mapping = {str(k): int(v) for k, v in raw["map"].items()}
        n = raw.get("num_classes")
        n_int = int(n) if n is not None else (len(set(mapping.values())) if mapping else None)
        return mapping, n_int
    if isinstance(raw, dict):
        mapping = {str(k): int(v) for k, v in raw.items() if not isinstance(v, (dict, list))}
        return mapping, (len(set(mapping.values())) if mapping else None)
    raise ValueError(f"Unrecognized label map: {path}")


def _label_map_from_cfg(cfg: TrainingConfig, bundle_dir: Path, source_dir: Path) -> tuple[dict[str, int], int | None]:
    candidates: list[Path] = [bundle_dir / "label_map.json", source_dir / "label_map.json"]
    if cfg.train_manifest:
        candidates.append(Path(cfg.train_manifest).parent / "label_map.json")
    if cfg.label_map_path:
        candidates.append(Path(cfg.label_map_path))
    seen: set[Path] = set()
    for cand in candidates:
        key = cand.resolve() if cand.exists() else cand
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return _read_label_map(cand)
    if cfg.label_map:
        return dict(cfg.label_map), int(cfg.num_classes)
    return {}, None


def _recipe_from_disk(bundle_dir: Path, source_dir: Path, cfg: TrainingConfig) -> PreprocessRecipe:
    for cand in (bundle_dir / "preprocess.json", source_dir / "bundle" / "preprocess.json"):
        if cand.is_file():
            return PreprocessRecipe.from_dict(json.loads(cand.read_text(encoding="utf-8")))
    zscore = None
    for zpath in (bundle_dir / "zscore.json", source_dir / "zscore.json"):
        if not zpath.is_file():
            continue
        raw = json.loads(zpath.read_text(encoding="utf-8"))
        mean, std = raw.get("mean"), raw.get("std")
        if mean is None or std is None:
            continue
        if float(std) < 1e-6:
            break
        zscore = (float(mean), float(std))
        break
    return PreprocessRecipe.from_training_config(cfg, zscore=zscore)


def load_classifier(
    source: Path,
    *,
    checkpoint: Path | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedClassifier:
    """Load model + preprocess + label map from a run dir or a bundle directory."""
    source_dir = Path(source)
    bundle_dir = resolve_bundle_dir(source_dir)
    snap = bundle_dir / "config_snapshot.yaml"
    if not snap.is_file():
        snap = source_dir / "config_snapshot.yaml"
    if not snap.is_file():
        raise FileNotFoundError(f"No config_snapshot.yaml in {source_dir} or {bundle_dir}")
    cfg = load_config_yaml(snap)
    mapping, n_classes = _label_map_from_cfg(cfg, bundle_dir, source_dir)
    if mapping:
        cfg.label_map = mapping
    if n_classes is not None:
        cfg.num_classes = int(n_classes)
    recipe = _recipe_from_disk(bundle_dir, source_dir, cfg)
    if recipe.zscore_mean is not None and recipe.zscore_std is not None:
        cfg.train_zscore = True
    model = build_model(cfg)
    weights = _find_weights(bundle_dir, source_dir, checkpoint)
    load_weights(model, weights)
    model.to(map_location)
    model.eval()
    return LoadedClassifier(
        model=model,
        cfg=cfg,
        recipe=recipe,
        label_map=mapping,
        bundle_dir=bundle_dir,
        source_dir=source_dir,
        weights_path=weights,
    )


def write_run_bundle(
    run_dir: Path,
    cfg: TrainingConfig,
    *,
    checkpoint: Path | None = None,
    model: nn.Module | None = None,
    dest: Path | None = None,
) -> Path:
    """Write a self-contained bundle (spec, preprocess, labels, weights)."""
    run_dir = Path(run_dir)
    dest = Path(dest) if dest is not None else run_dir / "bundle"
    dest.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None and Path(checkpoint).is_file():
        state = extract_state_dict(checkpoint)
        torch.save(state, dest / "state_dict.pt")
        shutil.copy2(checkpoint, dest / "model.pt")
    elif model is not None:
        torch.save(model.state_dict(), dest / "state_dict.pt")
    else:
        raise FileNotFoundError(f"Cannot write bundle for {run_dir}: no checkpoint or model")

    snap_src = run_dir / "config_snapshot.yaml"
    if snap_src.is_file():
        shutil.copy2(snap_src, dest / "config_snapshot.yaml")
    else:
        (dest / "config_snapshot.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

    spec_src = run_dir / "model_spec.yaml"
    if spec_src.is_file():
        shutil.copy2(spec_src, dest / "model_spec.yaml")
    else:
        dump_spec(spec_from_training_config(cfg), dest / "model_spec.yaml")

    zscore = None
    zsrc = run_dir / "zscore.json"
    if zsrc.is_file():
        shutil.copy2(zsrc, dest / "zscore.json")
        raw = json.loads(zsrc.read_text(encoding="utf-8"))
        mean, std = raw.get("mean"), raw.get("std")
        if mean is not None and std is not None and float(std) >= 1e-6:
            zscore = (float(mean), float(std))

    recipe = PreprocessRecipe.from_training_config(cfg, zscore=zscore)
    (dest / "preprocess.json").write_text(json.dumps(recipe.to_dict(), indent=2), encoding="utf-8")

    mapping, n_classes = _label_map_from_cfg(cfg, dest, run_dir)
    if not mapping and cfg.label_map:
        mapping = dict(cfg.label_map)
        n_classes = int(cfg.num_classes)
    sidecar = {
        "source": "bundle",
        "tag": None,
        "map": mapping,
        "num_classes": int(n_classes if n_classes is not None else cfg.num_classes),
    }
    copied_map = False
    for cand in (
        Path(cfg.train_manifest).parent / "label_map.json" if cfg.train_manifest else None,
        Path(cfg.label_map_path) if cfg.label_map_path else None,
        run_dir / "label_map.json",
    ):
        if cand is not None and cand.is_file() and cand.resolve() != (dest / "label_map.json").resolve():
            shutil.copy2(cand, dest / "label_map.json")
            copied_map = True
            break
    if not copied_map:
        (dest / "label_map.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    bundle_meta = {
        "schema": BUNDLE_SCHEMA,
        "architecture_id": cfg.architecture_id,
        "task_id": cfg.task_id,
        "num_classes": int(sidecar["num_classes"]),
        "files": {
            "weights": "state_dict.pt",
            "spec": "model_spec.yaml",
            "config": "config_snapshot.yaml",
            "preprocess": "preprocess.json",
            "label_map": "label_map.json",
            "zscore": "zscore.json" if zscore is not None else None,
        },
    }
    (dest / "bundle.json").write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")
    return dest
