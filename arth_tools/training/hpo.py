"""Optional Optuna HPO wrapping the trainer.

Search space lives on the control board (HPO_SEARCH). Each trial copies the
TrainingConfig, flips hpo.enabled off, and calls train().
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from arth_tools.training.config import TrainingConfig, load_config_yaml
from arth_tools.training.train import train


def _suggest(trial: Any, search: dict[str, Any], cfg: TrainingConfig) -> TrainingConfig:
    trial_cfg = replace(cfg, hpo=deepcopy(cfg.hpo))
    trial_cfg.hpo.enabled = False
    trial_cfg.run_id = f"{cfg.run_id or 'hpo'}_trial{trial.number:03d}"

    lr_spec = search.get("learning_rate")
    if isinstance(lr_spec, dict) and "low" in lr_spec:
        trial_cfg.learning_rate = trial.suggest_float(
            "learning_rate",
            float(lr_spec["low"]),
            float(lr_spec["high"]),
            log=bool(lr_spec.get("log", True)),
        )
    bs_spec = search.get("batch_size")
    if isinstance(bs_spec, dict) and "choices" in bs_spec:
        trial_cfg.batch_size = trial.suggest_categorical("batch_size", list(bs_spec["choices"]))
    return trial_cfg


def run_hpo(cfg: TrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainingConfig()
    if not cfg.hpo.enabled:
        return train(cfg)

    import optuna

    direction = "maximize" if cfg.freeze_mode == "max" else "minimize"

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = _suggest(trial, cfg.hpo.search, cfg)
        result = train(trial_cfg)
        metric = result.get("best_metric")
        if metric is None:
            raise optuna.TrialPruned("no metric")
        return float(metric)

    study = optuna.create_study(direction=direction)
    study.optimize(
        objective,
        n_trials=cfg.hpo.n_trials,
        timeout=cfg.hpo.timeout_s,
    )
    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "direction": direction,
    }
    report_root = Path(cfg.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    dest = report_root / f"{cfg.run_id or 'hpo'}_study.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["study_path"] = str(dest)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Optuna HPO (or a single train if hpo.enabled is false).")
    ap.add_argument("--config", type=Path, help="Task YAML (configs/kl_grade.yaml or kl_grade)")
    args = ap.parse_args(argv)
    cfg = load_config_yaml(args.config) if args.config else TrainingConfig()
    print(json.dumps(run_hpo(cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
