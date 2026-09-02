# arth_tools

Scripts for **exporting**, **splitting**, **training**, and **evaluating** arthritis imaging models. This is not an agent.

Remote: [github.com/robdima96/arth_tools](https://github.com/robdima96/arth_tools)

Edit **`arth_tools/training/config.py`** (the CONTROL BOARD) for paths, toggles, and hyperparameters. Image data and checkpoints default to `E:\ArthAgent`. DICOM trees default to `E:\DICOMwrapper`.

## Install

From the repository root:

```text
pip install -e .
```

Or:

```text
pip install -r requirements.txt
```

Then keep this folder on `PYTHONPATH`, or use the editable install above.

Optional Hugging Face ResNet-50:

```text
pip install -e ".[resnet]"
```

## Tools

```text
python -m arth_tools export
python -m arth_tools split --manifest E:\ArthAgent\manifests\manifest_master.csv
python -m arth_tools train
python -m arth_tools eval --run-dir reporting/training/<run_id>
python -m arth_tools hpo
python -m arth_tools smoke
```

After `pip install -e .` the same commands work as `arth-tools export`, etc.

Each tool is also a module (`python -m arth_tools.data.export`, `arth_tools.training.train`, `arth_tools.training.evaluate`).

## Pipeline

1. **Export** — DICOM tree or pickle → PNG + master manifest
2. **Split** — patient-id grouped train/val/test (no patient in two splits)
3. **Train** — explicit VGG-style CNN (`build_cnn_model`) or optional `ARCHITECTURE_ID = "resnet50"`
4. **Eval** — held-out test using that run's `config_snapshot.yaml`, `zscore.json`, and best checkpoint

Train and eval share `arth_tools/training/metrics.py`. Eval does **not** use whatever is currently on the control board unless you pass `--use-control-board`.

After fit, test scoring writes `reporting/training/<run_id>/eval_test/` (`metrics.json`, `predictions.csv`, confusion matrix, ROC when binary). Metrics are reported at **image** and **patient** level (mean scores per patient).

Weights also copy to `E:\ArthAgent\checkpoints` when that drive is present. A `frozen/` copy is written only if `val_<primary_metric>` meets the freeze threshold.

`python -m arth_tools smoke` uses synthetic fixtures under `fixtures/smoke/` so it does not need the E: disk.

## Layout

| Path | Role |
| --- | --- |
| `arth_tools/training/config.py` | CONTROL BOARD — edit this |
| `arth_tools/training/backbones.py` | `build_cnn_model()` |
| `arth_tools/training/train.py` | Fit loop |
| `arth_tools/training/evaluate.py` | Held-out eval |
| `arth_tools/training/metrics.py` | Shared scoring |
| `arth_tools/data/` | DICOM load, PNG export, splits, dataset |
| `reporting/training/` | Per-run reports (generated; gitignored except README) |
| `fixtures/default_training.yaml` | Optional YAML overlay |

## Git

```text
git init -b main
git remote add origin https://github.com/robdima96/arth_tools.git
git add -A
git status
git commit -m "Initial arth_tools package."
git push -u origin main
```

`.gitignore` keeps smoke outputs, run reports, weights, pickles, DICOM pixels, and local research folders (`coding examples/`, `lit review/`, `prompt_history/`) off the remote.
