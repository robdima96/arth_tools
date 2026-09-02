# arth_tools

Local tools for **knee osteoarthritis imaging** research. Point the library at a folder of **DICOM** files to prepare data, train or evaluate a classifier, run unlabeled inference, measure the hip–knee–ankle (HKA) angle, or browse the same tools in a Streamlit UI.

Repository: [github.com/robdima96/arth_tools](https://github.com/robdima96/arth_tools)

## Disclaimer

This is research software. It is **not** a medical device and is **not** intended for clinical diagnosis or treatment decisions. You supply the DICOMs; the library reads and writes files on the machine where you run it and does not upload studies.

## Install

Python 3.10 or newer. From the repository root:

```text
pip install -e .
```

Optional extras:

```text
pip install -e ".[ui]"       # Streamlit library
pip install -e ".[dev]"      # pytest
pip install -e ".[resnet]"   # Hugging Face ResNet-50
```

`requirements.txt` lists the same core dependencies if you prefer that install path.

## User interface

```text
python -m arth_tools ui
```

The library is a tree: **anatomy → modality → task → tool** (`configs/catalog.yaml`). Knee always lists X-ray, Ultrasound, and MRI; empty slots say there are no tasks yet.

Each tool has **Infer** and **Info**. **Train** is offered only on native CNN tools (`cnn` or `resnet50` from the architecture registry). Infer uses that tool’s backend: a saved run directory, a frozen bundle, the built-in HKA measurement, or a disabled stub when a literature pipeline is not wired yet.

Infer and Train require **Load preview** first (a random sample of up to 10 decoded images). Train also shows the same files after the preprocess recipe.

## Command line

After `pip install -e .` these also work as `arth-tools prepare`, and so on.

```text
python -m arth_tools prepare --config kl_grade --dicom-root /path/to/xrays
python -m arth_tools train --config kl_grade
python -m arth_tools train --config kl_grade --architecture resnet50
python -m arth_tools infer --run-dir reporting/training/<run_id> --dicom-root /path/to/new
python -m arth_tools infer --bundle data/kl_grade/frozen/<run> --images /path/to/pngs
python -m arth_tools hka --config hka --dicom-root /path/to/longleg
python -m arth_tools eval --run-dir reporting/training/<run_id>
python -m arth_tools hpo --config omeract_synovitis
python -m arth_tools ui
python -m arth_tools smoke
```

`prepare` exports PNGs and writes a patient-grouped train/val/test split. `export` and `split` remain available as lower-level commands.

**HKA** measures the mechanical axis on a standing AP long-leg radiograph (`CR` / `DX` / `RF`). Output is an annotated DICOM plus `hka_angles.csv` / `hka_angles.json`. 180° is a straight axis; negative deviation is varus, positive is valgus. Landmarking is classical, not a trained detector.

### Labels

Patient IDs come from `PatientID`, else `StudyInstanceUID`, else the parent folder. Class labels come from a DICOM tag (`StudyDescription` by default), class folders (`dicoms/0/...`), or a CSV joined on `patient_id` or `study_uid`.

```yaml
label_source: csv
label_csv: data/kl_grade/kl_grades.csv
label_csv_join: patient_id
label_csv_column: label
```

## Tasks

Recipe YAMLs under `configs/` overlay library defaults in `arth_tools/training/config.py`. `--config kl_grade` is enough; relative paths resolve from the repo root.

| Recipe | Role | Modality filter |
| --- | --- | --- |
| `configs/kl_grade.yaml` | Kellgren–Lawrence 0–4 | `CR`, `DX`, `RF` |
| `configs/omeract_synovitis.yaml` | OMERACT–EULAR 0–3 | `US` |
| `configs/hka.yaml` | Hip–knee–ankle angle | `CR`, `DX`, `RF` |

`configs/catalog.yaml` is the UI index (multiple tools can sit under one task). CLI recipes are unchanged.

Default folders when you omit `--config`: DICOMs in `<repo>/dicoms` (`ARTH_DICOM_ROOT`), derived files in `<repo>/data` (`ARTH_DATA_ROOT`).

## Pipeline

1. **Prepare** — DICOMs → PNG + master manifest → train/val/test CSVs (no patient in two splits).
2. **Train** — `cnn` or `resnet50` via the architecture registry; writes a reloadable `bundle/`.
3. **Eval** — held-out test from the run snapshot or bundle (not live defaults, unless `--use-control-board`).
4. **Infer** — unlabeled DICOM or PNG folder → `predictions.csv`.

Train sizes the classification head from `label_map.json` written by prepare. Metrics are reported at image and patient level.

## Smoke test

```text
python -m arth_tools smoke
```

This generates tiny synthetic DICOMs under `fixtures/smoke/`, then runs prepare → train → eval → infer and a short HKA check. Those outputs are gitignored.

```text
pip install -e ".[dev]"
pytest
```

## Layout

| Path | Role |
| --- | --- |
| `arth_tools/` | Package (data, training, inference, HKA, UI) |
| `arth_tools/training/config.py` | Library defaults |
| `configs/` | Task recipes and `catalog.yaml` |
| `tests/` | Unit tests |
| `fixtures/default_training.yaml` | Optional YAML overlay |
| `reporting/training/` | Per-run reports (generated; README only in git) |

## License

MIT. Third-party model weights cited in the UI (for example Gu et al. 2022) are **not** included. Those authors license their code and weights separately (CC BY-NC-SA 4.0); download them from the paper repository if you need them.
