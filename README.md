# arth_tools

Local tools for **knee osteoarthritis imaging** research. Point the library at a folder of **DICOM** files to prepare data, train or evaluate a classifier, run unlabeled inference, or browse the same tools in a Streamlit UI.

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


