# Training reports

Auto-written per-run reports from `python -m arth_tools train` / `eval`.
Do not hand-author results here.

Edit the CONTROL BOARD in `arth_tools/training/config.py` before a real run.
Weights belong on `E:\ArthAgent`; this folder is the paper trail.

Each run writes `reporting/training/<run_id>/` with:

- `hyperparameters.txt` — flat key: value dump
- `model_details.txt` — layer list
- `config_snapshot.yaml` — resolved TrainingConfig (eval reloads this)
- `model_spec.yaml` — architecture / freeze criteria
- `environment.json` — library versions
- `epochs.csv`, `curves_loss.png`, `curves_metric.png`
- `zscore.json` — training-set pixel mean/std when z-score is on
- `best_epoch.json`, `checkpoint_best.pt`
- `eval_test/` — `metrics.json`, `predictions.csv`, confusion matrix / ROC
- `frozen/` — only when freeze criteria are met
- `freeze_decision.json`
- `train.log`
