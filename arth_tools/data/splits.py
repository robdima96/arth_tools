# -*- coding: utf-8 -*-
"""
Patient-id grouped train/val/test splits with a leakage report.

Matches the IPFP control-board toggles:
  SPLIT_FORCE_PT_STRAT — no patient appears in more than one split
  SPLIT_FORCE_EQ_DIST  — shuffle within label buckets so class mix is roughly even
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from arth_tools.training.config import (
    LABEL_COLUMN,
    MANIFEST_DIR,
    PATIENT_ID_COLUMN,
    SEED,
    SPLIT_FORCE_EQ_DIST,
    SPLIT_FORCE_PT_STRAT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

PATIENT_COL = PATIENT_ID_COLUMN
LABEL_COL = LABEL_COLUMN


class SplitError(ValueError):
    pass


def _patient_label(df: pd.DataFrame, patient_col: str, label_col: str) -> pd.Series:
    """One label per patient (first row). Used only when stratify=True."""
    return df.groupby(patient_col, sort=False)[label_col].first()


def _assign_from_groups(
    df: pd.DataFrame,
    patient_col: str,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    for pid in train_ids:
        mapping[pid] = "train"
    for pid in val_ids:
        mapping[pid] = "val"
    for pid in test_ids:
        mapping[pid] = "test"
    out["split"] = out[patient_col].map(mapping)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), patient_col].unique().tolist()
        raise SplitError(f"Patients missing a split assignment: {missing}")
    return out


def leakage_report(df: pd.DataFrame, patient_col: str = PATIENT_COL) -> dict[str, Any]:
    by_split: dict[str, set[str]] = {}
    for split, sub in df.groupby("split"):
        by_split[str(split)] = set(sub[patient_col].astype(str))
    overlaps: dict[str, list[str]] = {}
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(by_split[a] & by_split[b])
            if shared:
                overlaps[f"{a}&{b}"] = shared
    return {
        "n_rows": int(len(df)),
        "n_patients": {k: len(v) for k, v in by_split.items()},
        "n_rows_per_split": df["split"].value_counts().to_dict(),
        "leaked_patients": overlaps,
        "ok": not overlaps,
    }


def split_manifest_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = PATIENT_COL,
    label_col: str = LABEL_COL,
    train_frac: float = TRAIN_SPLIT,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
    seed: int = SEED,
    stratify: bool | None = None,
    split_force_pt_strat: bool | None = None,
    split_force_eq_dist: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("\n" + "=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)

    use_patient = SPLIT_FORCE_PT_STRAT if split_force_pt_strat is None else split_force_pt_strat
    use_eq = SPLIT_FORCE_EQ_DIST if split_force_eq_dist is None else split_force_eq_dist
    if stratify is not None:
        use_eq = bool(stratify)

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        print(f"Warning: Split sizes sum to {total:.3f}, normalizing to 1.0")
        train_frac, val_frac, test_frac = train_frac / total, val_frac / total, test_frac / total

    if patient_col not in df.columns:
        raise SplitError(f"Missing patient column {patient_col!r}")

    work = df.copy()
    work[patient_col] = work[patient_col].astype(str)
    patients = work[patient_col].unique()
    n = len(patients)
    print(f"Total samples: {len(work)}")
    print(f"Target split: Train={train_frac:.2%}, Val={val_frac:.2%}, Test={test_frac:.2%}")
    print(f"Patient-level stratification: {'ON' if use_patient else 'OFF'}")
    print(f"Equal distribution: {'ON' if use_eq else 'OFF'}")

    if use_patient and n < 3:
        raise SplitError(f"Need at least 3 patients to form train/val/test; got {n}")

    rng = np.random.RandomState(seed)

    if use_patient and use_eq and label_col in work.columns:
        print(f"Unique patients: {n}")
        print("Active stratification: Patient-level + Equal distribution")
        y = _patient_label(work, patient_col, label_col).reindex(patients)
        buckets: dict[str, list[str]] = {}
        for pid, lab in y.items():
            buckets.setdefault(str(lab), []).append(str(pid))
        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []
        for pids in buckets.values():
            rng.shuffle(pids)
            n_tr = int(round(train_frac * len(pids)))
            if len(pids) >= 3:
                n_tr = min(max(1, n_tr), len(pids) - 2)
            n_va = max(0, int(round(val_frac * len(pids))))
            if n_tr + n_va >= len(pids):
                n_va = max(0, len(pids) - n_tr - 1)
            train_ids.extend(pids[:n_tr])
            val_ids.extend(pids[n_tr : n_tr + n_va])
            test_ids.extend(pids[n_tr + n_va :])
        assigned = _assign_from_groups(work, patient_col, set(train_ids), set(val_ids), set(test_ids))
        method = "patient_label_buckets"

    elif use_patient:
        print(f"Unique patients: {n}")
        print("Active stratification: Patient-level only")
        unique_patients = np.array(patients)
        train_patients, temp_patients = train_test_split(
            unique_patients, train_size=train_frac, random_state=seed, shuffle=True
        )
        val_ratio = val_frac / (val_frac + test_frac)
        val_patients, test_patients = train_test_split(
            temp_patients, train_size=val_ratio, random_state=seed, shuffle=True
        )
        assigned = _assign_from_groups(
            work,
            patient_col,
            set(map(str, train_patients)),
            set(map(str, val_patients)),
            set(map(str, test_patients)),
        )
        method = "train_test_split_by_patient_id"

        train_set = set(map(str, train_patients))
        val_set = set(map(str, val_patients))
        test_set = set(map(str, test_patients))
        assert not (train_set & val_set), "Patient leakage: train-val overlap"
        assert not (train_set & test_set), "Patient leakage: train-test overlap"
        assert not (val_set & test_set), "Patient leakage: val-test overlap"
        print("No patient leakage detected")

    else:
        print("Active stratification: Sample-level (no patient grouping)")
        idx = np.arange(len(work))
        try:
            strat = work[label_col] if (use_eq and label_col in work.columns) else None
            train_idx, temp_idx = train_test_split(
                idx, train_size=train_frac, random_state=seed, shuffle=True, stratify=strat
            )
            temp_strat = work.iloc[temp_idx][label_col] if strat is not None else None
            val_ratio = val_frac / (val_frac + test_frac)
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_ratio, random_state=seed, shuffle=True, stratify=temp_strat
            )
        except ValueError as exc:
            print(f"Equal-distribution split failed ({exc}); falling back to random.")
            train_idx, temp_idx = train_test_split(
                idx, train_size=train_frac, random_state=seed, shuffle=True
            )
            val_ratio = val_frac / (val_frac + test_frac)
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_ratio, random_state=seed, shuffle=True
            )
        assigned = work.copy()
        assigned["split"] = ""
        assigned.iloc[train_idx, assigned.columns.get_loc("split")] = "train"
        assigned.iloc[val_idx, assigned.columns.get_loc("split")] = "val"
        assigned.iloc[test_idx, assigned.columns.get_loc("split")] = "test"
        method = "sample_level_train_test_split"

    report = leakage_report(assigned, patient_col)
    report["fractions"] = {"train": train_frac, "val": val_frac, "test": test_frac}
    report["seed"] = seed
    report["stratify"] = use_eq
    report["method"] = method

    print("\nSplit results:")
    print(f"  Training: {report['n_patients'].get('train', 0)} patients ({report['n_rows_per_split'].get('train', 0)} samples)")
    print(f"  Validation: {report['n_patients'].get('val', 0)} patients ({report['n_rows_per_split'].get('val', 0)} samples)")
    print(f"  Test: {report['n_patients'].get('test', 0)} patients ({report['n_rows_per_split'].get('test', 0)} samples)")

    if use_patient and not report["ok"]:
        raise SplitError(f"Patient leakage across splits: {report['leaked_patients']}")
    return assigned, report


def write_split_manifests(
    assigned: pd.DataFrame,
    out_dir: Path,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        dest = out_dir / f"manifest_{split}.csv"
        assigned.loc[assigned["split"] == split].drop(columns=["split"]).to_csv(dest, index=False)
        paths[split] = dest
        print(f"  {int((assigned['split'] == split).sum())} rows -> {dest}")
    if report is not None:
        dest = out_dir / "split_report.json"
        dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        paths["report"] = dest
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Split a master manifest by patient_id.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--train", type=float, default=TRAIN_SPLIT)
    ap.add_argument("--val", type=float, default=VAL_SPLIT)
    ap.add_argument("--test", type=float, default=TEST_SPLIT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--stratify", action="store_true", default=SPLIT_FORCE_EQ_DIST)
    ap.add_argument("--no-patient-strat", action="store_true")
    ap.add_argument("--patient-col", default=PATIENT_COL)
    ap.add_argument("--label-col", default=LABEL_COL)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.manifest)
    assigned, report = split_manifest_by_patient(
        df,
        patient_col=args.patient_col,
        label_col=args.label_col,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
        split_force_eq_dist=bool(args.stratify),
        split_force_pt_strat=not args.no_patient_strat,
    )
    paths = write_split_manifests(assigned, args.out_dir, report=report)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
