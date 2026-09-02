# -*- coding: utf-8 -*-
"""
Shared scoring used by the trainer (epoch loops) and the evaluator (held-out test).

Keep this module free of I/O so train and eval cannot drift apart on how
logits become probabilities, predictions, or metric values.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from arth_tools.training.backbones import forward_logits


def batch_loss(loss_name: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if loss_name == "binary_crossentropy":
        return nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), targets.float().view(-1)
        )
    if loss_name == "mse":
        return nn.functional.mse_loss(logits.view(-1), targets.float().view(-1))
    return nn.functional.cross_entropy(logits, targets.long())


def scores_from_logits(logits: np.ndarray, loss_name: str) -> np.ndarray:
    t = torch.from_numpy(np.asarray(logits)).float()
    if loss_name == "binary_crossentropy":
        return torch.sigmoid(t.view(-1)).numpy().reshape(-1, 1)
    if loss_name == "mse":
        return np.asarray(logits).reshape(-1, 1)
    return nn.functional.softmax(t, dim=-1).numpy()


def predict_from_scores(scores: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    if scores.ndim == 2 and scores.shape[1] > 1:
        return scores.argmax(axis=1)
    return (scores.reshape(-1) >= threshold).astype(int)


def metric_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str,
    loss_name: str,
    *,
    threshold: float = 0.5,
) -> float:
    name_u = name.lower()
    y = np.asarray(y_true).ravel()
    if loss_name == "mse":
        pred = scores.reshape(-1)
        if name_u in {"mae"}:
            return float(-mean_absolute_error(y, pred))
        if name_u in {"r2"}:
            return float(r2_score(y, pred))
        return float(-np.sqrt(np.mean((pred - y) ** 2)))
    pred = predict_from_scores(scores, threshold=threshold)
    y_int = y.astype(int)
    if name_u in {"accuracy", ""}:
        return float(accuracy_score(y_int, pred))
    if name_u in {"f1", "macro_f1"}:
        return float(f1_score(y_int, pred, average="macro", zero_division=0))
    if name_u in {"precision"}:
        return float(precision_score(y_int, pred, average="macro", zero_division=0))
    if name_u in {"recall"}:
        return float(recall_score(y_int, pred, average="macro", zero_division=0))
    if name_u in {"auc", "roc_auc"}:
        if len(np.unique(y_int)) < 2:
            return float("nan")
        if scores.ndim == 2 and scores.shape[1] > 2:
            return float(roc_auc_score(y_int, scores, multi_class="ovo", average="macro"))
        if scores.ndim == 2 and scores.shape[1] == 2:
            return float(roc_auc_score(y_int, scores[:, 1]))
        return float(roc_auc_score(y_int, scores.reshape(-1)))
    return float(accuracy_score(y_int, pred))


def metric_suite(
    y_true: np.ndarray,
    scores: np.ndarray,
    loss_name: str,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y_true).ravel()
    if loss_name == "mse":
        pred = scores.reshape(-1)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return {
            "n": int(len(y)),
            "rmse": rmse,
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
            "primary": -rmse,
        }
    pred = predict_from_scores(scores, threshold=threshold)
    y_int = y.astype(int)
    labels = sorted(set(y_int.tolist()) | set(pred.tolist()))
    report = classification_report(
        y_int, pred, labels=labels, output_dict=True, zero_division=0
    )
    out: dict[str, Any] = {
        "n": int(len(y_int)),
        "accuracy": float(accuracy_score(y_int, pred)),
        "macro_f1": float(f1_score(y_int, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_int, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_int, pred, average="macro", zero_division=0)),
        "auc": metric_from_scores(y_int, scores, "auc", loss_name, threshold=threshold),
        "confusion_matrix": confusion_matrix(y_int, pred, labels=labels).tolist(),
        "labels": [int(x) for x in labels],
        "classification_report": report,
    }
    return out


def aggregate_by_patient(
    patient_ids: list[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    loss_name: str,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """One row per patient: first label, mean scores (majority via mean softmax)."""
    order: list[str] = []
    buckets: dict[str, list[int]] = {}
    for i, pid in enumerate(patient_ids):
        if pid not in buckets:
            buckets[pid] = []
            order.append(pid)
        buckets[pid].append(i)
    y_out = []
    s_out = []
    for pid in order:
        idx = buckets[pid]
        y_out.append(y_true[idx[0]])
        s_out.append(np.mean(scores[idx], axis=0))
    return np.asarray(y_out), np.vstack(s_out)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    primary_metric: str,
    *,
    threshold: float = 0.5,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = forward_logits(model, xb)
        loss = batch_loss(loss_name, logits, yb)
        losses.append(float(loss.cpu().item()))
        log_np = logits.detach().cpu().numpy()
        for i in range(yb.shape[0]):
            ys.append(np.array([yb[i].detach().cpu().item()]))
            rows.append(scores_from_logits(log_np[i : i + 1], loss_name))
    if not losses:
        return float("nan"), float("nan")
    y_np = np.vstack(ys).ravel()
    score_np = np.vstack(rows)
    return (
        float(statistics.mean(losses)),
        metric_from_scores(y_np, score_np, primary_metric, loss_name, threshold=threshold),
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses: list[float] = []
    ys: list[float] = []
    score_rows: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = forward_logits(model, xb)
        loss = batch_loss(loss_name, logits, yb)
        losses.append(float(loss.cpu().item()))
        log_np = logits.detach().cpu().numpy()
        for i in range(yb.shape[0]):
            ys.append(float(yb[i].detach().cpu().item()))
            score_rows.append(scores_from_logits(log_np[i : i + 1], loss_name).reshape(-1))
    y_np = np.asarray(ys, dtype=np.float64)
    score_np = np.vstack(score_rows) if score_rows else np.zeros((0, 1))
    avg_loss = float(statistics.mean(losses)) if losses else float("nan")
    return y_np, score_np, avg_loss
