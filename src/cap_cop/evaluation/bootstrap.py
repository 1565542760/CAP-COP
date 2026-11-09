# Statistical functions used by the final locked-prediction evaluations.
from __future__ import annotations

import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")

def metric_values(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, probability)),
        "AUPRC": float(average_precision_score(y_true, probability)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": specificity(y_true, predicted),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
    }

def bootstrap_metric_ci(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    point = metric_values(y_true, probability)
    values = {name: [] for name in point}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[indices]
        sample_p = probability[indices]
        if np.unique(sample_y).size < 2:
            continue
        sampled = metric_values(sample_y, sample_p)
        for name, value in sampled.items():
            if np.isfinite(value):
                values[name].append(value)
    intervals = {}
    for name, samples in values.items():
        if not samples:
            intervals[f"{name}_ci_low"] = float("nan")
            intervals[f"{name}_ci_high"] = float("nan")
        else:
            intervals[f"{name}_ci_low"] = float(np.percentile(samples, 2.5))
            intervals[f"{name}_ci_high"] = float(np.percentile(samples, 97.5))
    return point, intervals

def bootstrap_delta_ci(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    point_first = metric_values(y_true, first)
    point_second = metric_values(y_true, second)
    point = {name: point_first[name] - point_second[name] for name in point_first}
    values = {name: [] for name in point}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[indices]
        if np.unique(sample_y).size < 2:
            continue
        first_metrics = metric_values(sample_y, first[indices])
        second_metrics = metric_values(sample_y, second[indices])
        for name in values:
            delta = first_metrics[name] - second_metrics[name]
            if np.isfinite(delta):
                values[name].append(delta)
    intervals = {}
    for name, samples in values.items():
        intervals[f"{name}_ci_low"] = float(np.percentile(samples, 2.5))
        intervals[f"{name}_ci_high"] = float(np.percentile(samples, 97.5))
    return point, intervals
