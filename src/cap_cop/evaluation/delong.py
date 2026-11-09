# Statistical functions used by the final locked-prediction evaluations.
from __future__ import annotations

import csv

import hashlib

import math

import os

from pathlib import Path

import numpy as np

import pandas as pd

def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks

def delong_auc_covariance(predictions: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return DeLong AUC estimates and covariance for correlated predictors."""
    order = np.argsort(-labels, kind="mergesort")
    labels_sorted = labels[order]
    predictions_sorted = predictions[:, order]
    positive_count = int(labels_sorted.sum())
    negative_count = int(labels_sorted.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("DeLong comparison requires both outcome classes")

    positive = predictions_sorted[:, :positive_count]
    negative = predictions_sorted[:, positive_count:]
    tx = np.vstack([midrank(row) for row in positive])
    ty = np.vstack([midrank(row) for row in negative])
    tz = np.vstack([midrank(row) for row in predictions_sorted])

    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)

    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.cov(v01, bias=False)
    sy = np.cov(v10, bias=False)
    if np.ndim(sx) == 0:
        sx = np.array([[float(sx)]])
    if np.ndim(sy) == 0:
        sy = np.array([[float(sy)]])
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance

def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))

def compare(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, float | int | str]:
    aucs, covariance = delong_auc_covariance(np.vstack([first, second]), labels)
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if variance <= 0.0:
        raise ValueError(f"Non-positive DeLong variance: {variance}")
    z = difference / math.sqrt(variance)
    return {
        "n": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
        "auc_first": float(aucs[0]),
        "auc_second": float(aucs[1]),
        "difference_first_minus_second": difference,
        "delong_z": z,
        "p_value_two_sided": normal_two_sided_p(z),
    }

def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()
